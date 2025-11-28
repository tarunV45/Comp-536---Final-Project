"""A CPU worker class."""
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.distributed

import vllm.envs as envs
from vllm.attention import get_attn_backend
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
                         ParallelConfig, VllmConfig)
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.sequence import ExecuteModelRequest
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.worker.cpu_enc_dec_model_runner import CPUEncoderDecoderModelRunner
from vllm.worker.cpu_model_runner import CPUModelRunner, CPUModelRunnerBase
from vllm.worker.cpu_pooling_model_runner import CPUPoolingModelRunner
from vllm.worker.worker_base import (LocalOrDistributedWorkerBase,
                                     LoraNotSupportedWorkerBase, WorkerBase,
                                     WorkerInput)
from typing import Optional

logger = init_logger(__name__)


class CPUCacheEngine:
    """Manages the KV cache for CPU backend.

    This class is responsible for initializing and managing CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as copying.
    """

    def __init__(self, cache_config: CacheConfig, model_config: ModelConfig,
                 parallel_config: ParallelConfig,
                 device_config: DeviceConfig) -> None:
        assert device_config.device_type == "cpu"
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        # Note: In CacheConfig, num_gpu_blocks actual is num_cpu_blocks
        # for CPU backend, because we want to reuse KV cache management
        # in the scheduler.
        self.num_cpu_blocks = cache_config.num_gpu_blocks

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Get attention backend.
        self.attn_backend = get_attn_backend(
            self.model_config.get_head_size(),
            self.model_config.dtype,
            cache_config.cache_dtype,
            self.block_size,
            self.model_config.is_attention_free,
        )

        # Initialize the cache.
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks)

    def _allocate_kv_cache(
        self,
        num_blocks: int,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on CPU."""
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_heads, self.head_size)
        kv_cache: List[torch.Tensor] = []
        for _ in range(self.num_layers):
            kv_cache.append(
                torch.empty(kv_cache_shape, dtype=self.dtype, device="cpu"))
        return kv_cache

    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        raise NotImplementedError("Swap is not supported in CPUCacheEngine.")

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        raise NotImplementedError("Swap is not supported in CPUCacheEngine.")

    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        self.attn_backend.copy_blocks(self.cpu_cache, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        block_size: Optional [int],
        cache_dtype: str,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        if block_size is None:
            block_size=16
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        key_cache_block = block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        if cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        dtype_size = torch.tensor([], dtype=dtype).element_size()
        return dtype_size * total


class CPUWorker(LoraNotSupportedWorkerBase, LocalOrDistributedWorkerBase):
    """A worker class that executes (a partition of) the model on a CPU socket.

    Each worker is associated with a single CPU socket. The worker is 
    responsible for maintaining the KV cache and executing the model on the 
    CPU. In case of distributed inference, each worker is assigned a partition
    of the model.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        model_runner_cls: Optional[Type[CPUModelRunner]] = None,
    ) -> None:
        WorkerBase.__init__(self, vllm_config=vllm_config)

        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method

        self.is_driver_worker = is_driver_worker
        if self.is_driver_worker:
            assert self.rank == 0, "The driver worker must have rank 0."

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        # Setup OpenMP threads affinity.
        omp_cpuids = envs.VLLM_CPU_OMP_THREADS_BIND
        if omp_cpuids == "all":
            self.local_omp_cpuid = "all"
        else:
            self.local_omp_cpuid = omp_cpuids.split("|")[rank]

        # Return hidden states from target model if the draft model is an
        # mlp_speculator
        speculative_config = self.speculative_config
        model_config = self.model_config
        speculative_args = {} if speculative_config is None \
            or (speculative_config.draft_model_config.model ==
                model_config.model) \
            or (speculative_config.draft_model_config.hf_config.model_type
                not in ["medusa", "mlp_speculator", "eagle"]) \
                    else {"return_hidden_states": True}
        ModelRunnerClass: Type[CPUModelRunnerBase] = CPUModelRunner
        if self.model_config.runner_type == "pooling":
            ModelRunnerClass = CPUPoolingModelRunner
        elif self.model_config.is_encoder_decoder:
            ModelRunnerClass = CPUEncoderDecoderModelRunner
        self.model_runner: CPUModelRunnerBase = ModelRunnerClass(
            vllm_config=vllm_config,
            kv_cache_dtype=kv_cache_dtype,
            is_driver_worker=is_driver_worker,
            **speculative_args,
        )
        if model_runner_cls is not None:
            self.model_runner = model_runner_cls(self.model_runner)
        # Uninitialized cache engine. Will be initialized by
        # initialize_cache.
        self.cache_engine: List[CPUCacheEngine]
        # Initialize cpu_cache as pooling models don't initialize kv_caches
        self.cpu_cache: Optional[List[List[torch.Tensor]]] = None

        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        if envs.VLLM_TORCH_PROFILER_DIR:
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info("Profiling enabled. Traces will be saved to: %s",
                        torch_profiler_trace_dir)
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                ],
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir, use_gzip=True))
        else:
            self.profiler = None

    def start_profile(self):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        self.profiler.start()

    def stop_profile(self):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        self.profiler.stop()

    def init_device(self) -> None:
        if self.local_omp_cpuid != "all":
            ret = torch.ops._C_utils.init_cpu_threads_env(self.local_omp_cpuid)
            if ret:
                logger.info(ret)
        self.device = torch.device("cpu")
        self.init_distributed_environment()
        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of blocks available for the KV cache.

        This is a CPU-only implementation used in our course project setup.

        In newer vLLM versions, CacheConfig may define a field like
        `cpu_kvcache_space_bytes`. In our snapshot, that field might not
        exist, so we fall back to a small fixed number of blocks instead
        of crashing.
        """
        cache_block_size = self.get_cache_block_size_bytes()

        # Try to read a CPU KV cache space field if it exists.
        cpu_space_bytes = getattr(self.cache_config,
                                  "cpu_kvcache_space_bytes",
                                  None)

        if cpu_space_bytes is not None and cpu_space_bytes > 0:
            num_cpu_blocks = int(cpu_space_bytes // cache_block_size)
        else:
            # Fallback for our Mac/UnspecifiedPlatform: just give the scheduler
            # a modest number of blocks so it can run.
            num_cpu_blocks = 128

        # vLLM convention: blocks that can be modified are treated as "GPU"
        # blocks, and CPU blocks are returned as 0 in this emulated CPU flow.
        num_gpu_blocks = num_cpu_blocks
        num_cpu_blocks = 0

        return num_gpu_blocks, num_cpu_blocks


    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the KV cache. Currently, swappable CPU memory is not
        supported.

        Since this worker does not support GPUs, we use the num_gpu_blocks to
        determine how many non-swappable CPU blocks to allocate.
        """
        assert (num_cpu_blocks == 0
                ), f"{type(self)} does not support swappable cache"

        # Note: To reuse the cache management procedure,
        # use cpu cache as 'gpu cache'.
        num_cpu_blocks = num_gpu_blocks

        self._validate_num_cpu_blocks(num_cpu_blocks)
        self.cache_config.num_gpu_blocks = num_cpu_blocks
        self.cache_config.num_cpu_blocks = 0

        # Initialize the cache.
        self._init_cache_engine()

    def _validate_num_cpu_blocks(self, num_cpu_blocks: int) -> None:
        """Raise errors if the num_cpu_blocks is invalid.

            In our course/Mac CPU setup, some cache_config fields like block_size
            may be None, so we avoid using them here and just enforce a simple
            non-zero check.
        """
        if num_cpu_blocks <= 0:
            raise ValueError(
                "No available memory for the cache blocks. "
                "Try increasing `VLLM_CPU_KVCACHE_SPACE` when "
                "initializing the engine."
            )

        # For our CPU-emulation environment, skip any further validation that
        # depends on cache_config.block_size.
        return


    def _init_cache_engine(self) -> None:
        self.cache_engine = [
            CPUCacheEngine(self.cache_config, self.model_config,
                           self.parallel_config, self.device_config)
            for _ in range(self.parallel_config.pipeline_parallel_size)
        ]
        self.cpu_cache = [
            self.cache_engine[ve].cpu_cache
            for ve in range(self.parallel_config.pipeline_parallel_size)
        ]
        self.model_runner.block_size = self.cache_engine[0].block_size

        assert all(
            self.cpu_cache[ve] is not None
            for ve in range(self.parallel_config.pipeline_parallel_size))

        # Populate the cache to warmup the memory
        for ve in range(self.parallel_config.pipeline_parallel_size):
            for layer_cache in self.cpu_cache[ve]:
                layer_cache.fill_(0)

    @property
    def do_metadata_broadcast(self) -> bool:
        return self.parallel_config.tensor_parallel_size > 1

    @property
    def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
        return self.cpu_cache

    @property
    def vocab_size(self) -> int:
        return self.model_runner.vocab_size

    @property
    def max_model_len(self) -> int:
        return self.model_config.max_model_len

    def execute_worker(
        self,
        worker_input: WorkerInput,
    ) -> None:
        if (worker_input.blocks_to_copy is not None
                and worker_input.blocks_to_copy.numel() > 0):
            self.cache_engine[worker_input.virtual_engine].copy(
                worker_input.blocks_to_copy)

    @torch.inference_mode()
    def prepare_worker_input(
            self, execute_model_req: ExecuteModelRequest) -> WorkerInput:
        assert execute_model_req is not None
        virtual_engine: int = execute_model_req.virtual_engine
        num_seq_groups: int = len(execute_model_req.seq_group_metadata_list)
        blocks_to_copy = torch.tensor(execute_model_req.blocks_to_copy,
                                      device="cpu",
                                      dtype=torch.int64).view(-1, 2)
        assert len(execute_model_req.blocks_to_swap_in) == 0
        assert len(execute_model_req.blocks_to_swap_out) == 0
        return WorkerInput(
            num_seq_groups=num_seq_groups,
            blocks_to_copy=blocks_to_copy,
            virtual_engine=virtual_engine,
        )

    def init_distributed_environment(self) -> None:
        """Initialize the distributed environment."""

        parallel_config = self.parallel_config
        rank = self.rank
        distributed_init_method = self.distributed_init_method
        init_distributed_environment(
            world_size=parallel_config.world_size,
            rank=rank,
            distributed_init_method=distributed_init_method,
            backend="gloo",
        )

        # A small all_reduce for warmup.
        torch.distributed.all_reduce(torch.zeros(1).cpu())

        ensure_model_parallel_initialized(
            parallel_config.tensor_parallel_size,
            parallel_config.pipeline_parallel_size)

    def get_cache_block_size_bytes(self) -> int:
        """Return the size in bytes of a single KV cache block.
        """
        return CPUCacheEngine.get_cache_block_size(
            self.cache_config.block_size, self.cache_config.cache_dtype,
            self.model_config, self.parallel_config)
