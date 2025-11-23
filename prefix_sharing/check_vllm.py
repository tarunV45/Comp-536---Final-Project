import vllm
from vllm.core.scheduler import Scheduler
from vllm.engine.llm_engine import LLMEngine

print("vLLM imported OK. Version:", getattr(vllm, "__version__", "unknown"))
print("Scheduler class:", Scheduler)
print("LLMEngine class:", LLMEngine)
