from dataclasses import dataclass
from typing import Callable, Dict
from datasets import load_dataset
import random
import time


@dataclass
class SummarizationEvent:
    time: float
    idx: int
    prompt: str


class SummarizationClientSimulator:
    def __init__(
        self,
        dataset_name: str = "cnn_dailymail",
        dataset_config: str = "3.0.0",
        split: str = "train",
        max_docs: int = 200,
        rate_lambda: float = 2.0,
    ) -> None:
        self.ds = load_dataset(dataset_name, dataset_config, split=split)
        self.max_docs = max_docs
        self.rate_lambda = rate_lambda

    def run(self, send_fn: Callable[[SummarizationEvent], None]) -> None:
        t0 = time.monotonic()
        for i, row in enumerate(self.ds):
            if i >= self.max_docs:
                break

            article = row["article"]
            prompt = f"Summarize the following article:\n\n{article}"
            now = time.monotonic() - t0
            ev = SummarizationEvent(time=now, idx=i, prompt=prompt)
            send_fn(ev)

            # if you want actual timing:
            # gap = random.expovariate(self.rate_lambda)
            # time.sleep(gap)
