from dataclasses import dataclass
from typing import Callable
from datasets import load_dataset
import time


@dataclass
class QAEvent:
    time: float
    idx: int
    prompt: str


class QAClientSimulator:
    def __init__(
        self,
        dataset_name: str = "squad_v2",
        split: str = "train",
        max_examples: int = 200,
    ) -> None:
        self.ds = load_dataset(dataset_name, split=split)
        self.max_examples = max_examples

    def run(self, send_fn: Callable[[QAEvent], None]) -> None:
        t0 = time.monotonic()
        for i, row in enumerate(self.ds):
            if i >= self.max_examples:
                break
            ctx = row["context"]
            q = row["question"]
            prompt = f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:"
            now = time.monotonic() - t0
            ev = QAEvent(time=now, idx=i, prompt=prompt)
            send_fn(ev)
