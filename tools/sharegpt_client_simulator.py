# tools/sharegpt_client_simulator.py

import math
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer


@dataclass
class RequestEvent:
    """One synthetic request to send to vLLM."""
    timestamp: float        # simulated "send" time (seconds since t0)
    conv_id: str            # conversation ID from ShareGPT
    turn_index: int         # which turn within the conversation
    prompt_text: str        # fully formatted prompt text (chat template applied)
    meta: Dict              # any extra metadata (roles, raw messages, etc.)


class ShareGPTClientSimulator:
    """
    Client simulator for the ShareGPT_Vicuna_unfiltered dataset.

    It:
    - Loads ShareGPT.
    - Converts each conversation into HF 'messages' ({role, content}).
    - Uses the model's chat_template to produce final prompt strings.
    - Assigns timestamps using a Poisson process if no timestamps exist.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        sharegpt_name: str = "anon8231489123/ShareGPT_Vicuna_unfiltered",
        split: str = "train",
        rate_lambda: float = 5.0,  # avg requests per second for Poisson
        max_convs: Optional[int] = 100,  # limit for debugging
        seed: int = 42,
    ) -> None:
        self.model_name = model_name
        self.ds_name = sharegpt_name
        self.split = split
        self.rate_lambda = rate_lambda
        self.max_convs = max_convs
        self.rng = random.Random(seed)

        print(f"[ShareGPTClientSimulator] Loading dataset {sharegpt_name}:{split}...")

        # This dataset doesn't define standard splits; we must point to the JSON file.
        # We'll use the "no_imsorry" version by default.
        self.ds = load_dataset(
        "json",
        data_files=(
            "https://huggingface.co/datasets/"
            "anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/"
            "ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"
        ),
        split="train",
        )

        print(f"[ShareGPTClientSimulator] Raw examples: {len(self.ds)}")

        if max_convs is not None:
            self.ds = self.ds.select(range(min(max_convs, len(self.ds))))

        print(f"[ShareGPTClientSimulator] Using {len(self.ds)} conversations after truncation.")


        if max_convs is not None:
            self.ds = self.ds.select(range(min(max_convs, len(self.ds))))

        print(f"[ShareGPTClientSimulator] Loaded {len(self.ds)} conversations.")

        print(f"[ShareGPTClientSimulator] Loading tokenizer {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        )

        # For many base models (like gpt2), there is no chat_template.
        # That's fine: we'll fall back to a simple manual chat format.
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            print("[ShareGPTClientSimulator] Using tokenizer chat_template.")
        else:
            print("[ShareGPTClientSimulator] No chat_template found; using manual prompt formatting.")


    # ---------- timing ----------

    def _poisson_interarrival(self) -> float:
        """Draw an inter-arrival time from exponential(rate_lambda)."""
        u = self.rng.random()
        return -math.log(1.0 - u) / self.rate_lambda

    # ---------- ShareGPT → messages ----------

    def _conversation_to_messages(self, conv: Dict) -> List[Dict[str, str]]:
        """
        Convert one ShareGPT conversation into HF-style messages:
        [{"role": "user" / "assistant", "content": "..."}]
        """
        # Many ShareGPT variants store conversations as a list under "conversations"
        # with "from" (e.g., "human"/"gpt") and "value" (text) fields.
        if "conversations" not in conv:
            raise KeyError("Expected 'conversations' key in ShareGPT record")

        messages = []
        for turn in conv["conversations"]:
            speaker = turn.get("from", "")
            text = turn.get("value", "")

            if not text:
                continue

            if speaker.lower() in ["human", "user"]:
                role = "user"
            elif speaker.lower() in ["gpt", "assistant", "chatgpt"]:
                role = "assistant"
            else:
                # Default to 'user' if unknown; you can refine this later
                role = "user"

            messages.append({"role": role, "content": text})

        return messages

    # ---------- chat template application ----------

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """
    Turn messages into a prompt string.

    If the tokenizer has a chat_template, we use it.
    Otherwise, we build a simple manual format like:

        User: ...
        Assistant: ...
        User: ...
        Assistant:

    and let the model generate the next assistant turn.
    """
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            )

        # Fallback: simple manual format
        lines: List[str] = []
        for m in messages:
            role = m["role"]
            if role == "user":
                prefix = "User:"
            elif role == "assistant":
                prefix = "Assistant:"
            else:
                prefix = f"{role.capitalize()}:"
        lines.append(f"{prefix} {m['content']}")
        # Add a generation cue for the next assistant turn
        lines.append("Assistant:")
        return "\n".join(lines)


    # ---------- request event generation ----------

    def iter_events(
        self,
        mode: str = "single-turn",
        start_time: float = 0.0,
    ) -> Iterable[RequestEvent]:
        """
        Yield RequestEvent objects in simulated time order.

        mode:
          - "single-turn": only the first user+assistant turn per conversation
          - "multi-turn": full conversation, sequential user turns

        This does *not* actually call vLLM; it just yields prompts with timestamps.
        """

        t = start_time

        for idx, conv in enumerate(self.ds):
            conv_id = str(conv.get("id", idx))
            messages = self._conversation_to_messages(conv)

            if not messages:
                continue

            if mode == "single-turn":
                # Take only the first user message as prompt context
                # (you can refine this logic if you want more precise "first turn")
                first_user_idx = next(
                    (i for i, m in enumerate(messages) if m["role"] == "user"),
                    None,
                )
                if first_user_idx is None:
                    continue

                truncated = messages[: first_user_idx + 1]
                prompt_text = self._apply_chat_template(truncated)

                # Poisson arrival
                t += self._poisson_interarrival()
                yield RequestEvent(
                    timestamp=t,
                    conv_id=conv_id,
                    turn_index=0,
                    prompt_text=prompt_text,
                    meta={
                        "mode": "single-turn",
                        "raw_messages": messages,
                    },
                )

            elif mode == "multi-turn":
                turn_idx = 0
                # Every user turn becomes a new request (with prior context included)
                for i, m in enumerate(messages):
                    if m["role"] != "user":
                        continue
                    truncated = messages[: i + 1]
                    prompt_text = self._apply_chat_template(truncated)

                    t += self._poisson_interarrival()
                    yield RequestEvent(
                        timestamp=t,
                        conv_id=conv_id,
                        turn_index=turn_idx,
                        prompt_text=prompt_text,
                        meta={
                            "mode": "multi-turn",
                            "raw_messages": messages,
                        },
                    )
                    turn_idx += 1

            else:
                raise ValueError(f"Unknown mode {mode!r}")

    # ---------- convenience runner ----------

    def run(
        self,
        mode: str,
        send_fn: Callable[[RequestEvent], None],
    ) -> None:
        """
        Drive the simulation: for each RequestEvent, call send_fn(event).

        In your real system, send_fn would:
          - send event.prompt_text to vLLM
          - collect the response
          - log KV/prefix stats
        On your Mac you can just print/log for now.
        """
        t0 = time.time()
        for event in self.iter_events(mode=mode, start_time=0.0):
            # Here we don't actually sleep; we just simulate logical time.
            send_fn(event)
        print(f"[ShareGPTClientSimulator] Completed run in {time.time() - t0:.2f}s.")
