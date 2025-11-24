# vllm/simulator.py

from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class TraceRequest:
    request_id: str
    prompt_token_ids: List[int]
    pos: int = 0   # how many output tokens we've already "returned"

class Simulator:
    """
    Milestone 1: very simple simulator.

    For now:
    - It just echoes back the prompt tokens one by one as "generated" tokens.
    - Later you can replace this with reading from your trace / prefix logs.
    """

    def __init__(self) -> None:
        # request_id -> TraceRequest
        self._requests: Dict[str, TraceRequest] = {}

    def start_request(self, request_id: str, prompt_token_ids: List[int]) -> None:
        """Register a new request with its prompt tokens."""
        self._requests[request_id] = TraceRequest(
            request_id=request_id,
            prompt_token_ids=list(prompt_token_ids),
            pos=0,
        )

    def next_token(self, request_id: str) -> Optional[int]:
        """
        Return the next token for this request, or None when finished.

        Milestone 1: just walk through the prompt tokens, then stop.
        Later: you'll instead walk through simulated *output* tokens loaded
        from a trace.
        """
        req = self._requests.get(request_id)
        if req is None:
            return None

        if req.pos >= len(req.prompt_token_ids):
            # No more tokens to "generate"
            return None

        token = req.prompt_token_ids[req.pos]
        req.pos += 1
        return token
