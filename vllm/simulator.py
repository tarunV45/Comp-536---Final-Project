from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class TraceRequest:
    request_id: str
    prompt_tokens: List[int]
    response_tokens: List[int]
    position: int = 0  # how many response tokens have been "generated" so far


class FakeTraceDB:
    """
    Temporary placeholder for pre-collected traces.
    Later you will replace this with real traces (ShareGPT, etc.).
    """

    def lookup(self, prompt_tokens: List[int]) -> List[int]:
        # For now, just echo the prompt back as the "response".
        return prompt_tokens[:]


class Simulator:
    """
    Simulator that mimics GPU inference using pre-collected traces.
    """

    def __init__(self, trace_db: Optional[FakeTraceDB] = None):
        self.trace_db = trace_db or FakeTraceDB()
        self._requests: Dict[str, TraceRequest] = {}

    def start_request(self, request_id: str, prompt_tokens: List[int]) -> None:
        response_tokens = self.trace_db.lookup(prompt_tokens)
        self._requests[request_id] = TraceRequest(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            position=0,
        )

    def next_token(self, request_id: str) -> Optional[int]:
        """Return the next response token, or None if finished."""
        req = self._requests[request_id]
        if req.position >= len(req.response_tokens):
            return None
        token = req.response_tokens[req.position]
        req.position += 1
        return token

    def is_finished(self, request_id: str) -> bool:
        req = self._requests[request_id]
        return req.position >= len(req.response_tokens)
