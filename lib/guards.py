import json
from lib.state import AgentState

class LoopDetector:
    """Bắt lặp tool + tham số kế thừa từ demo2."""
    def __init__(self, window: int = 6, repeat_k: int = 2):
        self.history = []
        self.window = window
        self.repeat_k = repeat_k

    def check(self, tool_name: str, args: dict) -> bool:
        fingerprint = (tool_name, json.dumps(args, sort_keys=True))
        recent = self.history[-self.window:]
        if recent.count(fingerprint) >= self.repeat_k:
            return True
        self.history.append(fingerprint)
        return False

def _get(state, key, default=None):
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)

def check_permission(state, tool_name: str) -> bool:
    """Kiểm quyền: 'book_ticket' yêu cầu tài khoản member/admin."""
    user_role = _get(state, "user_role", "guest")
    if tool_name == "book_ticket" and user_role == "guest":
        return False
    return True

def verify_completion_code(state) -> bool:
    """Tiêu chí hoàn thành kiểm bằng code, không tin text của LLM."""
    booking_info = _get(state, "booking_info")
    if not booking_info:
        return False
    
    b_id = booking_info.get("booking_id") if isinstance(booking_info, dict) else getattr(booking_info, "booking_id", None)
    status = booking_info.get("status") if isinstance(booking_info, dict) else getattr(booking_info, "status", None)
    
    return bool(b_id and status == "confirmed")