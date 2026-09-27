from langchain_core.messages import AIMessage

def create_handoff(reason: str, attempts: list, state_snapshot: dict, question_for_human: str) -> dict:
    return {
        "stop_reason": reason,
        "da_thu": attempts,
        "trang_thai": state_snapshot,
        "cau_hoi_cho_nguoi": question_for_human
    }

def handoff_message(payload: dict) -> AIMessage:
    text = (
        f"🚨 DỪNG VÀ BÀN GIAO CHO CON NGƯỜI:\n"
        f"- Lý do dừng: {payload['stop_reason']}\n"
        f"- Các bước đã thử: {payload['da_thu']}\n"
        f"- Trạng thái: {payload['trang_thai']}\n"
        f"- Câu hỏi cần người xử lý: {payload['cau_hoi_cho_nguoi']}"
    )
    return AIMessage(content=text)