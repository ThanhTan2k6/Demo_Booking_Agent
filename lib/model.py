import os
from langchain_core.messages import AIMessage
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.genai")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_google_genai")

load_dotenv()

class MockLLM:
    """Giả lập LLM offline khi không có key hoặc key lỗi."""
    def __init__(self, mode="normal"):
        self.mode = mode

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages, *args, **kwargs):
        last_msg = messages[-1]
        content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        conversation = " ".join(
            str(getattr(message, "content", message)) for message in messages
        )

        if "liên tục" in conversation.lower() or "lặp" in conversation.lower():
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "search_flights",
                    "args": {"origin": "SGN", "destination": "HAN", "date": "2026-10-15"},
                    "id": "loop_call"
                }]
            )

        has_booked = any("BK_" in getattr(m, "content", "") or "confirmed" in getattr(m, "content", "") for m in messages)
        has_searched = any("seats" in getattr(m, "content", "") or "available" in getattr(m, "content", "") for m in messages)

        origin = "SGN" if "SGN" in conversation.upper() else "HAN"
        dest = "DAD" if "DAD" in conversation.upper() else ("HPH" if "HPH" in conversation.upper() else "HAN")
        date = "2024-01-01" if "2024" in conversation else "2026-10-15"

        if has_booked:
            return AIMessage(content="Đã hoàn tất đặt vé thành công.")
        
        if has_searched:
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "book_ticket",
                    "args": {"origin": origin, "destination": dest, "date": date, "customer_id": "KH_01"},
                    "id": "call_book"
                }]
            )

        return AIMessage(
            content="",
            tool_calls=[{
                "name": "search_flights",
                "args": {"origin": origin, "destination": dest, "date": date},
                "id": "call_search"
            }]
        )
    
def get_llm(mode=None):
    """Create the requested model, with an explicit offline fallback."""
    selected_mode = (mode or os.getenv("BOOKING_LLM_MODE", "offline")).lower()
    if selected_mode not in {"offline", "api"}:
        raise ValueError("BOOKING_LLM_MODE must be 'offline' or 'api'")

    if selected_mode == "offline":
        print("Run mode: [MockLLM] offline")
        return MockLLM()

    env_key = os.getenv("GEMINI_API_KEY")
    if env_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            print("Run mode: [Gemini API] (gemini-2.5-flash)")
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0,
                google_api_key=env_key
            )
        except Exception as e:
            print(f"Gemini initialization failed: {e}")

    raise RuntimeError("API mode requires a valid GEMINI_API_KEY")