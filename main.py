import argparse
from langchain_core.messages import HumanMessage
from agents import build_react_agent, build_plan_execute_agent, build_hybrid_agent
from lib.tools_booking import reset_db
from lib.model import get_llm

def main():
    parser = argparse.ArgumentParser(description="Chạy thử nghiệm BTVN#3 Booking Agent")
    parser.add_argument("--agent", choices=["react", "plan", "hybrid"], default="react", help="Chọn mẫu agent")
    parser.add_argument("--role", choices=["guest", "member", "admin"], default="member", help="Phân quyền người dùng")
    parser.add_argument("--query", type=str, default="Đặt cho tôi một vé máy bay từ SGN đến HAN ngày 2026-10-15.", help="Yêu cầu đặt vé")
    args = parser.parse_args()

    reset_db()
    llm = get_llm()
    
    if args.agent == "react":
        print(f"\n Đang chạy [ReAct Agent] với quyền: '{args.role}'...")
        agent = build_react_agent(llm)
    elif args.agent == "plan":
        print(f"\n Đang chạy [Plan-then-Execute Agent] với quyền: '{args.role}'...")
        agent = build_plan_execute_agent(llm)
    else:
        print(f"\n Đang chạy [Hybrid Agent] với quyền: '{args.role}'...")
        agent = build_hybrid_agent(llm)

    initial_state = {
        "messages": [HumanMessage(content=args.query)],
        "user_role": args.role,
        "is_completed": False
    }

    result = agent.invoke(initial_state)

    print("\n" + "=" * 25 + " NHẬT KÝ XỬ LÝ (TRACE) " + "=" * 25)
    for msg in result.get("messages", []):
        sender = getattr(msg, "type", "UNKNOWN").upper()
        if getattr(msg, "tool_calls", None):
            calls_desc = ", ".join([f"{c['name']}({c['args']})" for c in msg.tool_calls])
            print(f"[{sender} -> TOOL_CALL]: Đề xuất gọi {calls_desc}")
        elif msg.content:
            print(f"[{sender}]: {msg.content}")

    print("\n" + "=" * 25 + " KẾT QUẢ NGHIỆP VỤ BẰNG CODE " + "=" * 25)
    print(f"Tiêu chí hoàn thành (is_completed)? -> {result.get('is_completed', False)}")
    
    if result.get("booking_info"):
        print(f" Thông tin vé xác nhận: {result['booking_info']}")
        
    if result.get("handoff_payload"):
        payload = result["handoff_payload"]
        print(f" Gói bàn giao (Handoff):")
        print(f"   - Lý do dừng: {payload.get('stop_reason')}")
        print(f"   - Câu hỏi cho người: {payload.get('cau_hoi_cho_nguoi')}")
    print("=" * 73 + "\n")

if __name__ == "__main__":
    main()