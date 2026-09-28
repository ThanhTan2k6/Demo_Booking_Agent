import time
from langchain_core.messages import HumanMessage
from agents import build_react_agent, build_plan_execute_agent, build_hybrid_agent
from lib.tools_booking import reset_db
from lib.model import get_llm

def count_action_calls(messages):
    """Count tool calls for ReAct and direct action executions for other agents."""
    total = 0
    for message in messages:
        total += len(getattr(message, "tool_calls", []) or [])
        content = str(getattr(message, "content", ""))
        total += content.count("Thực thi search_flights")
        total += content.count("Thực thi book_ticket")
        total += content.count("Substep [search_flights]")
        total += content.count("Substep [book_ticket]")
    return total

def classify_result(result):
    if result.get("is_completed", False):
        return "COMPLETED"
    payload = result.get("handoff_payload") or {}
    reason = payload.get("stop_reason", "")
    if "quyền" in reason.lower():
        return "AUTH_HANDOFF"
    if "lặp" in reason.lower() or "bế tắc" in reason.lower():
        return "LOOP_HANDOFF"
    if "dữ liệu" in reason.lower():
        return "DATA_HANDOFF"
    if "vé" in reason.lower() or "chuyến" in reason.lower():
        return "FLIGHT_HANDOFF"
    return "HANDOFF"

def run_benchmark():
    llm = get_llm()

    agent_builders = {
        "ReAct": build_react_agent,
        "Plan-Execute": build_plan_execute_agent,
        "Hybrid": build_hybrid_agent
    }

    test_cases = [
        {"id": "TC1", "name": "Đặt vé thành công", "role": "member", "query": "Đặt cho tôi một vé máy bay từ SGN đến HAN ngày 2026-10-15."},
        {"id": "TC2", "name": "Chặn quyền Guest", "role": "guest", "query": "Đặt cho tôi một vé máy bay từ SGN đến HAN ngày 2026-10-15."},
        {"id": "TC3", "name": "Chuyến bay hết vé", "role": "member", "query": "Đặt vé máy bay từ SGN đến DAD ngày 2026-10-15."},
        {"id": "TC4", "name": "Không tìm thấy chuyến", "role": "member", "query": "Đặt vé máy bay từ SGN đến HPH ngày 2026-10-15."},
        {"id": "TC5", "name": "Dữ liệu sai ngày", "role": "member", "query": "Đặt vé từ SGN đến HAN ngày 2024-01-01."},
        {"id": "TC6", "name": "Yêu cầu vòng lặp", "role": "member", "query": "Tìm vé SGN đến HAN ngày 2026-10-15 liên tục."},
        {"id": "TC7", "name": "Admin đặt vé thành công", "role": "admin", "query": "Đặt vé máy bay từ SGN đến HAN ngày 2026-10-15."},
        {"id": "TC8", "name": "Guest gặp chuyến hết vé", "role": "guest", "query": "Đặt vé máy bay từ SGN đến DAD ngày 2026-10-15."}
    ]

    print(f"\n{'Agent':<15} | {'Mã':<4} | {'Kịch bản':<22} | {'Thời gian':<10} | {'Steps':<6} | {'Actions':<10} | {'Kết quả':<15}")
    print("=" * 105)

    stats = {
        name: {
            "total": 0,
            "success": 0,
            "handoff": 0,
            "total_time": 0.0,
            "total_steps": 0,
            "total_tool_calls": 0,
            "results": [],
        }
        for name in agent_builders
    }

    for a_name in agent_builders:
        for case in test_cases:
            reset_db()
            agent = agent_builders[a_name](llm)
            start = time.time()
            try:
                res = agent.invoke({
                    "messages": [HumanMessage(content=case["query"])],
                    "user_role": case["role"],
                    "is_completed": False
                })
                lat = round(time.time() - start, 3)
                is_comp = res.get("is_completed", False)
                has_handoff = bool(res.get("handoff_payload"))
                steps = res.get("step_count", 0)
                action_calls = count_action_calls(res.get("messages", []))
                result_type = classify_result(res)
            except Exception as e:
                lat = round(time.time() - start, 3)
                is_comp = False
                has_handoff = True
                steps = 0
                action_calls = 0
                result_type = "ERROR"
                print(f"  ERROR {a_name}/{case['id']}: {type(e).__name__}: {e}")

            stats[a_name]["total"] += 1
            if is_comp:
                stats[a_name]["success"] += 1
            if has_handoff:
                stats[a_name]["handoff"] += 1
            stats[a_name]["total_time"] += lat
            stats[a_name]["total_steps"] += steps
            stats[a_name]["total_tool_calls"] += action_calls
            stats[a_name]["results"].append(result_type)

            print(f"{a_name:<15} | {case['id']:<4} | {case['name']:<22} | {lat:<10}s | {steps:<6} | {action_calls:<10} | {result_type:<15}")
        print("-" * 105)

    print("\n" + "=" * 30 + " BẢNG SO SÁNH HÀNH VI " + "=" * 30)
    print(f"{'Mẫu thiết kế':<18} | {'Success':<9} | {'Handoff':<9} | {'Steps TB':<10} | {'Actions TB':<15} | {'Thời gian TB'}")
    print("-" * 100)
    for a_name, s in stats.items():
        succ_rate = f"{(s['success'] / s['total']) * 100:.1f}%"
        handoff_rate = f"{(s['handoff'] / s['total']) * 100:.1f}%"
        avg_steps = f"{s['total_steps'] / s['total']:.2f}"
        avg_tool_calls = f"{s['total_tool_calls'] / s['total']:.2f}"
        avg_time = f"{s['total_time'] / s['total']:.3f}s"
        print(f"{a_name:<18} | {succ_rate:<9} | {handoff_rate:<9} | {avg_steps:<10} | {avg_tool_calls:<15} | {avg_time}")

    print("\n" + "=" * 30 + " HỒ SƠ KẾT QUẢ THEO KỊCH BẢN " + "=" * 30)
    print(f"{'Kịch bản':<24} | " + " | ".join(f"{name:<15}" for name in agent_builders))
    print("-" * 85)
    for index, case in enumerate(test_cases):
        outcomes = " | ".join(f"{stats[name]['results'][index]:<15}" for name in agent_builders)
        print(f"{case['id'] + ' ' + case['name']:<24} | {outcomes}")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    run_benchmark()
