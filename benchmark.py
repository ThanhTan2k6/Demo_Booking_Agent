import time
from langchain_core.messages import HumanMessage
from agents import build_react_agent, build_plan_execute_agent, build_hybrid_agent
from lib.tools_booking import reset_db
from lib.model import get_llm

def run_benchmark():
    llm = get_llm()

    agents = {
        "ReAct": build_react_agent(llm),
        "Plan-Execute": build_plan_execute_agent(llm),
        "Hybrid": build_hybrid_agent(llm)
    }

    test_cases = [
        {"id": "TC1", "name": "Đặt vé thành công", "role": "member", "query": "Đặt cho tôi một vé máy bay từ SGN đến HAN ngày 2026-10-15."},
        {"id": "TC2", "name": "Chặn quyền Guest", "role": "guest", "query": "Đặt cho tôi một vé máy bay từ SGN đến HAN ngày 2026-10-15."},
        {"id": "TC3", "name": "Chuyến bay hết vé", "role": "member", "query": "Đặt vé máy bay từ SGN đến DAD ngày 2026-10-15."},
        {"id": "TC4", "name": "Không tìm thấy chuyến", "role": "member", "query": "Đặt vé máy bay từ SGN đến HPH ngày 2026-10-15."},
        {"id": "TC5", "name": "Dữ liệu sai ngày", "role": "member", "query": "Đặt vé từ SGN đến HAN ngày 2024-01-01."},
        {"id": "TC6", "name": "Yêu cầu vòng lặp", "role": "member", "query": "Tìm vé SGN đến HAN ngày 2026-10-15 liên tục."}
    ]

    print(f"\n{'Agent':<15} | {'Mã':<4} | {'Kịch bản':<22} | {'Thời gian':<10} | {'Steps':<6} | {'Latency/step':<24} | {'Hoàn thành':<12} | {'Handoff':<8}")
    print("=" * 125)

    stats = {name: {"total": 0, "success": 0, "handoff": 0, "total_time": 0.0} for name in agents}

    for a_name, agent in agents.items():
        for case in test_cases:
            reset_db()
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
                step_latencies = res.get("step_latencies", [])
            except Exception as e:
                lat = round(time.time() - start, 3)
                is_comp = False
                has_handoff = True
                steps = 0
                step_latencies = []
                print(f"  ERROR {a_name}/{case['id']}: {type(e).__name__}: {e}")

            stats[a_name]["total"] += 1
            if is_comp:
                stats[a_name]["success"] += 1
            if has_handoff:
                stats[a_name]["handoff"] += 1
            stats[a_name]["total_time"] += lat

            print(f"{a_name:<15} | {case['id']:<4} | {case['name']:<22} | {lat:<10}s | {steps:<6} | {str(step_latencies):<24} | {str(is_comp):<12} | {str(has_handoff):<8}")
        print("-" * 125)

    print("\n" + "=" * 30 + " BẢNG TỔNG HỢP HIỆU QUẢ " + "=" * 30)
    print(f"{'Mẫu thiết kế':<18} | {'Tỷ lệ thành công':<20} | {'Tỷ lệ bàn giao':<18} | {'Thời gian TB'}")
    print("-" * 75)
    for a_name, s in stats.items():
        succ_rate = f"{(s['success'] / s['total']) * 100:.1f}%"
        handoff_rate = f"{(s['handoff'] / s['total']) * 100:.1f}%"
        avg_time = f"{s['total_time'] / s['total']:.3f}s"
        print(f"{a_name:<18} | {succ_rate:<20} | {handoff_rate:<18} | {avg_time}")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    run_benchmark()
