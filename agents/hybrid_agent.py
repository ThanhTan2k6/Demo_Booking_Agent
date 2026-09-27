import re
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage
from lib.state import AgentState, TicketBooking
from lib.tools_booking import search_flights, book_ticket
from lib.guards import LoopDetector, check_permission, verify_completion_code
from lib.handoff import create_handoff, handoff_message

def parse_user_query(query: str):
    """Trích xuất linh hoạt origin, destination, date từ văn bản người dùng."""
    origin_match = re.search(r"\btừ\s+([A-Z]{3})\b", query, re.IGNORECASE)
    dest_match = re.search(r"\bđến\s+([A-Z]{3})\b", query, re.IGNORECASE)
    date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", query)

    origin = origin_match.group(1).upper() if origin_match else "SGN"
    dest = dest_match.group(1).upper() if dest_match else "HAN"
    date = date_match.group(1) if date_match else "2026-10-15"
    return origin, dest, date

def build_hybrid_agent(llm):
    loop_det = LoopDetector(window=3, repeat_k=2)

    def planner_node(state: AgentState):
        query = state["messages"][-1].content
        origin, dest, date = parse_user_query(query)
        
        plan = [
            {"action": "search_flights", "origin": origin, "destination": dest, "date": date},
            {"action": "book_ticket", "origin": origin, "destination": dest, "date": date, "customer_id": "KH_01"}
        ]
        return {"plan": plan, "current_step": 0}

    def react_substep_node(state: AgentState):
        step_idx = state.get("current_step", 0)
        plan = state.get("plan", [])
        step = plan[step_idx]
        action = step["action"]
        args = {k: v for k, v in step.items() if k != "action"}

        # Loop detection
        if loop_det.check(action, args):
            return {"is_completed": False, "current_step": 998}

        new_booking = state.get("booking_info")

        if action == "search_flights":
            res = search_flights.invoke(args)
            obs_text = f"Kết quả tra cứu: {res}"
            if res.get("seats", 0) <= 0:
                obs_text += " -> Chuyến bay đã hết chỗ, dừng thực hiện đặt vé."
                return {
                    "messages": [AIMessage(content=obs_text)],
                    "current_step": 997, # Báo hết vé
                    "booking_info": None
                }
        else:
            if not check_permission(state, action):
                return {"is_completed": False, "current_step": 999}

            res = book_ticket.invoke(args)
            obs_text = f"Kết quả đặt vé: {res}"
            if res.get("status") == "confirmed":
                new_booking = TicketBooking(
                    booking_id=res["booking_id"],
                    customer_id=args.get("customer_id", "KH_01"),
                    origin=args["origin"],
                    destination=args["destination"],
                    date=args["date"],
                    price=1500000,
                    status="confirmed"
                )

        return {
            "messages": [AIMessage(content=f"Substep [{action}]: {obs_text}")],
            "current_step": step_idx + 1,
            "booking_info": new_booking
        }

    def hybrid_router(state: AgentState):
        step = state.get("current_step")
        if step == 999:
            return "auth_handoff"
        if step == 998:
            return "loop_handoff"
        if step == 997:
            return "sold_out_handoff"
        if step >= len(state.get("plan", [])):
            return "verify_node"
        return "react_substep_node"

    def verify_node(state: AgentState):
        return {"is_completed": verify_completion_code(state)}

    def auth_handoff_node(state: AgentState):
        payload = create_handoff(
            reason="Người dùng chưa có quyền đặt vé (Yêu cầu tài khoản Member).",
            attempts=[m.content for m in state.get("messages", [])],
            state_snapshot={"current_step": state.get("current_step")},
            question_for_human="Hỗ trợ cấp quyền hoặc chuyển sang thanh toán thủ công."
        )
        return {"messages": [handoff_message(payload)], "handoff_payload": payload, "is_completed": False}

    def loop_handoff_node(state: AgentState):
        payload = create_handoff(
            reason="Phát hiện bế tắc lặp bước trong quá trình thực thi kế hoạch lai.",
            attempts=[m.content for m in state.get("messages", [])],
            state_snapshot={"current_step": state.get("current_step")},
            question_for_human="Có muốn đổi tuyến bay khác không?"
        )
        return {"messages": [handoff_message(payload)], "handoff_payload": payload, "is_completed": False}

    def sold_out_handoff_node(state: AgentState):
        payload = create_handoff(
            reason="Hết vé trên chuyến bay đã chọn.",
            attempts=[m.content for m in state.get("messages", [])],
            state_snapshot={"current_step": state.get("current_step")},
            question_for_human="Khách có muốn đổi sang ngày khác hoặc bay chặng thay thế không?"
        )
        return {"messages": [handoff_message(payload)], "handoff_payload": payload, "is_completed": False}

    graph = StateGraph(AgentState)
    graph.add_node("planner_node", planner_node)
    graph.add_node("react_substep_node", react_substep_node)
    graph.add_node("verify_node", verify_node)
    graph.add_node("auth_handoff", auth_handoff_node)
    graph.add_node("loop_handoff", loop_handoff_node)
    graph.add_node("sold_out_handoff", sold_out_handoff_node)

    graph.add_edge(START, "planner_node")
    graph.add_edge("planner_node", "react_substep_node")
    graph.add_conditional_edges(
        "react_substep_node",
        hybrid_router,
        {
            "react_substep_node": "react_substep_node",
            "verify_node": "verify_node",
            "auth_handoff": "auth_handoff",
            "loop_handoff": "loop_handoff",
            "sold_out_handoff": "sold_out_handoff"
        }
    )
    graph.add_edge("verify_node", END)
    graph.add_edge("auth_handoff", END)
    graph.add_edge("loop_handoff", END)
    graph.add_edge("sold_out_handoff", END)

    return graph.compile()
