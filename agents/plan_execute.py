from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from lib.state import AgentState, TicketBooking
from lib.tools_booking import search_flights, book_ticket
from lib.guards import LoopDetector, check_permission, verify_completion_code
from lib.handoff import create_handoff, handoff_message

class PlannedStep(BaseModel):
    action: Literal["search_flights", "book_ticket"] = Field(description="Tên công cụ cần thực thi")
    origin: str = Field(description="Mã sân bay đi, ví dụ: SGN, HAN")
    destination: str = Field(description="Mã sân bay đến, ví dụ: HAN, SGN, DAD")
    date: str = Field(description="Ngày bay định dạng YYYY-MM-DD")
    customer_id: Optional[str] = Field(default="KH_01", description="Mã khách hàng")

class PlanSchema(BaseModel):
    steps: List[PlannedStep] = Field(description="Danh sách các bước tuần tự để hoàn thành yêu cầu")

def build_plan_execute_agent(llm):
    loop_detector = LoopDetector(window=4, repeat_k=2)

    def planner_node(state: AgentState):
        user_req = state["messages"][-1].content
        sys_prompt = (
            "Bạn là chuyên gia lập kế hoạch đặt vé. Hãy phân tích yêu cầu của khách hàng "
            "và trích xuất danh sách các bước (action, origin, destination, date, customer_id)."
        )
        try:
            # Ưu tiên structured output nếu LLM hỗ trợ
            structured_llm = llm.with_structured_output(PlanSchema)
            plan_obj = structured_llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_req)])
            steps_dict = [step.model_dump() for step in plan_obj.steps]
        except Exception:
            # Fallback trích xuất nếu dùng MockLLM
            origin = "SGN" if "SGN" in user_req.upper() else "HAN"
            dest = (
                "DAD" if "DAD" in user_req.upper()
                else "HPH" if "HPH" in user_req.upper()
                else "HAN"
            )
            import re
            date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", user_req)
            date = date_match.group(1) if date_match else "2026-10-15"
            steps_dict = [
                {"action": "search_flights", "origin": origin, "destination": dest, "date": date, "customer_id": "KH_01"},
                {"action": "book_ticket", "origin": origin, "destination": dest, "date": date, "customer_id": "KH_01"}
            ]
        return {"plan": steps_dict, "current_step": 0}

    def executor_node(state: AgentState):
        idx = state.get("current_step", 0)
        plan = state.get("plan", [])
        if idx >= len(plan):
            return {}

        step = plan[idx]
        action = step["action"]
        args = {k: v for k, v in step.items() if k != "action"}

        if loop_detector.check(action, args):
            return {"is_completed": False, "current_step": 998}

        if not check_permission(state, action):
            return {"is_completed": False, "current_step": 999}

        new_booking = state.get("booking_info")
        if action == "search_flights":
            res = search_flights.invoke(args)
            msg = AIMessage(content=f"Thực thi {action}: {res}")
        elif action == "book_ticket":
            res = book_ticket.invoke(args)
            msg = AIMessage(content=f"Thực thi {action}: {res}")
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
        else:
            msg = AIMessage(content=f"Hành động không xác định: {action}")

        return {
            "messages": [msg],
            "current_step": idx + 1,
            "booking_info": new_booking
        }

    def execution_router(state: AgentState):
        step = state.get("current_step", 0)
        if step == 999:
            return "auth_handoff"
        if step == 998:
            return "loop_handoff"
        if step >= len(state.get("plan", [])):
            return "verify_node"
        return "executor_node"

    def verify_node(state: AgentState):
        return {"is_completed": verify_completion_code(state)}

    def auth_handoff_node(state: AgentState):
        payload = create_handoff(
            reason="Người dùng chưa có quyền đặt vé (Yêu cầu tài khoản Member).",
            attempts=state.get("plan", []),
            state_snapshot={"user_role": state.get("user_role")},
            question_for_human="Chuyển cho bộ phận CSKH để nâng cấp tài khoản."
        )
        return {"messages": [handoff_message(payload)], "handoff_payload": payload, "is_completed": False}

    def loop_handoff_node(state: AgentState):
        payload = create_handoff(
            reason="Phát hiện vòng lặp khi thực thi kế hoạch.",
            attempts=state.get("plan", []),
            state_snapshot={"current_step": state.get("current_step")},
            question_for_human="Cần người rà soát lại chuỗi hành động lặp."
        )
        return {"messages": [handoff_message(payload)], "handoff_payload": payload, "is_completed": False}

    graph = StateGraph(AgentState)
    graph.add_node("planner_node", planner_node)
    graph.add_node("executor_node", executor_node)
    graph.add_node("verify_node", verify_node)
    graph.add_node("auth_handoff", auth_handoff_node)
    graph.add_node("loop_handoff", loop_handoff_node)

    graph.add_edge(START, "planner_node")
    graph.add_edge("planner_node", "executor_node")
    graph.add_conditional_edges(
        "executor_node",
        execution_router,
        {
            "executor_node": "executor_node",
            "verify_node": "verify_node",
            "auth_handoff": "auth_handoff",
            "loop_handoff": "loop_handoff"
        }
    )
    graph.add_edge("verify_node", END)
    graph.add_edge("auth_handoff", END)
    graph.add_edge("loop_handoff", END)

    return graph.compile()