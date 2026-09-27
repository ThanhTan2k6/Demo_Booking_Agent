from langgraph.graph import StateGraph, START, END
from langchain_core.messages import ToolMessage
from lib.state import AgentState, TicketBooking
from lib.tools_booking import tools, search_flights, book_ticket
from lib.guards import LoopDetector, check_permission, record_step, validate_booking_data, verify_completion_code
from lib.handoff import create_handoff, handoff_message

def build_react_agent(llm):
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState):
        import time
        started = time.perf_counter()
        msgs = state.get("messages", [])
        response = llm_with_tools.invoke(msgs)
        return {"messages": [response], **record_step(state, "llm_reasoning", started)}

    def guards_router(state: AgentState):
        messages = state.get("messages", [])
        if not messages:
            return "verify_node"
            
        last_msg = messages[-1]
        tool_calls = getattr(last_msg, "tool_calls", None)
        
        if not tool_calls:
            text = getattr(last_msg, "content", "")
            if "BK_" in text:
                return "verify_node"
            return "verify_node"

        det = LoopDetector(window=4, repeat_k=2)
        for m in messages[:-1]:
            calls = getattr(m, "tool_calls", None)
            if calls:
                for c in calls:
                    det.check(c.get("name", ""), c.get("args", {}))

        for call in tool_calls:
            t_name = call.get("name", "")
            t_args = call.get("args", {})
            valid, _ = validate_booking_data(t_args)
            if not valid:
                return "data_handoff"
            if det.check(t_name, t_args):
                return "loop_handoff"
            if not check_permission(state, t_name):
                return "auth_handoff"
                
        return "tools_node"

    def tools_node(state: AgentState):
        import time
        started = time.perf_counter()
        last_msg = state["messages"][-1]
        results = []
        new_booking_info = state.get("booking_info")
        
        tool_map = {t.name: t for t in tools}
        for call in last_msg.tool_calls:
            t_name = call.get("name")
            t_args = call.get("args", {})
            call_id = call.get("id", "call_id")
            
            tool_func = tool_map.get(t_name)
            if not tool_func:
                continue
                
            obs = tool_func.invoke(t_args)
            results.append(ToolMessage(tool_call_id=call_id, content=str(obs)))
            
            if t_name == "book_ticket" and obs.get("status") == "confirmed":
                new_booking_info = TicketBooking(
                    booking_id=obs["booking_id"],
                    customer_id=t_args.get("customer_id", "KH_01"),
                    origin=t_args.get("origin", "SGN"),
                    destination=t_args.get("destination", "HAN"),
                    date=t_args.get("date", "2026-10-15"),
                    price=1500000,
                    status="confirmed"
                )
        return {
            "messages": results,
            "booking_info": new_booking_info,
            **record_step(state, "tool_execution", started)
        }

    def verify_node(state: AgentState):
        return {"is_completed": verify_completion_code(state)}

    def handoff_node(reason: str):
        def _node(state: AgentState):
            payload = create_handoff(
                reason=reason,
                attempts=[m.content for m in state.get("messages", []) if getattr(m, "type", "") == "tool"],
                state_snapshot={"user_role": state.get("user_role"), "is_completed": False},
                question_for_human="Cần can thiệp do vi phạm quyền hoặc lỗi vòng lặp."
            )
            return {"messages": [handoff_message(payload)], "handoff_payload": payload, "is_completed": False}
        return _node

    def data_handoff_node(state: AgentState):
        payload = create_handoff(
            reason="Dữ liệu đặt vé không hợp lệ theo ràng buộc của Harness.",
            attempts=[m.content for m in state.get("messages", [])],
            state_snapshot={"step_count": state.get("step_count", 0)},
            question_for_human="Vui lòng cung cấp mã sân bay và ngày bay hợp lệ."
        )
        return {"messages": [handoff_message(payload)], "handoff_payload": payload, "is_completed": False}

    graph = StateGraph(AgentState)
    graph.add_node("agent_node", agent_node)
    graph.add_node("tools_node", tools_node)
    graph.add_node("verify_node", verify_node)
    graph.add_node("loop_handoff", handoff_node("Phát hiện vòng lặp vô ích lặp lại cùng tham số."))
    graph.add_node("auth_handoff", handoff_node("Người dùng chưa đủ quyền hạn để thực thi hành động này."))
    graph.add_node("data_handoff", data_handoff_node)

    graph.add_edge(START, "agent_node")
    graph.add_conditional_edges(
        "agent_node",
        guards_router,
        {
            "tools_node": "tools_node",
            "verify_node": "verify_node",
            "loop_handoff": "loop_handoff",
            "auth_handoff": "auth_handoff",
            "data_handoff": "data_handoff"
        }
    )
    graph.add_edge("tools_node", "agent_node")
    graph.add_edge("verify_node", END)
    graph.add_edge("loop_handoff", END)
    graph.add_edge("auth_handoff", END)
    graph.add_edge("data_handoff", END)

    return graph.compile()