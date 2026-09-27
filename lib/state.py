from typing import Annotated, List, Optional, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages

class TicketBooking(BaseModel):
    booking_id: Optional[str] = None
    customer_id: str
    origin: str
    destination: str
    date: str
    seat_class: Literal["economy", "business"] = "economy"
    price: float
    status: Literal["pending", "confirmed", "failed"] = "pending"

class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    user_role: str
    booking_info: Optional[TicketBooking]
    plan: List[str]
    current_step: int
    is_completed: bool
    handoff_payload: Optional[dict]