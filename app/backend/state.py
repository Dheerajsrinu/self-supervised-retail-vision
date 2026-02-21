from typing import TypedDict, List, Annotated, Optional, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from langgraph.types import Command

class HealthWarningItem(TypedDict):
    category: str
    quantity: int

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    thread_id: str
    tools_done: bool
    health_warning_input: List[HealthWarningItem]
    detected_items: Optional[Dict[str, int]]
    user_role: Optional[str]  # 'customer' or 'store_manager'

class GaurdrailState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    thread_id: str
    validator_status: str
    validator_reason: str
    images_list: List[str]
    command: Optional[Command]
    user_input: Optional[str]

    __delegate__: Dict[str, Any]