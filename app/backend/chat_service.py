from langchain_core.messages import HumanMessage
from typing import Optional, List
from app.backend.db import save_message
from app.backend.langgraph_flow import get_graph
from app.backend.gaurdrail_flow import get_validator_graph
from langgraph.types import Command


def run_chat_stream(
    thread_id: str,
    user_input: str | None = None,
    images_list: Optional[List[str]] = None,
    command: Command | None = None,
):
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_turn",
    }

    # RESUME PATH — BYPASS VALIDATOR COMPLETELY
    if command and command.resume:
        print("[RESUME] Resuming chat graph")
        chat_graph = get_graph()

        for event in chat_graph.stream(
            command,
            config=config,
            stream_mode="values",
        ):
            yield event

        return

    # ▶️ NEW MESSAGE PATH — VALIDATE FIRST
    validator_graph = get_validator_graph()

    input_payload = {
        "user_input": user_input,
        "thread_id": thread_id,
        "images_list": images_list,
        "command": None,  # never pass command here
    }

    for event in validator_graph.stream(
        input_payload,
        config=config,
        stream_mode="values",
    ):
        print("event is ", event)

        if "__delegate__" in event:
            print("[DELEGATE] Delegating to chat graph")
            chat_graph = get_graph()

            for chat_event in chat_graph.stream(
                event["__delegate__"]["input"],
                config=config,
                stream_mode="values",
            ):
                yield chat_event
        else:
            yield event
