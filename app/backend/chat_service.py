from langchain_core.messages import HumanMessage
from typing import Optional, List
from app.backend.db import save_message
from app.backend.langgraph_flow import get_graph
from app.backend.gaurdrail_flow import get_validator_graph
from langgraph.types import Command


# def build_prompt(user_input: str, images_list: Optional[List[str]] = None, thread_id: str = None) -> str:
#     """
#     Build a structured prompt so downstream tools can reliably
#     extract text + image context.
#     """
#     prompt_parts = []

#     # 1. User instruction
#     prompt_parts.append(
#         f"""
#         USER_QUERY:
#         {user_input}
#         """.strip()
#     )

#     # 2. Optional image context
#     if images_list:
#         image_block = "\n".join(
#             [f"- {img}" for img in images_list]
#         )
#         prompt_parts.append(
#             f"""
#             IMAGE_INPUTS:
#             The user has provided the following image file paths.
#             Use them when calling vision/image tools.

#             {image_block}
#             """.strip()
#         )

#         prompt_parts.append(
#             f"""
#             TOOLS_INPUTS:
#             For Any Tool which requires request_id you can use the following id

#             id - {thread_id}
#             """.strip()
#         )

#     # 3. Tool instruction hint (optional but recommended)
#     prompt_parts.append(
#         """
#         INSTRUCTIONS:
#         - If image understanding is required, use the IMAGE_INPUTS.
#         - If no images are relevant, answer using text only.
#         - Do not assume image content unless explicitly provided.
#         """.strip()
#     )
#     print(prompt_parts)
#     return "\n\n".join(prompt_parts)

# def run_chat_stream(
#     thread_id: str,
#     user_input: str | None = None,
#     images_list: Optional[List[str]] = None,
#     command: Command | None = None,
# ):
#     """
#     Streams LangGraph messages.
#     Streamlit is responsible for rendering tokens + tools.
#     """
#     graph = get_graph()
#     if user_input is not None:
#         save_message(thread_id, "user", user_input)

#         prompt = build_prompt(
#             user_input=user_input,
#             images_list=images_list
#         )
#     else:
#         user_input = command.update["messages"][0].content
#         save_message(thread_id, "user", user_input)

#     if command:
#         input_payload = command
#     else:
#         input_payload = {
#             "messages": [
#                 HumanMessage(
#                     content=prompt,
#                     additional_kwargs={"images": images_list or []}
#                 )
#             ],
#             "thread_id": thread_id,
#         }

#     config = {
#         "configurable": {"thread_id": thread_id},
#         "metadata": {"thread_id": thread_id},
#         "run_name": "chat_turn",
#     }

#     for message_chunk in graph.stream(
#         input_payload,
#         config=config,
#         stream_mode="values",
#     ):
#         yield message_chunk



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
