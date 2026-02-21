from langgraph.graph import StateGraph, START, END
from app.backend.state import GaurdrailState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from typing import Optional, List

from langchain_openai import ChatOpenAI
from app.config import OPENAI_API_KEY
from pydantic import BaseModel

from app.backend.db import save_message
from app.backend.langgraph_flow import get_graph

from langgraph.types import Command

graph = get_graph()

llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0.2,
    api_key=OPENAI_API_KEY
)

class ValidatorSchema(BaseModel):
    allowed: bool
    reason: str

def build_prompt(user_input: str, images_list: Optional[List[str]] = None, thread_id: str = None) -> str:
    """
    Build a structured prompt so downstream tools can reliably
    extract text + image context.
    """
    prompt_parts = []

    # 1. User instruction
    prompt_parts.append(
        f"""
        USER_QUERY:
        {user_input}
        """.strip()
    )

    # 2. Optional image context
    if images_list:
        image_block = "\n".join(
            [f"- {img}" for img in images_list]
        )
        prompt_parts.append(
            f"""
            IMAGE_INPUTS:
            The user has provided the following image file paths.
            Use them when calling vision/image tools.

            {image_block}
            """.strip()
        )

    # 3. Always include user_id for tools that need it
    prompt_parts.append(
        f"""
        USER_CONTEXT:
        For any tool that requires user_id or request_id, use: {thread_id}
        """.strip()
    )

    # 4. Tool instruction hint
    prompt_parts.append(
        """
        INSTRUCTIONS:
        - If the user wants to place a direct order (e.g., "order 2 candy", "buy 5 chocolates"), use the place_direct_order tool.
        - If image understanding is required, use the IMAGE_INPUTS.
        - If no images are relevant, answer using text only.
        - Do not assume image content unless explicitly provided.
        """.strip()
    )
    print(prompt_parts)
    return "\n\n".join(prompt_parts)

def validate_request(state: GaurdrailState):
    prompt="""
            Role: You are a request validator for a Retail Checkout Assistant chatbot.
            
            Task:
            Determine whether the user's request is allowed based on the approved topics listed below.
            Only evaluate intent and subject matter of the request. Do not answer the request itself.

            Allowed Topics (the request must be primarily about one or more of these):

            1. General Greetings & Conversational Messages
               - Greetings like "hi", "hello", "good morning", "good evening", "hey", "thanks", "thank you", "bye", "goodbye"
               - Polite acknowledgments, small talk, or casual conversation starters
               - Questions about the assistant's capabilities or what it can do
               - Requests for help or assistance

            2. Retail Images Analysis
               - Images of stores, shelves, aisles, refrigerators, displays, or product arrangements
               - Requests to analyze, process, or understand retail/store images

            3. Shelf Detection & Count
               - Detecting shelves in images
               - Counting shelves, racks, rows, or display levels in a retail image

            4. Product Detection & Count
               - Detecting products in retail images
               - Counting how many products, items, units, or packages appear in a retail image

            5. Empty Space Calculation
               - Calculating empty shelf space percentage
               - Analyzing shelf occupancy or availability
               - Finding gaps or empty areas on shelves

            6. Product Recognition & Names
               - Identifying or listing visible product names or brands shown in a retail image
               - Recognizing specific products or SKUs

            7. Product Nutrition Information
               - Nutrition-related information (e.g., calories, ingredients, macros, nutrition labels) for products shown or referenced in a retail image

            8. Order-Related Tasks
               - Creating an order, placing an order
               - Updating an order, modifying cart items
               - Summarizing an order, reviewing cart contents
               - Extracting order details from an image or conversation
               - Confirming orders, checkout assistance
               - Any task related to ordering or purchasing products

            Disallowed Topics (examples):

            - Requests completely unrelated to retail, products, or store images
            - Price prediction, demand forecasting, or sales strategy
            - Customer behavior analysis or market research
            - Personal data extraction or facial recognition
            - Medical, legal, or financial advice
            - Image manipulation, editing, or image generation
            - Coding, programming, or technical development questions
            - Political, religious, or controversial topics
            - Any topic not clearly connected to the allowed list above

            Output Format (STRICT)

            Respond with only one of the following JSON objects:

            If allowed:
            {
            "allowed": true,
            "reason": "The request is about <brief explanation tied to allowed topics>."
            }

            If disallowed:
            {
            "allowed": false,
            "reason": "The request is not related to retail checkout assistance. I can help with: analyzing retail images, detecting shelves/products, calculating empty space, recognizing products, product nutrition info, or creating orders."
            }

            Validation Rules:
            1. General greetings and polite messages should ALWAYS be allowed.
            2. If the request clearly matches at least one allowed topic, mark it as allowed.
            3. If the request is ambiguous but reasonably related to retail, products, images, or orders, mark it as allowed.
            4. If the request does not match any allowed topic, mark it as disallowed.
            5. Do not infer hidden intent beyond what is stated.
            6. When in doubt about retail-related queries, lean towards allowing them.
        """
    validator_model = llm.with_structured_output(ValidatorSchema)
    response = validator_model.invoke([
                    SystemMessage(content=prompt),
                    HumanMessage(content=state["user_input"])
                ])
    print("validator response -> ", response)
    if response.allowed:
        return {
            **state,
            "validator_status": "approved",
            "validator_reason": response.reason
        }
    else:
        return {
            **state,
            "validator_status": "rejected",
            "validator_reason": response.reason
        }

def process_flow(state: GaurdrailState):
    thread_id = state["thread_id"]
    images_list = state["images_list"]
    command = state["command"]
    user_input = state["user_input"]

    if user_input is not None:
        save_message(thread_id, "user", user_input)

        prompt = build_prompt(
            user_input=user_input,
            images_list=images_list,
            thread_id=thread_id
        )
    else:
        user_input = command.update["messages"][0].content
        save_message(thread_id, "user", user_input)

    if command:
        input_payload = command
    else:
        input_payload = {
            "messages": [
                HumanMessage(
                    content=prompt,
                    additional_kwargs={"images": images_list or []}
                )
            ],
            "thread_id": thread_id,
        }

    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_turn",
    }

    # for message_chunk in graph.stream(
    #     input_payload,
    #     config=config,
    #     stream_mode="values",
    # ):
    #     yield message_chunk
    return {
        **state,
        "__delegate__": {
            "graph": "chat",
            "input": {
                "user_input": state["user_input"],
                "thread_id": state["thread_id"],
                "images_list": state["images_list"],
                "command": state["command"],
                "messages": [
                    HumanMessage(
                        content=prompt,
                        additional_kwargs={"images": images_list or []}
                    )
                ]
            }
        }
    }


def review_router(state: GaurdrailState) -> str:
    return state["validator_status"]

def validator_approved_node(state: GaurdrailState):
    print("approved")
    
    # Show appropriate processing message based on request type
    images_list = state.get("images_list", [])
    user_input = state.get("user_input", "").lower()
    
    if images_list and len(images_list) > 0:
        message = "Processing your request. Please wait while I analyze the image...\n\n"
    elif any(keyword in user_input for keyword in ["order", "buy", "purchase", "add"]):
        # Direct order request
        message = "Processing your order...\n\n"
    elif any(keyword in user_input for keyword in ["search", "find", "look"]):
        # Search request
        message = "Searching...\n\n"
    else:
        # General request - show brief processing message
        message = ""
    
    return {
        **state,
        "messages": [
            AIMessage(content=message)
        ] if message else []
    }

def validator_rejected_node(state: GaurdrailState):
    return {
        "messages": [
            AIMessage(content="I can help you with product recognition, shelf analysis, and order management. Please upload a retail image or ask about products to get started.")
        ]
    }

def get_validator_graph():

    gaurd_rail_graph = StateGraph(GaurdrailState)

    gaurd_rail_graph.add_node("validate_request",validate_request)
    gaurd_rail_graph.add_node("approved", validator_approved_node)
    gaurd_rail_graph.add_node("rejected", validator_rejected_node)
    gaurd_rail_graph.add_node("process_flow", process_flow)

    gaurd_rail_graph.set_entry_point("validate_request")
    gaurd_rail_graph.add_conditional_edges(
                "validate_request",
                review_router,
                {
                    "approved": "approved",
                    "rejected": "rejected",
                }
            )
    gaurd_rail_graph.add_edge("approved", "process_flow")
    gaurd_rail_graph.add_edge("rejected", END)
    gaurd_rail_graph = gaurd_rail_graph.compile()

    return gaurd_rail_graph
