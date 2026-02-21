import psycopg
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition


from app.backend.state import ChatState
from app.config import POSTGRES_URI, OPENAI_API_KEY
from app.backend.tools import tools_list
import tiktoken

import tiktoken
import json
from ast import literal_eval
from datetime import datetime
from dotenv import load_dotenv
from app.backend.db import mark_waiting_for_review, clear_waiting_for_review, is_waiting_for_review, create_order, log_model_performance, get_user_age_by_thread, get_user_role_by_thread, log_model_verification, compare_model_outputs
import re
import time
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.types import interrupt

load_dotenv()


def count_tokens_and_log(messages, tools):
    """Count and log token usage for debugging"""
    encoding = tiktoken.encoding_for_model("gpt-4-turbo")
    
    # Create detailed analysis
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "messages": [],
        "tools": [],
        "summary": {}
    }
    
    # Count message tokens with detailed breakdown
    message_tokens = 0
    for i, msg in enumerate(messages):
        content = str(msg.content) if hasattr(msg, 'content') else str(msg)
        msg_tokens = len(encoding.encode(content))
        message_tokens += msg_tokens
        
        msg_analysis = {
            "index": i,
            "type": type(msg).__name__,
            "role": getattr(msg, 'role', 'unknown') if hasattr(msg, 'role') else 'unknown',
            "tokens": msg_tokens,
            "character_count": len(content),
            "content": content
        }
        analysis["messages"].append(msg_analysis)
    
    # Count tool tokens
    tool_tokens = 0
    for i, tool in enumerate(tools):
        tool_str = str(tool)
        individual_tokens = len(encoding.encode(tool_str))
        tool_tokens += individual_tokens
        
        tool_analysis = {
            "index": i,
            "name": getattr(tool, 'name', 'unknown'),
            "tokens": individual_tokens,
            "character_count": len(tool_str),
            "content": tool_str
        }
        analysis["tools"].append(tool_analysis)
    
    total_tokens = message_tokens + tool_tokens
    
    analysis["summary"] = {
        "message_tokens": message_tokens,
        "tool_tokens": tool_tokens,
        "total_tokens": total_tokens,
        "limit": 16385,
        "over_limit": total_tokens > 16385
    }
    
    # Save to file
    debug_file = f"token_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(debug_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    print(f"🔍 TOKEN ANALYSIS:")
    print(f"  📝 Message tokens: {message_tokens}")
    print(f"  🔧 Tool tokens: {tool_tokens}")
    print(f"  📊 Total tokens: {total_tokens}")
    print(f"  Limit: 16,385 tokens")
    print(f"  {'OVER LIMIT' if total_tokens > 16385 else 'Within limit'}")
    print(f"  📄 Detailed analysis saved to: {debug_file}")
    
    return total_tokens

WARNING_THRESHOLDS = {
    "Alcohol": 0,
    "Candy": 2,
    "Canned Food": 5,
    "Chocolate": 2,
    "Dessert": 3,
    "Dried Food": 4,
    "Dried Fruit": 3,
    "Drink": 4,
    "Gum": 5,
    "Instant Drink": 2,
    "Instant Noodles": 1,
    "Milk": 3,
    "Personal Hygiene": 5,
    "Puffed Food": 1,
    "Seasoner": 4,
    "Stationery": 10,
    "Tissue": 3
}

# Target stock levels for store manager restocking
# Order quantity = RESTOCK_TARGET - detected_items
RESTOCK_TARGETS = {
    "Alcohol": 10,
    "Candy": 15,
    "Canned Food": 20,
    "Chocolate": 15,
    "Dessert": 12,
    "Dried Food": 15,
    "Dried Fruit": 12,
    "Drink": 20,
    "Gum": 15,
    "Instant Drink": 15,
    "Instant Noodles": 20,
    "Milk": 15,
    "Personal Hygiene": 15,
    "Puffed Food": 12,
    "Seasoner": 15,
    "Stationery": 20,
    "Tissue": 15
}


# -----------------------------
# LLM
# -----------------------------
llm = ChatOpenAI(
    model="gpt-5-nano",
    # model="gpt-3.5-turbo-0125",
    # model = "gpt-4-turbo",
    temperature=0.2,
    api_key=OPENAI_API_KEY
)

# -----------------------------
# Chat Node
# -----------------------------

def parse_llm_detected_items(llm_content: str) -> dict:
    """
    Parse LLM response to extract detected items.
    
    Looks for patterns like:
    - Category × Quantity
    - Category: Quantity
    - - Category × Quantity
    
    Returns dict like {"Candy": 3, "Alcohol": 2}
    """
    detected = {}
    
    if not llm_content:
        return detected
    
    # Pattern 1: "- Category × Quantity" or "Category × Quantity"
    pattern1 = r'[-•]?\s*([A-Za-z\s]+?)\s*[×x]\s*(\d+)'
    matches = re.findall(pattern1, llm_content, re.IGNORECASE)
    for category, qty in matches:
        category = category.strip().title()
        # Normalize common variations
        category = category.replace("Dried Food", "Dried food").replace("Canned Food", "Canned food")
        if category and qty:
            detected[category] = int(qty)
    
    # Pattern 2: "Category: Quantity" (for lists)
    pattern2 = r'[-•]?\s*([A-Za-z\s]+?):\s*(\d+)'
    matches2 = re.findall(pattern2, llm_content, re.IGNORECASE)
    for category, qty in matches2:
        category = category.strip().title()
        if category and qty and category not in detected:
            detected[category] = int(qty)
    
    return detected


def chatbot_node(state: ChatState):
    messages = state["messages"].copy()
    thread_id = state["thread_id"]
    
    # Get user_role from state, or fetch from DB if not available
    user_role = state.get("user_role")
    if not user_role:
        user_role = get_user_role_by_thread(thread_id)
    
    print("thread id is -> ",thread_id)
    print("user_role in chatbot_node -> ", user_role)
    
    # General system prompt for handling all requests
    GENERAL_SYSTEM_PROMPT = f"""
    You are a retail checkout assistant. Your primary job is to help users with:
    
    1. **Direct Orders** (NO image required):
       When the user says things like "order 2 candy", "buy 5 chocolates", "I want 3 drinks":
       - Immediately use the `place_direct_order` tool
       - Extract the product name and quantity from the user's message
       - Use the user_id from the USER_CONTEXT in the message
       - Valid products: Alcohol, Candy, Canned Food, Chocolate, Dessert, Dried Food, Dried Fruit, 
         Drink, Gum, Instant Drink, Instant Noodles, Milk, Personal Hygiene, Puffed Food, Seasoner, 
         Stationery, Tissue
       
    2. **Image-based Analysis** (Image required):
       When the user uploads an image and asks to recognize products, count shelves, etc.:
       - Use `recognize_products`, `detect_shelves`, or `calculate_empty_shelf_percentage` tools
       
    3. **General Questions**: Answer politely and offer help.
    
    IMPORTANT: 
    - For direct orders like "order X <product>", DO NOT ask for clarification. 
    - DO NOT respond with a greeting or list of capabilities.
    - Just call the `place_direct_order` tool directly with the extracted product and quantity.
    """
    
    # Different system prompts based on user role
    SYSTEM_HEALTH_PROMPT_CUSTOMER = f"""
    You are a health-aware retail assistant.

You must produce a response with EXACTLY this structure:

**Detected Items** \n\n

- <Category> × <Quantity>

    **Health Warnings** \n\n
<only include items that need warnings>
- <Category> (<Quantity>): <warning text>

Rules:
- EVERY Category line MUST start with a hyphen (-)
- Use the hyphen (-) exactly so the output can be rendered as a Streamlit list
- Always show Detected Items
- Only show Health Warnings if at least one exists
- Do NOT warn for non-food items (e.g. Tissue, Stationery, Personal Hygiene)
- Health warnings must be 1–2 sentences
- General wellness only
- No medical advice, diagnosis, or fear-based language
- Mention moderation or balance

Do not use bullet points.
Do not add extra sections.

    """
    
    SYSTEM_PROMPT_MANAGER = f"""
    You are a retail inventory assistant helping a store manager with restocking.

    You must produce a response with EXACTLY this structure:

    **Detected Items** \n\n

    - <Category> × <Quantity>

    Rules:
    - EVERY Category line MUST start with a hyphen (-)
    - Use the hyphen (-) exactly so the output can be rendered as a Streamlit list
    - Always show Detected Items
    - Do NOT show any health warnings (this is for store inventory, not personal consumption)
    - Keep the response professional and focused on inventory

    Do not use bullet points.
    Do not add extra sections.

    """
    
    print("\n")
    print("messages in chatbot_node -> ", messages)
    print("\n")


    # Only add detected items prompt if:
    # 1. We have detected items from image analysis
    # 2. tools_done is True (tools just completed)
    # 3. The last message was from image-based analysis (has images in kwargs)
    last_msg = messages[-1] if messages else None
    has_recent_images = (
        last_msg and 
        isinstance(last_msg, HumanMessage) and 
        last_msg.additional_kwargs.get("images")
    )
    
    # Check if any recent message in the conversation had images
    has_image_context = any(
        isinstance(m, HumanMessage) and m.additional_kwargs.get("images")
        for m in messages[-5:] if isinstance(m, HumanMessage)
    )
    
    if state.get("detected_items") and state.get("tools_done") and has_image_context:
        # Use appropriate system prompt based on role
        if user_role == "store_manager":
            messages.insert(0, SystemMessage(content=SYSTEM_PROMPT_MANAGER))
            messages.append(
                HumanMessage(
                    content=f"""
Detected items on shelf:
{state["detected_items"]}

This is a store inventory scan. Show only detected items, no health warnings needed.
"""
                )
            )
        else:
            # Customer flow with health warnings
            messages.insert(0, SystemMessage(content=SYSTEM_HEALTH_PROMPT_CUSTOMER))
        messages.append(
            HumanMessage(
                content=f"""
Detected items:
{state["detected_items"]}

Items requiring health warnings:
{state.get("health_warning_input", [])}
"""
            )
        )
    else:
        # For non-image requests (direct orders, general queries)
        # Use the general system prompt to guide the LLM
        messages.insert(0, SystemMessage(content=GENERAL_SYSTEM_PROMPT))

    # Track LLM performance
    llm_start_time = time.time()
    response = llm.bind_tools(tools_list).invoke(messages)
    llm_end_time = time.time()
    
    # Log LLM performance
    llm_duration_ms = (llm_end_time - llm_start_time) * 1000
    try:
        log_model_performance(
            model_name="gpt_llm",
            duration_ms=llm_duration_ms,
            operation="chat_completion",
            input_size=str(len(messages)),
            thread_id=state.get("thread_id"),
            metadata={
                "model": "gpt-5-nano",
                "message_count": len(messages),
                "has_tools": True
            }
        )
    except Exception as e:
        print(f"Failed to log LLM performance: {e}")

    # --- Model Verification: Compare Inference vs LLM Output ---
    # Only verify after image-based product recognition (when tools_done and detected_items exist)
    if state.get("tools_done") and state.get("detected_items") and has_image_context:
        try:
            inference_output = state.get("detected_items", {})
            llm_content = response.content if hasattr(response, 'content') else ""
            
            # Parse LLM response to extract reported items
            llm_output = parse_llm_detected_items(llm_content)
            
            # Compare outputs
            match_status, mismatched_items = compare_model_outputs(inference_output, llm_output)
            
            # Log verification
            log_model_verification(
                thread_id=state.get("thread_id"),
                inference_output=inference_output,
                llm_output=llm_output,
                match_status=match_status,
                mismatched_items=mismatched_items if not match_status else None
            )
            
            print(f"[VERIFICATION] Match: {match_status}, Inference: {inference_output}, LLM: {llm_output}")
            if not match_status:
                print(f"[VERIFICATION] Mismatches: {mismatched_items}")
                
        except Exception as e:
            print(f"[VERIFICATION] Error logging verification: {e}")

    return {
        "messages": [response],
        "tools_done": state.get("tools_done", False)
    }


def review_interrupt_node(state: ChatState):
    thread_id = state["thread_id"]
    user_role = state.get("user_role", "customer")
    detected_items = state.get("detected_items", {})
    
    # Mark DB state
    mark_waiting_for_review(thread_id)
    print(f"starting interrupt - user_role: {user_role}")
    
    # Different confirmation messages based on role
    if user_role == "store_manager":
        # Calculate restock quantities for store manager
        restock_items = []
        for product, detected_qty in detected_items.items():
            target = RESTOCK_TARGETS.get(product, 10)  # Default target is 10
            restock_qty = max(0, target - detected_qty)
            if restock_qty > 0:
                restock_items.append(f"- {product}: {restock_qty} units (current: {detected_qty}, target: {target})")
        
        if restock_items:
            restock_summary = "\n".join(restock_items)
            user_feedback_query = f"**Restock Order Summary:**\n{restock_summary}\n\nWould you like to place this restock order? Please respond with yes or no."
        else:
            user_feedback_query = "All products are adequately stocked. No restock order needed. Would you like to proceed anyway? Please respond with yes or no."
    else:
        # Customer message
        user_feedback_query = "Would you like to place an order for these items? Please respond with yes or no."
    
    interrupt(
        {
            "question": user_feedback_query
        }
    )
    print("Done interrupt")

def review_decision_node(state: ChatState):
    print("into review decision")
    print(state["messages"])
    last_message = state["messages"][-1].content.lower().strip()
    thread_id = state["thread_id"]

    clear_waiting_for_review(thread_id)

    if "yes" in last_message:
        return {"decision": "approved"}
    elif "no" in last_message:
        return {"decision": "rejected"}
    else:
        return {"decision": "rejected"}  # Default to rejected if unclear

def approved_node(state: ChatState):
    print("=== ORDER APPROVED - Proceeding to create order ===")
    # Don't add a message here - let create_order_node handle the response
    return {
        "tools_done": False
    }

def rejected_node(state: ChatState):
    return {
        "messages": [
            AIMessage(content="Order Creation Cancelled.")
        ],
        "tools_done": False
    }

def review_router(state: dict):
    return state["decision"]

def chatbot_router(state: ChatState):
    print("\n")
    print("state in chatbot router -> ", state)
    print("\n")
    tool_route = tools_condition(state)

    if tool_route:
        # Case 1: plain string
        if isinstance(tool_route, str):
            if tool_route == "tools":
                return "tools"

        # Case 2: tuple or list (take first element)
        if isinstance(tool_route, (list, tuple)) and tool_route:
            first = tool_route[0]
            if isinstance(first, str) and first == "tools":
                return "tools"

            # Duck-type Send without importing it
            if hasattr(first, "to") and first.to == "tools":
                return "tools"

        # Case 3: duck-type Send directly
        if hasattr(tool_route, "to") and tool_route.to == "tools":
            return "tools"

    # --- Check for direct order completion FIRST ---
    # If place_direct_order was just called, go straight to END
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, ToolMessage):
            if msg.name == "place_direct_order":
                print("[ROUTER] Direct order completed, going to END")
                return END
            break  # Only check the last tool message
    
    # --- Review after image-based tools ---
    # Only go to review if:
    # 1. tools_done is True (from recognize_products)
    # 2. We have detected items
    if state.get("tools_done", False) and not is_waiting_for_review(state["thread_id"]):
        detected_items = state.get("detected_items", {})
        if detected_items and len(detected_items) > 0:
            return "needs_review"
        else:
            print("No items detected, skipping review")

    print("Going to end")
    # --- End ---
    return END


def tools_done_node(state: ChatState):
    warnings_input = []
    detected_items = {}
    has_successful_detection = False
    thread_id = state["thread_id"]
    
    # Get user role to determine if health warnings apply
    user_role = get_user_role_by_thread(thread_id)
    print(f"User role: {user_role}")
    print("state['messages']-> ", state["messages"])
    
    # Check for direct order - skip processing
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            if msg.name == "place_direct_order":
                print("[TOOLS_DONE] Direct order tool, skipping state updates")
                return {"user_role": user_role}
            break
    
    # Find the latest tool message for recognize_products
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage) and msg.name == "recognize_products":
            try:
                payload = json.loads(msg.content)
                
                if payload.get("status") == "success" and "data" in payload:
                    data = payload["data"]
                    
                    if isinstance(data, dict):
                        if "products_count" in data:
                            detected_items = data["products_count"]
                        elif all(isinstance(v, (int, float)) for v in data.values()):
                            detected_items = data
                        else:
                            detected_items = data
                    
                    if detected_items and len(detected_items) > 0:
                        has_successful_detection = True
                    
                    products = detected_items if isinstance(detected_items, dict) else {}
                    
                    if user_role != "store_manager":
                        for category, qty in products.items():
                            if isinstance(qty, (int, float)):
                                threshold = WARNING_THRESHOLDS.get(category)
                                if threshold is not None and qty > threshold:
                                    warnings_input.append({
                                        "category": category,
                                        "quantity": int(qty)
                                    })
                                    
                elif payload.get("status") == "error":
                    print(f"Tool returned error: {payload.get('message', 'Unknown error')}")
                    has_successful_detection = False
                    
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                print(f"Error parsing tool message: {e}")
                has_successful_detection = False
            break
    
    print("warnings_input is -> ", warnings_input)
    print(f"has_successful_detection: {has_successful_detection}, detected_items: {detected_items}")
    
    # Only set tools_done=True if we successfully detected products
    return {
        "tools_done": has_successful_detection,
        "detected_items": detected_items, 
        "health_warning_input": warnings_input,
        "user_role": user_role
    }

def preprocess_node(state: ChatState):
    """
    Reset per-request tool state for new requests.
    This prevents old detected_items from showing in new conversations.
    
    IMPORTANT: Must RETURN state changes for LangGraph checkpointer to persist them.
    """
    messages = state["messages"]

    if messages:
        last = messages[-1]

        # Reset state for any new HumanMessage (both image and text-only requests)
        if isinstance(last, HumanMessage):
            # Always clear detection state for new requests
            # This ensures direct orders don't mix with previous image scans
            print("[PREPROCESS] Clearing detection state for new request")
            return {
                "detected_items": {},
                "health_warning_input": [],
                "tools_done": False
            }

    # No changes needed
    return {}

def parse_user_order_request(message: str, detected_items: dict) -> dict:
    """
    Use LLM to parse user's free text message and extract requested products and quantities.
    Example: "I want to order 3 bottles of alcohol and one chocolate" -> {'Alcohol': 3, 'Chocolate': 1}
    """
    
    # Known product categories
    known_products = [
        "Alcohol", "Candy", "Canned Food", "Chocolate", "Dessert", 
        "Dried Food", "Dried Fruit", "Drink", "Gum", "Instant Drink", 
        "Instant Noodles", "Milk", "Personal Hygiene", "Puffed Food", 
        "Seasoner", "Stationery", "Tissue"
    ]
    
    system_prompt = f"""You are a product order parser. Extract products and quantities from the user's message.

Available product categories: {', '.join(known_products)}

Rules:
1. Match user's mentioned products to the closest available category (case-insensitive)
2. Extract the quantity for each product (default to 1 if not specified)
3. If user says "all" or "everything" or "yes", return exactly: {{"all": true}}
4. Return ONLY a valid JSON object with product names as keys and quantities as integer values
5. Use the exact category names from the available list (with proper capitalization)
6. If no products are mentioned or the message is unclear, return an empty object: {{}}

Examples:
- "order 3 alcohol and 1 chocolate" -> {{"Alcohol": 3, "Chocolate": 1}}
- "I want two desserts and 5 candies" -> {{"Dessert": 2, "Candy": 5}}
- "give me some milk" -> {{"Milk": 1}}
- "order all items" -> {{"all": true}}
- "yes please" -> {{}}
- "I'd like to get 3 beers and a sweet treat" -> {{"Alcohol": 3, "Dessert": 1}}

Respond with ONLY the JSON object, no explanation."""

    try:
        # Create a simple LLM instance for parsing (low temperature for consistency)
        parser_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=OPENAI_API_KEY
        )
        
        response = parser_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User message: {message}")
        ])
        
        # Parse the JSON response
        response_text = response.content.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        parsed = json.loads(response_text)
        
        print(f"LLM parsed order: {parsed}")
        
        # Handle "all" case
        if parsed.get("all") == True:
            return detected_items.copy()
        
        # Ensure all values are integers
        result = {}
        for key, value in parsed.items():
            if key != "all" and isinstance(value, (int, float)):
                result[key] = int(value)
        
        return result
        
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error parsing order with LLM: {e}")
        # Fallback: return empty dict (will use all detected items)
        return {}

def validate_order_against_detected(requested: dict, detected: dict) -> tuple[bool, str, dict]:
    """
    Validate that requested products don't exceed detected quantities.
    Returns: (is_valid, error_message, validated_products)
    """
    validated_products = {}
    errors = []
    
    # Create lowercase mapping for case-insensitive comparison
    detected_lower = {k.lower(): (k, v) for k, v in detected.items()}
    
    for product, requested_qty in requested.items():
        product_lower = product.lower()
        
        if product_lower not in detected_lower:
            errors.append(f"'{product}' was not detected in the image")
            continue
        
        original_name, available_qty = detected_lower[product_lower]
        
        if requested_qty > available_qty:
            errors.append(f"'{original_name}': requested {requested_qty}, but only {available_qty} detected")
        else:
            validated_products[original_name] = requested_qty
    
    if errors:
        return False, "\n".join(errors), validated_products
    
    return True, "", validated_products


def create_order_node(state: ChatState):
    print("\n=== CREATE_ORDER_NODE STARTED ===")
    
    detected_items = state.get("detected_items", {})
    thread_id = state["thread_id"]
    user_role = state.get("user_role")
    
    # Fetch role from DB if not in state
    if not user_role:
        user_role = get_user_role_by_thread(thread_id)
    
    print(f"detected_items: {detected_items}")
    print(f"thread_id: {thread_id}")
    print(f"user_role: {user_role}")
    
    # If no detected items, cannot create order
    if not detected_items:
        print("ERROR: No detected items")
        return {
            "messages": [
                AIMessage(content="Order creation failed. No products were detected from the image.")
            ]
        }
    
    # =============================================
    # STORE MANAGER FLOW - Restock Order
    # =============================================
    if user_role == "store_manager":
        print("Processing STORE MANAGER restock order")
        
        # Calculate restock quantities: target - detected
        restock_products = {}
        for product, detected_qty in detected_items.items():
            target = RESTOCK_TARGETS.get(product, 10)  # Default target is 10
            restock_qty = max(0, target - detected_qty)
            if restock_qty > 0:
                restock_products[product] = restock_qty
        
        print(f"Restock products to order: {restock_products}")
        
        if not restock_products:
            return {
                "messages": [
                    AIMessage(content="No restock order needed. All products are at or above target stock levels.")
                ]
            }
        
        try:
            # Create the restock order (no age restrictions for managers)
            order_id = create_order(user_id=thread_id, products=restock_products)
            print(f"Restock order created with ID: {order_id}")
            
            order_details = "\n".join([
                f"- {name}: {qty} units" for name, qty in restock_products.items()
            ])
            
            return {
                "messages": [
                    AIMessage(content=f"Restock order has been created.\n\nOrder ID: {order_id}\n\nItems ordered for restocking:\n{order_details}")
                ]
            }
        except Exception as e:
            print(f"ERROR creating restock order: {e}")
            return {
                "messages": [
                    AIMessage(content=f"Failed to create restock order. Error: {str(e)}")
                ]
            }
    
    # =============================================
    # CUSTOMER FLOW - Regular Order
    # =============================================
    
    # Get the user's order request message (the message before the confirmation)
    user_order_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            content = msg.content.lower().strip()
            # Skip confirmation messages (yes/no)
            if content not in ['yes', 'no', 'y', 'n']:
                user_order_message = msg.content
                break
    
    print("User order message is -> ", user_order_message)
    
    # Parse user's requested products from their message
    requested_products = parse_user_order_request(user_order_message, detected_items)
    
    print("\n", "requested_products are -> ", requested_products, "\n")
    
    # If no specific products requested, use all detected items
    if not requested_products:
        requested_products = detected_items.copy()
        print("No specific products parsed, using all detected items")
    
    # Validate requested products against detected items (customers can't order more than detected)
    is_valid, error_message, validated_products = validate_order_against_detected(
        requested_products, detected_items
    )
    
    if not is_valid:
        return {
            "messages": [
                AIMessage(content=f"Order creation cancelled. You have requested more products than detected:\n{error_message}\n\nPlease adjust your order quantities.")
            ]
        }
    
    if not validated_products:
        return {
            "messages": [
                AIMessage(content="Order creation cancelled. No valid products found in your request.")
            ]
        }
    
    # Age restriction check for alcohol (CUSTOMERS ONLY)
    alcohol_quantity = 0
    for item_name, qty in validated_products.items():
        if item_name.lower() == "alcohol":
            alcohol_quantity = qty
            break
    
    if alcohol_quantity > 0:
        user_age = get_user_age_by_thread(thread_id)
        print("\n", "user_age is -> ", user_age, "\n")
        if user_age < 21:
            # Remove alcohol from the order but allow other items
            validated_products = {k: v for k, v in validated_products.items() if k.lower() != "alcohol"}
            if not validated_products:
                return {
                    "messages": [
                        AIMessage(content="Order creation failed. Alcohol cannot be sold to customers under 21 years old, and no other products were requested.")
                    ]
                }
            else:
                # Create order without alcohol
                order_id = create_order(user_id=thread_id, products=validated_products)
                return {
                    "messages": [
                        AIMessage(content=f"**Note:** Alcohol removed from order (not available for customers under 21).\n\nOrder created successfully with order_id {order_id}.\n\n**Ordered Products:**\n" + 
                                  "\n".join([f"- {name}: {qty}" for name, qty in validated_products.items()]))
                    ]
                }
    
    # Create the order with validated products
    order_id = create_order(user_id=thread_id, products=validated_products)
    
    return {
        "messages": [
            AIMessage(content=f"Order created successfully with order_id {order_id}.\n\n**Ordered Products:**\n" + 
                      "\n".join([f"- {name}: {qty}" for name, qty in validated_products.items()]))
        ]
    }

# -----------------------------
# SINGLETONS
# -----------------------------
_graph = None
_checkpointer = None
_pg_conn = None




def get_graph():
    global _graph, _checkpointer, _pg_conn

    if _graph is None:
        # -----------------------------
        # DB + CHECKPOINTER
        # -----------------------------
        _pg_conn = psycopg.connect(POSTGRES_URI)
        _pg_conn.autocommit = True

        _checkpointer = PostgresSaver(_pg_conn)
        _store = PostgresStore(_pg_conn)
        _store.setup()
        _checkpointer.setup()

        # -----------------------------
        # GRAPH
        # -----------------------------
        graph = StateGraph(ChatState)

        tool_node = ToolNode(tools_list)

        # -----------------------------
        # NODES
        # -----------------------------
        graph.add_node("chatbot", chatbot_node)
        graph.add_node("tools", tool_node)
        graph.add_node("tools_done", tools_done_node)
        graph.add_node("review_interrupt", review_interrupt_node)
        graph.add_node("review_decision", review_decision_node)
        graph.add_node("approved", approved_node)
        graph.add_node("rejected", rejected_node)
        graph.add_node("preprocess", preprocess_node)
        graph.add_node("create_retail_order", create_order_node)


        # -----------------------------
        # ENTRY
        # -----------------------------
        graph.set_entry_point("preprocess")
        graph.add_edge("preprocess", "chatbot")

        # -----------------------------
        # ROUTING (IMPORTANT)
        # -----------------------------
        graph.add_conditional_edges(
            "chatbot",
            chatbot_router,
            {
                "tools": "tools",
                "needs_review": "review_interrupt",
                END: END,
            }
        )

        # -----------------------------
        # TOOLS LOOP
        # -----------------------------
        graph.add_edge("tools", "tools_done")
        graph.add_edge("tools_done", "chatbot")

        # -----------------------------
        # REVIEW FLOW
        # -----------------------------
        graph.add_edge("review_interrupt", "review_decision")

        graph.add_conditional_edges(
            "review_decision",
            review_router,
            {
                "approved": "approved",
                "rejected": "rejected",
            }
        )

        graph.add_edge("approved", "create_retail_order")
        graph.add_edge("create_retail_order", END)
        graph.add_edge("rejected", END)

        # -----------------------------
        # COMPILE
        # -----------------------------
        _graph = graph.compile(checkpointer=_checkpointer)

    return _graph

