import time
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from app.use_case.fetch_shelf_details import FetchShelfDetails
from app.use_case.product_recognition import ProductDetails
from typing import List, Dict
from app.use_case.product_as_object_detection import FetchProductAsObjectDetails
from app.use_case.calculate_empty_shelf_percentage import EmptyShelfPercentageDetails
from app.backend.db import log_model_performance, create_order, get_user_age_by_thread, get_user_role_by_thread

@tool
def calculator( first_num: int, second_num: int, operation: str) -> float:
    """
    Perform operations for given 2 numbers
    Supported operations: add, sub, mul, div

    input fields
        first_num: int
        second_num: int
        operation: str

    response_type
        result: float
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

browser_search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def detect_shelves(image_path: str) -> dict:
    """
    Detect and count shelves in a retail grocery shelf image.

    This tool is used to identify the physical shelf structures
    (horizontal racks/rows) present in a retail or grocery store image.
    It does NOT detect products or product names — only shelf units.

    Typical use cases:
    - Counting the number of shelves in a grocery rack
    - Shelf-level analytics (planogram validation, shelf utilization)
    - Preprocessing step before product or empty-space analysis

    input_fields:
        image_path (str):
            Local file path to a retail shelf image.

    response:
        shelf bounding boxes and metadata
    """
    try:
        print("into detect_shelves")
        start_time = time.time()
        
        request_body = {"file_path": image_path}
        result = FetchShelfDetails().execute(request_body)
        
        # Log model performance
        duration_ms = (time.time() - start_time) * 1000
        try:
            log_model_performance(
                model_name="YOLO_Shelf_Detection",
                operation="detect_shelves",
                duration_ms=duration_ms,
                input_size=image_path
            )
        except:
            pass  # Don't fail if telemetry fails
        
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        print("exception is -> ",e)
        return {"status": "error", "message": str(e)}


@tool
def detect_products(image_path: str) -> dict:
    """
    Detect products present in a retail shelf image and return their count.

    This tool detects physical product objects (boxes, bottles, packs, etc.)
    present on shelves. It ONLY detects the presence and bounding boxes
    of products and returns their count.

    Important:
    - This tool does NOT recognize or identify product names or brands.

    Typical use cases:
    - Product density analysis
    - Shelf occupancy measurement
    - Supporting empty-shelf or availability calculations

    input_fields:
        image_path (str):
            Local file path to a retail shelf image.

    response:
        product bounding boxes and count
    """
    try:
        start_time = time.time()
        
        request_body = {"file_path": image_path}
        result = FetchProductAsObjectDetails().execute(request_body)
        
        # Log model performance
        duration_ms = (time.time() - start_time) * 1000
        try:
            log_model_performance(
                model_name="YOLO_Product_Detection",
                operation="detect_products",
                duration_ms=duration_ms,
                input_size=image_path
            )
        except:
            pass
        
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@tool
def calculate_empty_shelf_percentage(image_path: str) -> dict:
    """
    Calculate the percentage of empty shelf space in a retail shelf image.

    This tool analyzes shelf space versus detected products to determine
    how much shelf area is empty or missing products.

    It helps quantify shelf availability and out-of-stock scenarios.

    Typical use cases:
    - Empty shelf detection
    - Out-of-stock monitoring
    - Retail compliance and merchandising analysis

    input_fields:
        image_path (str):
            Local file path to a retail shelf image.

    response:
        empty percentage, capacity, missing products
    """
    try:
        start_time = time.time()
        
        request_body = {"file_path": image_path}
        result = EmptyShelfPercentageDetails().execute(request_body)
        
        # Log model performance
        duration_ms = (time.time() - start_time) * 1000
        try:
            log_model_performance(
                model_name="Empty_Shelf_Calculator",
                operation="calculate_empty_percentage",
                duration_ms=duration_ms,
                input_size=image_path
            )
        except:
            pass
        
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@tool
def recognize_products(
    image_paths: List[str],
    request_id: str
) -> dict:
    """
    Recognize and identify products across multiple retail shelf images.

    Unlike detect_products, this tool performs full product recognition.
    It identifies product names / SKUs and aggregates their counts
    across all provided images.

    This tool answers:
    - What products are present?
    - How many times does each product appear across all images?

    Typical use cases:
    - Product recognition and catalog matching
    - Brand / SKU-level shelf analysis
    - Inventory and assortment tracking

    input_fields:
        image_paths (List[str]):
            List of local file paths to retail shelf images.
        request_id (str):
            Unique identifier for logging, tracing, and debugging.

    response:
        product names, confidence scores
    """
    try:
        start_time = time.time()
        
        request_body = {
            "file_paths_list": image_paths,
            "request_id": request_id
        }
        result = ProductDetails().execute(request_body)
        
        # Log model performance
        duration_ms = (time.time() - start_time) * 1000
        try:
            log_model_performance(
                model_name="SimCLR_Product_Recognition",
                operation="recognize_products",
                duration_ms=duration_ms,
                input_size=f"{len(image_paths)} images"
            )
        except:
            pass
        
        return {
            "status": "success",
            "request_id": request_id,
            "data": result
        }
    except Exception as e:
        print("exception -> ",e)
        return {
            "status": "error",
            "request_id": request_id,
            "message": str(e)
        }


# Valid product categories that can be ordered
VALID_PRODUCTS = [
    "Alcohol", "Candy", "Canned Food", "Chocolate", "Dessert",
    "Dried Food", "Dried Fruit", "Drink", "Gum", "Instant Drink",
    "Instant Noodles", "Milk", "Personal Hygiene",
    "Puffed Food", "Seasoner", "Stationery", "Tissue"
]

# Health warning thresholds for customers
HEALTH_WARNING_THRESHOLDS_DIRECT = {
    "Alcohol": 1,
    "Candy": 3,
    "Chocolate": 3,
    "Dessert": 2,
    "Puffed Food": 2,
    "Instant Noodles": 3,
    "Drink": 3,
}

# Health warning messages
HEALTH_WARNINGS_MESSAGES = {
    "Alcohol": "Please consume alcohol responsibly and in moderation.",
    "Candy": "High sugar intake may affect your health. Consider moderation.",
    "Chocolate": "Chocolate in large quantities may contribute to sugar intake.",
    "Dessert": "Desserts are high in sugar. Enjoy in moderation for a balanced diet.",
    "Puffed Food": "Puffed snacks are often high in sodium. Consider balancing with healthier options.",
    "Instant Noodles": "Instant noodles are high in sodium. Balance with vegetables and protein.",
    "Drink": "Sugary drinks can contribute to excess calorie intake.",
}

@tool
def place_direct_order(
    products: Dict[str, int],
    user_id: str
) -> dict:
    """
    Place a direct order for products without image detection.
    
    Use this tool when a customer wants to order specific products directly,
    without uploading an image for product detection.
    
    Examples of when to use:
    - "Order 2 candy"
    - "I want to buy 5 chocolates"
    - "Add 3 instant noodles to my order"
    - "Place an order for 10 drinks"
    
    input_fields:
        products (Dict[str, int]):
            Dictionary of product names and quantities.
            Example: {"Candy": 2, "Chocolate": 5}
            
            Valid product categories (ONLY these can be ordered):
            - Alcohol, Candy, Canned Food, Chocolate, Dessert
            - Dried Food, Dried Fruit, Drink, Gum, Instant Drink
            - Instant Noodles, Milk, Personal Hygiene
            - Puffed Food, Seasoner, Stationery, Tissue
            
        user_id (str):
            The user/thread ID for the order.
    
    response:
        Order confirmation with order ID and items ordered.
    """
    try:
        print(f"[DIRECT_ORDER] Placing order for user: {user_id}")
        print(f"[DIRECT_ORDER] Products requested: {products}")
        
        if not products:
            return {
                "status": "error",
                "message": "No products specified. Please tell me what you'd like to order."
            }
        
        # Get user role and age for validation
        user_role = get_user_role_by_thread(user_id)
        user_age = get_user_age_by_thread(user_id)
        
        print(f"[DIRECT_ORDER] User role: {user_role}, Age: {user_age}")
        
        # Normalize product names and validate against valid products
        normalized_products = {}
        invalid_products = []
        valid_products_lower = {p.lower(): p for p in VALID_PRODUCTS}
        
        for product, qty in products.items():
            product_lower = product.strip().lower()
            
            # Check if product exists in valid products list
            if product_lower in valid_products_lower:
                normalized_name = valid_products_lower[product_lower]  # Use correct casing
                if qty > 0:
                    normalized_products[normalized_name] = qty
            else:
                invalid_products.append(product.strip())
        
        # Report invalid products
        if invalid_products:
            valid_list = ", ".join(VALID_PRODUCTS)
            if not normalized_products:
                return {
                    "status": "error",
                    "message": f"Invalid product(s): {', '.join(invalid_products)}.\n\nAvailable products are: {valid_list}"
                }
            else:
                # Some products are valid, continue with those
                print(f"[DIRECT_ORDER] Invalid products ignored: {invalid_products}")
        
        # For customers, check alcohol restrictions
        if user_role != "store_manager":
            alcohol_qty = normalized_products.get("Alcohol", 0)
            if alcohol_qty > 0 and user_age < 21:
                # Remove alcohol from order
                del normalized_products["Alcohol"]
                if not normalized_products:
                    return {
                        "status": "error",
                        "message": "Order cannot be placed. Alcohol is not available for customers under 21 years old, and no other products were requested."
                    }
                else:
                    # Create order without alcohol
                    order_id = create_order(user_id=user_id, products=normalized_products)
                    order_details = "\n".join([f"- {name}: {qty}" for name, qty in normalized_products.items()])
                    return {
                        "status": "success",
                        "order_id": order_id,
                        "message": f"Note: Alcohol removed from order (not available for customers under 21).\n\nOrder created successfully!\n\nOrder ID: {order_id}\n\nItems ordered:\n{order_details}"
                    }
        
        # Create the order
        order_id = create_order(user_id=user_id, products=normalized_products)
        print(f"[DIRECT_ORDER] Order created with ID: {order_id}")
        
        order_details = "\n".join([f"- {name}: {qty}" for name, qty in normalized_products.items()])
        
        # Generate health warnings for customers
        health_warnings = []
        if user_role != "store_manager":
            for product, qty in normalized_products.items():
                threshold = HEALTH_WARNING_THRESHOLDS_DIRECT.get(product)
                if threshold and qty >= threshold:
                    warning_msg = HEALTH_WARNINGS_MESSAGES.get(product)
                    if warning_msg:
                        health_warnings.append(f"- {product}: {warning_msg}")
        
        # Build response message
        response_msg = f"Order created successfully!\n\nOrder ID: {order_id}\n\nItems ordered:\n{order_details}"
        
        if health_warnings:
            response_msg += f"\n\n**Health Reminders:**\n" + "\n".join(health_warnings)
        
        return {
            "status": "success",
            "order_id": order_id,
            "message": response_msg
        }
        
    except Exception as e:
        print(f"[DIRECT_ORDER] Error: {e}")
        return {
            "status": "error",
            "message": f"Failed to create order: {str(e)}"
        }


tools_list = [
    calculator, 
    browser_search_tool,
    detect_shelves,
    # detect_products,
    calculate_empty_shelf_percentage,
    recognize_products,
    place_direct_order,
    ]
