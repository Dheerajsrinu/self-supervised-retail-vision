import streamlit as st
import os
from ultralytics import YOLO
from app import model_store

@st.cache_resource(show_spinner="Loading ML models...")
def load_models():
    """
    This runs ONCE per Streamlit server start.
    Equivalent to FastAPI @app.on_event("startup")
    """
    
    try:
        shelf_detector_v14_path = "models/shelf_detector_v14/weights/best.pt"
        if os.path.exists(shelf_detector_v14_path):
            model_store.shelf_detector = YOLO(shelf_detector_v14_path)
            print(f"Loaded shelf_detector from {shelf_detector_v14_path}")
        else:
            print(f"WARNING: Shelf detector model not found at {shelf_detector_v14_path}")

        product_model_path = "models/product_recognition_yolo11/weights/best.pt"
        if os.path.exists(product_model_path):
            model_store.product_object_model = YOLO(product_model_path)
            print(f"Loaded product_object_model from {product_model_path}")
        else:
            print(f"WARNING: Product object model not found at {product_model_path}")

        product_rec_model_path = "models/rpc_yolov11_4dh3/weights/best.pt"
        if os.path.exists(product_rec_model_path):
            model_store.product_rec_model = YOLO(product_rec_model_path)
            print(f"Loaded product_rec_model from {product_rec_model_path}")
        else:
            print(f"WARNING: Product recognition model not found at {product_rec_model_path}")

        # Verify models are loaded
        print(f"Model status - shelf_detector: {model_store.shelf_detector is not None}")
        print(f"Model status - product_object_model: {model_store.product_object_model is not None}")
        print(f"Model status - product_rec_model: {model_store.product_rec_model is not None}")
        
        print("Models loaded at startup")
        
    except Exception as e:
        print(f"ERROR loading models: {e}")
        import traceback
        traceback.print_exc()

    return True
