from fastapi import Request, Response, Query, Body, FastAPI, UploadFile, File, Form
import uuid
import os
import logging
from ultralytics import YOLO
from app.use_case.fetch_shelf_details import FetchShelfDetails
from app.use_case.product_as_object_detection import FetchProductAsObjectDetails
from app.use_case.calculate_empty_shelf_percentage import EmptyShelfPercentageDetails
from app.use_case.product_recognition import ProductDetails

from app import model_store
from app.middleware.logging import LoggingMiddleware
from fastapi.responses import StreamingResponse
from typing import List
from joblib import load

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from app.backend.chat_service import run_chat_stream
from langgraph.types import Command

from app.backend.db import init_db

logging.basicConfig(
    level=logging.INFO,  # or DEBUG
    format="%(name)s - %(levelname)s - %(message)s"
)

app = FastAPI(title="Just invoke and get...!")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.add_middleware(LoggingMiddleware, ignore_routes=["/healthz"])

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.on_event("startup")
def load_model():
    
    shelf_detector_v14_path = "models/shelf_detector_v14/weights/best.pt"
    model_store.shelf_detector = YOLO(shelf_detector_v14_path)

    product_model_path = "models/product_recognition_yolo11/weights/best.pt"
    model_store.product_object_model = YOLO(product_model_path)

    product_rec_model_path = "models/rpc_yolov11_4dh3/weights/best.pt"
    model_store.product_rec_model = YOLO(product_rec_model_path)

    print("models loaded at startup")
    init_db()

@app.post("/shelves_detection")
async def run_shelves_detection(
    image: UploadFile = File(...),
    conf: float = 0.5,
):
    # Save uploaded file
    image_id = str(uuid.uuid4())
    file_path = f"{UPLOAD_DIR}/{image_id}_{image.filename}"

    with open(file_path, "wb") as f:
        f.write(await image.read())

    request_body = {"file_path": file_path}

    response = FetchShelfDetails().execute(request_body)
    print(response)
    return response

@app.post("/product_count_detection")
async def run_product_count_detection(
    image: UploadFile = File(...),
    conf: float = 0.5,
):
    # Save uploaded file
    image_id = str(uuid.uuid4())
    file_path = f"{UPLOAD_DIR}/{image_id}_{image.filename}"

    with open(file_path, "wb") as f:
        f.write(await image.read())

    request_body = {"file_path": file_path}

    response = FetchProductAsObjectDetails().execute(request_body)
    print(response)
    return response

@app.post("/calculate_empty_percentage")
async def run_calculate_empty_percentage(
    image: UploadFile = File(...),
):
    # Save uploaded file
    image_id = str(uuid.uuid4())
    file_path = f"{UPLOAD_DIR}/{image_id}_{image.filename}"

    with open(file_path, "wb") as f:
        f.write(await image.read())

    request_body = {"file_path": file_path}

    response = EmptyShelfPercentageDetails().execute(request_body)
    print(response)
    return response

@app.post("/product_recognition")
async def run_product_recognition(
    request: Request,
    images: list[UploadFile] = File(...)
):
    file_paths = []
    request_id = request.state.request_id
    
    for img in images:
        image_id = str(uuid.uuid4())
        file_path = f"{UPLOAD_DIR}/{image_id}_{img.filename}"

        with open(file_path, "wb") as f:
            f.write(await img.read())

        file_paths.append(file_path)

    request_body = {
        "file_paths_list": file_paths,
        "request_id": request_id
    }

    response = ProductDetails().execute(request_body)
    print(response)
    return response

@app.post("/chat/stream")
async def chat_stream(
    user_input: str = Form(...),
    thread_id: str = Form(...),
    images: List[UploadFile] = File(None),
):
    def ai_stream(thread_id: str, user_input: str, image_paths: list[str]):
        final_text_chunks = []
        status_holder = {"box": None}

        for msg in run_chat_stream(
            thread_id=thread_id,
            user_input=user_input,
            images_list=image_paths,
        ):
            if "__interrupt__" in msg:
                interrupt_obj = msg["__interrupt__"][0]
                question = interrupt_obj.value.get("question", "yes")
                user_input = "yes"
                Command(
                    resume=True,
                    update={
                        "messages": [HumanMessage(content=user_input)]
                    }
                )
                print("starting interrupt to take user response")
                yield "starting interrupt to take user response"
                
            # -------- Tool handling --------
            if isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", "tool")
                # Log tool usage (FastAPI has no UI status)
                print(f"Using tool: {tool_name}")

            # -------- Assistant tokens --------
            if isinstance(msg, AIMessage):
                final_text_chunks.append(msg.content)
                yield msg.content

    image_paths = []

    # Save uploaded images
    if images:
        for image in images:
            image_id = str(uuid.uuid4())
            file_path = f"{UPLOAD_DIR}/{image_id}_{image.filename}"

            with open(file_path, "wb") as f:
                f.write(await image.read())

            image_paths.append(file_path)

    return StreamingResponse(
        ai_stream(thread_id, user_input, image_paths),
        media_type="text/plain"
    )