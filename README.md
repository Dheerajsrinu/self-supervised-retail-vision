# IIITH Chatbot Architecture Documentation

A retail-focused AI chatbot for product recognition, shelf analysis, and automated checkout using computer vision and LLM-powered conversational AI.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture Diagram](#architecture-diagram)
3. [System Components](#system-components)
4. [Data Flow](#data-flow)
5. [Database Schema](#database-schema)
6. [ML Models](#ml-models)
7. [Tools & Capabilities](#tools--capabilities)
8. [Technology Stack](#technology-stack)
9. [Directory Structure](#directory-structure)

---

## Overview

The IIITH Retail Chatbot is an AI-powered retail checkout assistant designed for both **Customers** and **Store Managers**. It combines computer vision, natural language processing, and intelligent workflow automation to streamline retail operations.

### What Can Users Do?

#### For Customers

| Feature | Description |
|---------|-------------|
| **Image-Based Shopping** | Upload photos of retail shelves to automatically detect and identify products, then place orders directly from detected items |
| **Direct Ordering** | Place orders through natural conversation (e.g., "Order 2 chocolates and 1 milk") without uploading images |
| **Product Recognition** | Get detailed information about products visible in uploaded shelf images |
| **Shelf Analysis** | View shelf occupancy, count products, and see empty space percentages |
| **Nutrition Information** | Look up nutritional details for recognized products |
| **Health Warnings** | Receive automatic health advisories when ordering excessive quantities of certain products (candy, alcohol, etc.) |
| **Age Verification** | Age-restricted items like alcohol are validated against user profile before checkout |
| **Order History** | View past orders with details and status on the Orders Dashboard |

#### For Store Managers

| Feature | Description |
|---------|-------------|
| **All Customer Features** | Full access to product recognition, shelf analysis, and ordering capabilities |
| **Restocking Orders** | Upload shelf images to automatically calculate restock quantities based on inventory thresholds |
| **Manager Dashboard** | Comprehensive admin panel with multiple tabs for business insights |
| **User Management** | View all registered users with search and role filters |
| **Order Oversight** | Browse all orders across the platform with date range and user filters |
| **Telemetry Dashboard** | Monitor system performance including model response times, transaction durations, and usage statistics |
| **Model Verification** | Audit ML model accuracy by comparing inference outputs against LLM interpretations with agreement rate metrics |

### Core Capabilities

| Capability | Description |
|------------|-------------|
| **Shelf Detection** | Automatically identify and count shelf rows/levels in retail images |
| **Product Detection** | Detect and count individual product items on shelves |
| **Empty Space Calculation** | Calculate the percentage of empty shelf space for inventory management |
| **Product Recognition** | Identify specific product names and categories using trained ML models |
| **Conversational AI** | Natural language interface for all operations - just chat like you would with a store assistant |
| **Human-in-the-Loop** | Review and confirm orders before they are placed |

### User Roles

| Role | Access Level | Special Features |
|------|--------------|------------------|
| **Customer** | Standard | Health warnings, age restrictions, personal order history |
| **Store Manager** | Full | Manager Dashboard, all orders view, telemetry, no health/age restrictions on restocking |

### Technology Highlights

- **Computer Vision**: YOLO-based object detection for shelf and product analysis
- **LLM Integration**: GPT-powered natural language understanding and response generation
- **LangGraph Flow**: State machine-based conversation orchestration with tool calling
- **Guardrail Validation**: Request filtering to ensure retail-focused conversations
- **Role-Based Access Control**: Different experiences for Customers vs Store Managers
- **Telemetry & Monitoring**: Real-time performance tracking and model verification

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                  USER INTERFACE                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│    ┌──────────────────────┐         ┌──────────────────────────┐                │
│    │    UI Chat           │ ──────► │     Streamlit UI         │                │
│    │  Text + Image Upload │         │    (chatbot.py)          │                │
│    └──────────────────────┘         └───────────┬──────────────┘                │
│                                                  │                               │
└──────────────────────────────────────────────────┼──────────────────────────────┘
                                                   │
                                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               BACKEND LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│    ┌────────────────────────────────────────────────────────────────────┐       │
│    │                      Chat Service                                   │       │
│    │                   (chat_service.py)                                 │       │
│    │                                                                     │       │
│    │   • Handles user input + images                                     │       │
│    │   • Routes to Validator or Chat Graph                               │       │
│    │   • Manages resume/interrupt flow                                   │       │
│    └───────────────────────────┬────────────────────────────────────────┘       │
│                                │                                                 │
│                    ┌───────────┴───────────┐                                    │
│                    ▼                       ▼                                    │
│    ┌─────────────────────────┐   ┌────────────────────────────┐                 │
│    │   Guardrail Validator   │   │        PostgreSQL          │                 │
│    │  (gaurdrail_flow.py)    │   │        Database            │                 │
│    │                         │   │                            │                 │
│    │  • Validates requests   │   │  ┌────────────────────┐   │                 │
│    │  • Filters off-topic    │   │  │   chat_threads     │   │                 │
│    │  • Structured output    │   │  │   (id, thread_id,  │   │                 │
│    └───────────┬─────────────┘   │  │    user_id)        │   │                 │
│                │                  │  └────────────────────┘   │                 │
│                ▼                  │  ┌────────────────────┐   │                 │
│    ┌─────────────────────────┐   │  │   chat_messages    │   │                 │
│    │     LangGraph Flow      │   │  │   (id, thread_id,  │   │                 │
│    │  (langgraph_flow.py)    │   │  │    role, content)  │   │                 │
│    │                         │   │  └────────────────────┘   │                 │
│    │  START                  │   │  ┌────────────────────┐   │                 │
│    │    │                    │   │  │      users         │   │                 │
│    │    ▼                    │   │  │   (user_id, email, │   │                 │
│    │  Preprocess             │   │  │    password, etc.) │   │                 │
│    │    │                    │   │  └────────────────────┘   │                 │
│    │    ▼                    │   │  ┌────────────────────┐   │                 │
│    │  Chatbot Node           │   │  │      orders        │   │                 │
│    │    │                    │   │  │   (id, user_id,    │   │                 │
│    │    ├──► Tools           │   │  │    products)       │   │                 │
│    │    │      │             │   │  └────────────────────┘   │                 │
│    │    │      ▼             │   └────────────────────────────┘                 │
│    │    │   Tools Done       │                                                  │
│    │    │      │             │                                                  │
│    │    ◄──────┘             │                                                  │
│    │    │                    │                                                  │
│    │    ▼                    │                                                  │
│    │  Review Interrupt       │                                                  │
│    │    │                    │                                                  │
│    │    ▼                    │                                                  │
│    │  Review Decision        │                                                  │
│    │    ├──► Approved ──► Create Order ──► END                                  │
│    │    └──► Rejected ──► END                                                   │
│    └─────────────────────────┘                                                  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                                   │
                                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            LLM & TOOLS LAYER                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│    ┌─────────────────────────┐         ┌────────────────────────────────────┐   │
│    │    GPT-5-nano / LLM     │ ◄─────► │            Tools                   │   │
│    │   Model.bind(tools)     │         │                                    │   │
│    │                         │         │  T1: detect_shelves                │   │
│    │   • Temperature: 0.2    │         │      - Detects shelf structures    │   │
│    │   • Tool calling        │         │                                    │   │
│    │   • Health warnings     │         │  T2: detect_products               │   │
│    └─────────────────────────┘         │      - Counts product objects      │   │
│                                         │                                    │   │
│                                         │  T3: calculate_empty_shelf_%      │   │
│                                         │      - Computes empty space        │   │
│                                         │                                    │   │
│                                         │  T4: recognize_products            │   │
│                                         │      - Identifies product names    │   │
│                                         │                                    │   │
│                                         │  T5: calculator                    │   │
│                                         │      - Basic math operations       │   │
│                                         │                                    │   │
│                                         │  T6: browser_search                │   │
│                                         │      - DuckDuckGo web search       │   │
│                                         └─────────────────┬──────────────────┘   │
│                                                           │                      │
└───────────────────────────────────────────────────────────┼──────────────────────┘
                                                            │
                                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           INFERENCE LAYER                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│    ┌────────────────────────────────────────────────────────────────────┐       │
│    │                    Model Loader (Startup)                           │       │
│    │                     (model_loader.py)                               │       │
│    │                                                                     │       │
│    │   Loads YOLO models once using @st.cache_resource                   │       │
│    └───────────────────────────┬────────────────────────────────────────┘       │
│                                │                                                 │
│                                ▼                                                 │
│    ┌────────────────────────────────────────────────────────────────────┐       │
│    │                     Inference Modules                               │       │
│    │                                                                     │       │
│    │   ┌───────────────────────┐  ┌───────────────────────────────┐     │       │
│    │   │ shelf_object_pred_inf │  │    cropped_products_inf       │     │       │
│    │   │   - Shelf detection   │  │   - Product cropping          │     │       │
│    │   │   - YOLO v14          │  │   - Bounding box extraction   │     │       │
│    │   └───────────────────────┘  └───────────────────────────────┘     │       │
│    │                                                                     │       │
│    │   ┌───────────────────────┐  ┌───────────────────────────────┐     │       │
│    │   │  prd_as_obj_det_inf   │  │      simclr_mlp_inf           │     │       │
│    │   │   - Product detection │  │   - Product recognition       │     │       │
│    │   │   - YOLO v11          │  │   - SimCLR + MLP classifier   │     │       │
│    │   └───────────────────────┘  └───────────────────────────────┘     │       │
│    └────────────────────────────────────────────────────────────────────┘       │
│                                │                                                 │
│                                ▼                                                 │
│    ┌────────────────────────────────────────────────────────────────────┐       │
│    │                     Output Generation                               │       │
│    │                                                                     │       │
│    │   • Save annotated images                                           │       │
│    │   • Return detection results                                        │       │
│    │   • Aggregate product counts                                        │       │
│    └────────────────────────────────────────────────────────────────────┘       │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## System Components

### 1. **UI Layer** (`chatbot.py`)

The Streamlit-based user interface provides:

- **Chat Interface**: Text input with streaming responses
- **Image Upload**: Support for multiple retail shelf images (PNG, JPG, JPEG)
- **Thread Management**: Create and switch between conversation threads
- **User Authentication**: Login/logout with session management
- **Interrupt Handling**: Human-in-the-loop confirmation for order creation

### 2. **Backend Layer**

#### Chat Service (`app/backend/chat_service.py`)
- Entry point for all chat interactions
- Handles both new messages and resume flows
- Routes to validator graph first, then delegates to chat graph

#### Guardrail Flow (`app/backend/gaurdrail_flow.py`)
- Validates incoming requests against allowed topics
- Uses structured output to determine if request is retail-related
- Allowed topics: Retail images, shelf count, product count, product names, nutrition

#### LangGraph Flow (`app/backend/langgraph_flow.py`)
- State machine orchestration for conversation flow
- Nodes:
  - `preprocess`: Resets state for new image requests
  - `chatbot`: Main LLM interaction node
  - `tools`: Tool execution node
  - `tools_done`: Post-tool processing with health warnings
  - `review_interrupt`: Human approval checkpoint
  - `review_decision`: Routes based on user yes/no
  - `approved/rejected`: Final action nodes
  - `create_retail_order`: Order creation in database

### 3. **LLM Layer**

- **Model**: GPT-5-nano (configurable)
- **Temperature**: 0.2 (low for consistency)
- **Tool Binding**: Dynamic tool registration via LangChain
- **Health Warnings**: Category-based thresholds for product quantities

### 4. **Inference Layer**

#### Model Loader (`app/backend/model_loader.py`)
Loads ML models at startup using Streamlit's caching:
- `shelf_detector_v14`: Shelf detection YOLO model
- `product_object_model`: Product detection YOLO model
- `product_rec_model`: Product recognition YOLO model

#### Inference Modules (`app/inference/`)
- `shelf_object_prediction_mdl_inf.py`: Shelf bounding box detection
- `cropped_products_inf.py`: Crop individual products from shelves
- `prd_as_obj_det_inf.py`: Product object detection
- `simclr_mlp_inf.py`: SimCLR-based product classification

---

## Data Flow

```
1. User Input (Text + Images)
         │
         ▼
2. Streamlit UI (chatbot.py)
         │
         ▼
3. Chat Service (run_chat_stream)
         │
         ├──► [Resume Flow] ──► Chat Graph directly
         │
         ▼
4. Guardrail Validator
         │
         ├──► [Rejected] ──► Return error message
         │
         ▼
5. Chat Graph (LangGraph)
         │
         ├──► Preprocess ──► Chatbot Node
         │                        │
         │                        ├──► [Tool Call] ──► Tools ──► Tools Done ──► Loop back
         │                        │
         │                        ├──► [Review Needed] ──► Interrupt ──► User Decision
         │                        │                                           │
         │                        │                         ┌─────────────────┤
         │                        │                         ▼                 ▼
         │                        │                    [Approved]        [Rejected]
         │                        │                         │                 │
         │                        │                    Create Order          END
         │                        │                         │
         │                        ▼                         ▼
         │                      END                        END
         │
         ▼
6. Response Stream to UI
```

---

## Database Schema

### Tables

```sql
-- User conversations
CREATE TABLE chat_threads (
    thread_id UUID PRIMARY KEY,
    title TEXT,
    user_id TEXT,
    is_waiting_for_review BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Individual messages
CREATE TABLE chat_messages (
    id SERIAL PRIMARY KEY,
    thread_id UUID NOT NULL,
    role TEXT CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User accounts
CREATE TABLE users (
    user_id UUID PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    address TEXT NOT NULL,
    pincode INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Retail orders
CREATE TABLE orders (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    products JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_orders_thread
        FOREIGN KEY (user_id)
        REFERENCES chat_threads(thread_id)
        ON DELETE CASCADE
);
```

### Checkpointing

LangGraph state is persisted using `PostgresSaver` for:
- Conversation continuity across sessions
- Interrupt/resume capability
- State recovery

---

## ML Models

| Model | Path | Purpose |
|-------|------|---------|
| Shelf Detector v14 | `models/shelf_detector_v14/weights/best.pt` | Detect shelf rows in retail images |
| Product Recognition YOLO11 | `models/product_recognition_yolo11/weights/best.pt` | Detect product objects |
| RPC YOLO v11 | `models/rpc_yolov11_4dh3/weights/best.pt` | Retail product category recognition |
| SimCLR MLP | `models/simclr_checkpoints_finetuned/simclr_epoch_100.pth` | Product embedding classification |
| MLP Classifier | `models/mlp_4_eval_out/mlp4_classifier.pt` | Final product classification |

---

## Tools & Capabilities

### Available Tools

| Tool | Function | Description |
|------|----------|-------------|
| `detect_shelves` | Shelf Detection | Identify and count shelf structures in images |
| `detect_products` | Product Detection | Detect product objects and return count |
| `calculate_empty_shelf_percentage` | Empty Space Analysis | Calculate percentage of empty shelf space |
| `recognize_products` | Product Recognition | Identify product names/categories across images |
| `calculator` | Math Operations | Basic arithmetic (add, sub, mul, div) |
| `browser_search` | Web Search | DuckDuckGo search for additional information |

### Health Warning Thresholds

Automatic health warnings are generated when detected quantities exceed:

```python
WARNING_THRESHOLDS = {
    "Alcohol": 0,
    "Candy": 2,
    "Chocolate": 2,
    "Dessert": 3,
    "Drink": 4,
    "Instant Noodles": 1,
    "Milk": 3,
    "Puffed Food": 1,
    # ... and more
}
```

---

## Technology Stack

### Core Framework
- **Python 3.x**
- **Streamlit**: Web UI framework
- **LangChain / LangGraph**: LLM orchestration and state management

### AI/ML
- **OpenAI GPT**: Language model
- **Ultralytics YOLO**: Object detection
- **PyTorch**: Deep learning framework
- **SimCLR**: Contrastive learning for embeddings

### Database
- **PostgreSQL**: Primary database
- **psycopg**: PostgreSQL adapter
- **LangGraph Checkpoint Postgres**: State persistence

### Additional
- **python-dotenv**: Environment configuration
- **tiktoken**: Token counting
- **DuckDuckGo Search**: Web search integration

---

## Directory Structure

```
iiith-chatbot/
├── app/
│   ├── backend/
│   │   ├── chat_service.py      # Main chat orchestration
│   │   ├── db.py                # Database operations
│   │   ├── gaurdrail_flow.py    # Request validation
│   │   ├── langgraph_flow.py    # Conversation state machine
│   │   ├── model_loader.py      # ML model initialization
│   │   ├── security.py          # Password hashing
│   │   ├── state.py             # State type definitions
│   │   └── tools.py             # LangChain tool definitions
│   ├── inference/
│   │   ├── cropped_products_inf.py
│   │   ├── prd_as_obj_det_inf.py
│   │   ├── shelf_object_prediction_mdl_inf.py
│   │   └── simclr_mlp_inf.py
│   ├── use_case/
│   │   ├── calculate_empty_shelf_percentage.py
│   │   ├── fetch_shelf_details.py
│   │   ├── product_as_object_detection.py
│   │   └── product_recognition.py
│   ├── config.py                # Environment configuration
│   ├── helper.py                # Utility functions
│   └── model_store.py           # Global model references
├── models/                      # Trained ML model weights
│   ├── shelf_detector_v14/
│   ├── product_recognition_yolo11/
│   ├── rpc_yolov11_4dh3/
│   ├── simclr_checkpoints_finetuned/
│   └── mlp_4_eval_out/
├── views/
│   └── auth_view.py             # Authentication UI
├── pages/
│   └── orders_dashboard.py      # Orders page
├── chatbot.py                   # Main Streamlit entry point
├── requirements.txt             # Python dependencies
└── Dockerfile                   # Container configuration
```

---

## Environment Variables

Required environment variables (via `.env`):

```bash
# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=chatbot_db
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password

# OpenAI
OPENAI_API_KEY=your_openai_api_key
```

---

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run chatbot.py
```

---

## Key Features

1. **Multi-modal Input**: Text + Image support for retail analysis
2. **Guardrails**: LLM-based request validation for topic relevance
3. **Tool Calling**: Automatic tool selection based on user intent
4. **Human-in-the-Loop**: Review workflow before order creation
5. **Health Awareness**: Automatic warnings for excessive product quantities
6. **Persistent State**: Conversation history and checkpointing
7. **Streaming Responses**: Real-time token streaming for better UX

---

## License

© IIITH Capstone Project 2025
