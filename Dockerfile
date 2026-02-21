# ------------------------------
# Base image
# ------------------------------
    FROM public.ecr.aws/docker/library/python:3.11-slim

    # ------------------------------
    # Environment variables
    # ------------------------------
    ENV PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1 \
        PORT=8000
    
    # ------------------------------
    # System dependencies
    # Required for OpenCV / Ultralytics
    # ------------------------------
    RUN apt-get update && apt-get install -y \
        libgl1 \
        libglib2.0-0 \
        curl \
        && rm -rf /var/lib/apt/lists/*
    
    # ------------------------------
    # Working directory
    # ------------------------------
    WORKDIR /app
    
    # ------------------------------
    # Install Python dependencies
    # ------------------------------
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    
    # ------------------------------
    # Copy application code
    # ------------------------------
    COPY app ./app
    
    # ------------------------------
    # Copy models
    # ------------------------------
    COPY models ./models

    # ------------------------------
    # Expose port
    # ------------------------------
    EXPOSE 8000
    
    # ------------------------------
    # Start FastAPI using Uvicorn
    # ------------------------------
    CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
    