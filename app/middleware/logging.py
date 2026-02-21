"""
Logging middleware for request/response logging.
"""
import logging
import time
import uuid
from typing import Callable, List
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses."""

    def __init__(self, app, logger_name: str = "app", ignore_routes: List[str] = None):
        super().__init__(app)
        self.logger = logging.getLogger(logger_name)
        self.ignore_routes = ignore_routes or []


    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log details."""
        # Check if the route should be ignored
        if request.url.path in self.ignore_routes:
            # Skip logging for ignored routes
            return await call_next(request)

        # Generate request ID if not provided
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Start timing
        start_time = time.time()

        # Log incoming request
        self.logger.info(
            f"Request started - {request.method} {request.url.path} - "
            f"Request ID: {request_id} - Client: {request.client.host if request.client else 'unknown'}"
        )

        # Add request ID to request state
        request.state.request_id = request_id

        try:
            # Process request
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Log response
            self.logger.info(
                f"Request completed - {request.method} {request.url.path} - "
                f"Status: {response.status_code} - Duration: {duration:.3f}s - "
                f"Request ID: {request_id}"
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time

            # Log error
            self.logger.error(
                f"Request failed - {request.method} {request.url.path} - "
                f"Error: {str(e)} - Duration: {duration:.3f}s - "
                f"Request ID: {request_id}"
            )

            # Re-raise the exception
            raise