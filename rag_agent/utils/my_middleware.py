import time
import random
import string
import asyncio
import logging
from typing import Dict
from collections import defaultdict
from fastapi import Request, Response
from fastapi.responses import PlainTextResponse
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
  
class AdvancedMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.rate_limit_records: Dict[str, float] = defaultdict(float)

    async def log_message(self, message: str):
        print(message)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = time.time()
        if current_time - self.rate_limit_records[client_ip] < 0.1:
            return Response(content="Rate limit exceeded", status_code=429)
        
        self.rate_limit_records[client_ip] = current_time
        path = request.url.path
        await self.log_message(f"Request to {path}")

        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        custom_headers = {"X-Process-Time": str(process_time)}
        for header, value in custom_headers.items():
            response.headers.append(header, value)

        random_letters = ''.join(random.choice(string.ascii_letters) for _ in range(10))
        response.headers["X-Request-ID"] = random_letters
        
        await self.log_message(f"Reponse for {path} took {process_time} seconds")

        return response
    
class TimeoutMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, timeout: int):
        super().__init__(app)
        self.timeout = timeout

    async def dispatch(self, request: Request, call_next):
        try:
            return await asyncio.wait_for(call_next(request), timeout=self.timeout)
        except asyncio.TimeoutError:
            return PlainTextResponse(status_code=504, content="Request timed out")
        
logger = logging.getLogger("my_logger")

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        logger.info(f"Request: {request.method} {request.url}")
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        return response
    
class CustomHeaderMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers['Cache-Control'] = 'public, max-age=3600'
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response
    
class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
        except Exception as e:
            response = JSONResponse({"error": str(e)}, status_code=500)
        return response