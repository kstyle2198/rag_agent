import os
import json
import random
import uvicorn
import asyncio
from pydantic import BaseModel
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from utils import VectordbManager, AdvancedMiddleware, TimeoutMiddleware, LoggingMiddleware, CustomHeaderMiddleware, ErrorHandlingMiddleware
from utils import web_search_stream, sim_search_stream, rag_stream, sql_stream, total_stream

from dotenv import load_dotenv
load_dotenv()

### Input Schema ##########################################
class AskRequest(BaseModel):
	question: str
	recursion_limit: int = 5

class ReadVectorDB(BaseModel):
	db_path: str = "./db/chroma_db_02"

app = FastAPI(title="AI CAPTAIN", version="0.1.0")

### Middlewares ##########################################################
app.add_middleware(TimeoutMiddleware, timeout=90)
app.add_middleware(AdvancedMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(ErrorHandlingMiddleware)
# app.add_middleware(CustomHeaderMiddleware)


origins = [
    "*"
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=origins)

#### API ENDPOINT #####################################################
@app.post("/multi_agent", tags=["AII in One API"])
def multi_agent(request: AskRequest):
	res = total_stream(question=request.question, recursion_limit=request.recursion_limit)
	json_str = json.dumps(res, indent=4, default=str)
	return Response(content=json_str, media_type='application/json')

@app.post("/sql_agent", tags=["Individual API"])
def sql_agent(request: AskRequest):
	res = sql_stream(question=request.question, recursion_limit=request.recursion_limit)
	json_str = json.dumps(res, indent=4, default=str)
	return Response(content=json_str, media_type='application/json')

@app.post("/sim_agent", tags=["Individual API"])
def sim_agent(request: AskRequest):
	res = sim_search_stream(question=request.question, recursion_limit=request.recursion_limit)
	json_str = json.dumps(res, indent=4, default=str)
	return Response(content=json_str, media_type='application/json')

@app.post("/rag_agent", tags=["Individual API"])
def rag_agent(request: AskRequest):
	res = rag_stream(question=request.question, recursion_limit=request.recursion_limit)
	json_str = json.dumps(res, indent=4, default=str)
	return Response(content=json_str, media_type='application/json')

@app.post("/web_agent", tags=["Individual API"])
def web_agent(request: AskRequest):
	res = web_search_stream(question=request.question, recursion_limit=request.recursion_limit)
	json_str = json.dumps(res, indent=4, default=str)
	return Response(content=json_str, media_type='application/json')


@app.get("/filenames", tags=["Document List"])
def filenames(request: ReadVectorDB):
	res = VectordbManager.get_filename(db_path=request.db_path)
	return res

@app.websocket("/ws/random-number")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Generate a random integer between 1 and 10
            random_number = random.randint(1, 10)

            # Send the random number to the client
            await websocket.send_text(f"Check Random number: {random_number}")

            # Check if the number is greater than 5
            if random_number > 5:
                # Send an alert message if the condition is met
                await websocket.send_text(f"Alert: Check Random Number({random_number}) is greater than 5!")

            # Wait for 5 seconds before sending the next number
            await asyncio.sleep(5)

    except WebSocketDisconnect:
        print("Client disconnected")


if __name__ == "__main__":
	uvicorn.run(app, port=8000, host="0.0.0.0")