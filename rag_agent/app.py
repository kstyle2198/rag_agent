import os
import json
import shutil
import uvicorn
from pydantic import BaseModel
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from main import open_chat
from utils import MyRag, app_stream, sql_agent, adv_agentic_rag, VectordbManager, AdvancedMiddleware, TimeoutMiddleware, LoggingMiddleware, CustomHeaderMiddleware, ErrorHandlingMiddleware

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


@app.post("/agentic_rag")
def agentic_rag(request: AskRequest):
	res = app_stream(question=request.question, recursion_limit=request.recursion_limit)
	json_str = json.dumps(res, indent=4, default=str)
	return Response(content=json_str, media_type='application/json')


@app.get("/filenames")
def filenames(request: ReadVectorDB):
	res = VectordbManager.get_filename(db_path=request.db_path)
	return res


if __name__ == "__main__":
	uvicorn.run(app, port=8000, host="0.0.0.0")