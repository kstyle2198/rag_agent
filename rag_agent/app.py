import os
import json
import shutil
import uvicorn
from fastapi import FastAPI, File, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from dotenv import load_dotenv
load_dotenv()

from main import open_chat
from utils import MyRag, sql_agent, adv_agentic_rag, VectordbManager, AdvancedMiddleware, TimeoutMiddleware, LoggingMiddleware, CustomHeaderMiddleware, ErrorHandlingMiddleware


from pydantic import BaseModel
from typing import Optional


class AskRequest(BaseModel):
	prompt: str
	json_style: bool = True
	db_path: str = "./db/chroma_db_02"


app = FastAPI(title="AI CAPTAIN", version="0.1.0")

### Middlewares ##########################################################
app.add_middleware(TimeoutMiddleware, timeout=90)
app.add_middleware(AdvancedMiddleware)
app.add_middleware(CustomHeaderMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(ErrorHandlingMiddleware)


# origins = [
#     "*"
#     ]

# app.add_middleware(TrustedHostMiddleware, allowed_hosts=origins)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )



@app.post("/ask")
async def ask(request: AskRequest):
	res = open_chat(question=request.prompt)
	result = {"output": res}
	json_str = json.dumps(result, indent=4, default=str)
	return Response(content=json_str, media_type='application/json')

@app.post("/basic_rag")
def basic_rag(request: AskRequest):
	res = MyRag.rag_chat(query=request.prompt, json_style=request.json_style)
	json_str = json.dumps(res, indent=4, default=str)
	return Response(content=json_str, media_type='application/json')

@app.post("/agent_rag")
def adv_agent(request: AskRequest):
	res = adv_agentic_rag(user_input=request.prompt)
	json_str = json.dumps(res, indent=4, default=str)
	return Response(content=json_str, media_type='application/json')


@app.get("/filenames")
def filenames(db_path:str="./db/chroma_db_02"):
	res = VectordbManager.get_filename(db_path=db_path)
	return res


@app.post("/search")
def search(request: AskRequest):
	res = VectordbManager.similarity_search(query=request.prompt, db_path=request.db_path)
	json_str = json.dumps(res, indent=4, default=str)
	return Response(content=json_str, media_type='application/json')


if __name__ == "__main__":
	uvicorn.run(app, port=8000, host="0.0.0.0")