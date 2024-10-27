import os
import json
import shutil
import uvicorn
from fastapi import FastAPI, File, UploadFile, Response
from dotenv import load_dotenv
load_dotenv()

from main import open_chat
from utils import MyRag, sql_agent, adv_agentic_rag, VectordbManager

from pydantic import BaseModel
from typing import Optional


class AskRequest(BaseModel):
	prompt: str
	json_style: bool = True
	db_path: str = "./db/chroma_db_02"



app = FastAPI(title="AI CAPTAIN", version="0.1.0")


@app.post("/ask")
async def ask(request: AskRequest):
	res = open_chat(question=request.prompt)
	result = {"output": res}
	json_str = json.dumps(result, indent=4, default=str)
	return Response(content=json_str, media_type='application/json')

# @app.get("/ask")
# def ask(prompt:str):
# 	res = open_chat(question=prompt)
# 	return res

# @app.post("/uploadfile/")
# async def upload_file(file: UploadFile = File(...)):
# 	UPLOAD_DIR = "DOCUMENTS/MANUAL"
# 	os.makedirs(UPLOAD_DIR, exist_ok=True)
# 	file_location = f"{UPLOAD_DIR}/{file.filename}"

#     # Save the uploaded file to the specified directory
# 	with open(file_location, "wb") as buffer:
# 		shutil.copyfileobj(file.file, buffer)
# 	return {"filename": file.filename}

@app.post("/basic_rag")
def basic_rag(request: AskRequest):
	res = MyRag.rag_chat(query=request.prompt, json_style=request.json_style)
	json_str = json.dumps(res, indent=4, default=str)
	return Response(content=json_str, media_type='application/json')

# @app.get("/agent_rag")
# def agent_rag(prompt:str):
# 	res = agentic_rag(user_input=prompt)
# 	json_str = json.dumps(res, indent=4, default=str)
# 	return Response(content=json_str, media_type='application/json')

@app.post("/agent_rag")
def adv_agent(request: AskRequest):
	res = adv_agentic_rag(user_input=request.prompt)
	json_str = json.dumps(res, indent=4, default=str)
	return Response(content=json_str, media_type='application/json')

# @app.get("/data_rag")
# def data_rag(prompt:str):
# 	res = sql_agent(user_input=prompt)
# 	return res

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