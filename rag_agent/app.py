import uvicorn
from fastapi import FastAPI

from main import open_chat
from utils import MyRag

app = FastAPI(title="AI CAPTAIN", version="0.1.0")

@app.get("/ask")
def ask(prompt:str):
	res = open_chat(question=prompt)
	return res

@app.get("/rag")
def rag(prompt:str, json_style:bool=True, offline_mode:bool=False):
	res = MyRag.rag_chat(query=prompt, json_style=json_style)
	return res

if __name__ == "__main__":
	uvicorn.run(app, port=8000, host="0.0.0.0")