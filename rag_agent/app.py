import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv
load_dotenv()

from main import open_chat
from utils import MyRag, agentic_rag, sql_agent

app = FastAPI(title="AI CAPTAIN", version="0.1.0")

@app.get("/ask")
def ask(prompt:str):
	res = open_chat(question=prompt)
	return res

@app.get("/basic_rag")
def basic_rag(prompt:str, json_style:bool=True):
	res = MyRag.rag_chat(query=prompt, json_style=json_style)
	return res

@app.get("/agent_rag")
def agent_rag(prompt:str):
	res = agentic_rag(user_input=prompt)
	return res

@app.get("/data_rag")
def data_rag(prompt:str):
	res = sql_agent(user_input=prompt)
	return res


if __name__ == "__main__":
	uvicorn.run(app, port=8000, host="0.0.0.0")