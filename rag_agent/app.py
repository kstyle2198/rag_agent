import uvicorn
import requests
from fastapi import FastAPI, Response


app = FastAPI()

@app.get("/")
def main():
	return {'message':'hello'}

@app.get("/ask")
def ask(prompt:str):
	res = requests.post("http://ollama:11434/api/generate", json={"prompt": prompt,"stream": False, "model": "llama3.2"})
	return Response(content=res.text, media_type="application/json")


if __name__ == "__main__":
	uvicorn.run(app, port=8000, host="0.0.0.0")