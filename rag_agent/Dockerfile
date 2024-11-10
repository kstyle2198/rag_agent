FROM python:3.12.7-slim

WORKDIR rag_agent

COPY requirements.txt .

COPY . ./

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "./app.py"]
