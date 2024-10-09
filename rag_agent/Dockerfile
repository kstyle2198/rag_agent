FROM python:3.10

WORKDIR rag_agent

COPY requirements.txt .

COPY . ./rag_agent

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "./app.py"]
