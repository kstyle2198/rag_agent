# Rag Agent
---

#### 1. Main Features
- Dockerized Ollama and LLM App
- Open Chat, Agentic Rag, Sql Agent
- Conditional API Implementation between Online and Offline 

#### 2. Folder Structure
```
ollama/
├─ pull-ollama.sh
├─ Dockerfile
rag_agent/
├─ db/
│  ├─ chroma_index/
│  ├─ Chinook.db
├─ utils/
│  ├─ __init__.py
│  ├─ agent_openchat.py
│  ├─ rag.py
│  ├─ agent_rag.py
│  ├─ sql_agent.py
├─ main.py
├─ app.py
├─ Dockerfile
compose.yml

```

#### 3. Command

```python
docker compose up --build
```


