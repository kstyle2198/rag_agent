# Rag Agent
---

#### 1. Main Features
- Dockerized Ollama and LLM App
- Open Chat and Rag with VectorStore
- Conditional API Implementation between Online and Offline 

#### 2. Folder Structure
ollama/
├─ Dockerfile
├─ pull-ollama.sh
rag_agent/
├─ db/
│  ├─ chroma_index
├─ utils/
│  ├─ agent_openchat.py
│  ├─ rag.py
│  ├─ __init__.py
├─ app.py
├─ Dockerfile
├─ main.py
compose.yml


