./bin/ollama serve &

pid=$!

sleep 5

echo "Pulling Ollama models"

ollama pull llama3.2:latest

ollama pull bge-m3:latest

wait $pid
