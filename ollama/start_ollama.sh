./bin/ollama serve &

pid=$!

sleep 10

echo "Pulling models"
ollama pull qwen:1.8b

wait $pid