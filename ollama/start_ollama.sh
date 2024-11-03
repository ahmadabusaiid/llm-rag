./bin/ollama serve &

pid=$!

sleep 10

EMBEDDING_MODEL=$(yq eval '.EMBEDDING.MODEL' ../config.yml)
CHAT_MODEL=$(yq eval '.CHAT.MODEL' ../config.yml)

echo "Pulling embedding model: $EMBEDDING_MODEL"
ollama pull "$EMBEDDING_MODEL"
echo "Pulling chat model: $CHAT_MODEL"
ollama pull "$CHAT_MODEL"

wait $pid