# Semantic RAG with LLM

This repository provides a containerized semantic RAG pipeline with LLMs.
The data expected are pdfs of any specific specialised topic that is then embedded and stored in ChromaDB with LangChain. The LLM model used to get context and chat with, is hosted on Ollama. The current config used is summarized below:
- Chunk Splitting Method: LangChain Semantic Text Splitter (all-MiniLM-L6-v2 Sentence Transformer)
- Embedding Model: all-MiniLM-L6-v2
- Vector DB: ChromaDB
- Similarity Search: Cosine Similarity
- LLM Chat: LLama 3.2 (3b)

## 1. Steps to setup RAG LLM (assuming you have Docker):
1. Clone repo
    ```bash
    git clone https://github.com/ahmadabusaiid/semantic-rag-llm.git
    cd semantic-rag-llm
    ```

2. Setup environment and install requirements
    - Windows
    ```bash
    python -m venv venv
    venv/Scripts/activate
    pip install -r requirements.txt
    ```

    - Linux/MacOS
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3. Docker containers setup
    ```bash
    docker-compose up -d --build
    ```
    Please give some time for the models to be downloaded, you can check logs by performing
    ```bash
    docker logs -f ollama
    ```

4. Load and store data (Remember to place your pdfs in the `/data` folder)
    ```bash
    python store_vectors.py
    ```

5. Ask LLM a question with RAG query (as many times as you want)
    ```bash
    python query_rag.py "You can place your question here"
    ```


## 2. Modifying configurations
- YACS config system is used to manage changes for models/embeddings/chunking etc.
- You can easily edit the `config.yml` file to choose your own models etc. Please remember to check documentations on which models are available first, and whether your system can handle it.



---
### Acknowledgements
- Some of the starting points for this project was inspired by tutorials provided by [pixegami](https://github.com/pixegami)