import argparse
import ollama
import chromadb
from config import load_config
from langchain.prompts import ChatPromptTemplate
from embedding_functions import OllamaEmbeddingFunction

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    """
    Prompts the LLM model to answer user query with RAG based context.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    config = load_config()
    
    query_rag(query_text, config)


def query_rag(query_text, config):
    """
    Performs similarity search with ChromaDB embeddings, for RAG based query.
    """
    embedding_func = OllamaEmbeddingFunction(config=config)
    chroma_client = chromadb.HttpClient(host='localhost', port=8000)
    ollama_client = ollama.Client(host='http://localhost:11434')
    
    db = chroma_client.get_collection(
        name="col_1",
        embedding_function=embedding_func
    )
    results = db.query(
        query_texts=[query_text],
        n_results=5
    )

    context_text = "\n\n--\n\n".join([result for i, result in enumerate(results['documents'][0])])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    response = ollama_client.chat(
        model=config.CHAT.MODEL, 
        messages=[
            {
                'role': 'user',
                'content': prompt,
            }
        ]
    )

    print("-----")
    print(response["message"]["content"])
    print("-----")
    print("Sources:", results["ids"])


if __name__ == "__main__":
    main()
