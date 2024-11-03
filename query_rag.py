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
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    config = load_config()
    
    query_rag(query_text, config)


def query_rag(query_text, config):
    print(query_text)
    embedding_func = OllamaEmbeddingFunction(config=config)
    chroma_client = chromadb.HttpClient(host='localhost', port=8000)
    db = chroma_client.get_collection(
        name="col_1",
        embedding_function=embedding_func
    )
    # results = db.similarity_search_with_score(query_text, k=5)
    results = db.query(
        query_texts=[query_text],
        n_results=5
    )

    # context_text = "\n\n--\n\n".join([doc.page.content for doc, _score in results])
    context_text = "\n\n--\n\n".join([result for i, result in enumerate(results['documents'][0])])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)


if __name__ == "__main__":
    main()
