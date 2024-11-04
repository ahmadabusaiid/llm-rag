import os
import argparse
import shutil
import chromadb
from config import load_config
from langchain.schema.document import Document
from embedding_functions import OllamaEmbeddingFunction
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main():
    """
    Loads the data and stores the vector embeddings into ChromaDB.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    config = load_config()
    chroma_client = chromadb.HttpClient(host='localhost', port=8000)
    embedding_function = OllamaEmbeddingFunction(config=config)

    if args.reset:
        clear_database(path=config.EMBEDDING.STORAGE_PATH)

    documents = load_docs(path=config.DATA.STORAGE_PATH)
    chunks = split_docs(documents)
    # load_to_db(chroma_client, chunks, embedding_function)


def load_docs(path):
    """
    Loads PDF documents from a directory.
    """
    doc_loader = PyPDFDirectoryLoader(path)
    return doc_loader.load()


def split_docs(docs: list[Document]):
    """
    Splits documents into smaller chunks with overlap for context retention.
    """
    txt_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return txt_splitter.split_documents(docs)


def calculate_chunk_ids(chunks):
    """
    Assigns unique IDs to each chunk based on source and page number.
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

    return chunks


def load_to_db(chroma_client, chunks, embedding_func):
    """
    Adds document chunks with embeddings to Chroma.
    """
    db = chroma_client.get_or_create_collection(
        name="col_1",
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"},
    )
    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"Adding new documents: {len(new_chunks)}")
        chunk_dict = {
            "text": [chunk.page_content for chunk in new_chunks], 
            "metadata": [chunk.metadata for chunk in new_chunks], 
            "id": [chunk.metadata["id"] for chunk in new_chunks]
        }

        db.add(
            documents=chunk_dict["text"],
            metadatas=chunk_dict["metadata"],
            ids=chunk_dict["id"],
        )
    else:
        print("No new documents to add.")


def clear_database(path):
    """
    Removes the Chroma database directory to reset it.
    """
    if os.path.exists(path):
        shutil.rmtree(path)
        print("Cleared Chroma database.")



if __name__ == "__main__":
    main()
