import os
import argparse
import shutil
import ollama
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

DATA_PATH = "data"

def main():

    print(get_embedding(input_text="blue whale")[:5])

    def get_embedding(input_text, model="llama3.2"):
        embedding = ollama.embeddings(model=model, prompt=input_text)
        return embedding["embedding"]
    
    def load_docs(path=DATA_PATH):
        doc_loader = PyPDFDirectoryLoader(path)
        return doc_loader.load()
    
    def split_documents(docs: list[Document]):
        txt_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_size=80,
            length_function=len,
            is_separator_regex=False,
        )
        return txt_splitter.split_documents(docs)
    
    
    def clear_db():
        # if os.path.exists()
        pass
    


    

if __name__ == "__main__":
    main()