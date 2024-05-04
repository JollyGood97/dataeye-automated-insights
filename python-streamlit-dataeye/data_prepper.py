from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import CSVLoader
from typing import List
from langchain.globals import set_debug
import streamlit as st

set_debug(True)


class DataPrepper:
    def __init__(
        self,
        openai_api_key,
        embeddings_model="text-embedding-ada-002",
        persist_directory="./chroma_db_gpt3_5_turbo",
    ):
        self.llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key, model=embeddings_model
        )
        self.persist_directory = persist_directory
        self.chroma_db = None

    def load_csv(self, csv_file_path: str):
        """Load the CSV file and return a list of Documents."""
        loader = CSVLoader(file_path=csv_file_path)
        return loader.load()

    def store_in_vector_db(self, documents):
        """Convert Document page_content to embeddings and store them in ChromaDB."""
        if self.chroma_db is None:
            # Extract page_content from each Document to create a list of strings
            document_texts = [doc.page_content for doc in documents]

            # Embed the documents using embed_documents method
            document_embeddings = self.embeddings.embed_documents(document_texts)

            # Assuming Chroma.from_documents requires a list of embedded documents
            self.chroma_db = Chroma.from_documents(
                documents=document_embeddings,  # Use the embedded documents here
                persist_directory=self.persist_directory,
            )

            self.chroma_db.persist()
        # else:
        #     # If ChromaDB already exists, just add new documents
        #     for doc in documents:
        #         content = doc.page_content
        #         embedding = self.embeddings.embed_text(content)
        #         self.chroma_db.add_document(document=embedding)

    def preprocess_and_store(self, csv_file_path: str):
        """Complete workflow from loading CSV to storing embeddings in ChromaDB."""
        documents = self.load_csv(csv_file_path)
        self.store_in_vector_db(documents)


if __name__ == "__main__":
    # Example setup - replace with your actual values
    openai_api_key = "YOURKEYHERE"
    csv_file_path = "./csvs/Marks.csv"

    # Initialize your DataPrepper
    data_prepper = DataPrepper(
        openai_api_key=openai_api_key,
        embeddings_model="text-embedding-ada-002",
        persist_directory="./chroma_db_gpt3_5_turbo",
    )

    # Load CSV and process
    documents = data_prepper.load_csv(csv_file_path)
    print("Loaded Documents:", documents)

    # Store in vector DB (debug this method as needed)
    data_prepper.store_in_vector_db(documents)
