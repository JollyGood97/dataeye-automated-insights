from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders import TextLoader

import matplotlib.pyplot as plt
from flask_cors import CORS
import openai
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import pandas as pd
import langchain
import csv
import json
import os
from PIL import Image
import numpy as np

langchain.debug = True

langchain.verbose = True


key = "YOURKEYHERE"
csv_path = "./csvs/Marks.csv"
# persist_directory_gpt3 = "./chroma_gpt3_marks_csvloader"

# persist_directory_gpt3 = "./chroma_db_gpt3_csv"
persist_directory_gpt3 = "./chroma_db_gpt3_textloader"
# gpt-4-turbo-preview
# Create embeddings
llm = ChatOpenAI(
    openai_api_key=key,
    # model="gpt-3.5-turbo",
    model="gpt-4-turbo-preview",
)


def interact(prompt):
    response = llm.invoke(prompt)
    print(response)


def vectordbcsv(prompt):
    embeddings = OpenAIEmbeddings(openai_api_key=key)
    chroma_db = Chroma(
        persist_directory=persist_directory_gpt3, embedding_function=embeddings
    )
    qa_chain_insights = RetrievalQA.from_chain_type(
        llm,
        retriever=chroma_db.as_retriever(search_type="mmr"),
    )

    result = qa_chain_insights({"query": prompt})
    print(result)


def load_csv(prompt):
    """Load the CSV file and return a list of Documents."""
    # loader = CSVLoader(file_path=csv_path)
    # documents = loader.load()
    embeddings = OpenAIEmbeddings(openai_api_key=key)
    # Set up ChromaDB
    # chroma_db = Chroma.from_documents(
    #     documents, embeddings, persist_directory=persist_directory_gpt3
    # )
    # chroma_db.persist()

    # load
    chroma_db = Chroma(
        persist_directory=persist_directory_gpt3, embedding_function=embeddings
    )

    qa_chain_insights = RetrievalQA.from_chain_type(
        llm,
        retriever=chroma_db.as_retriever(search_type="mmr"),
    )

    result = qa_chain_insights({"query": prompt})
    print(result)


if __name__ == "__main__":
    # prompt = "Who scored the highest marks for Math in Term 1?"
    prompt = "How much marks did Jack score for Math in Term 1?"

    # vectordb(prompt)
    load_csv(prompt)
