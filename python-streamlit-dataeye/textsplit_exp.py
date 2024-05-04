from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

import requests
import io
import base64
import re
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
# Create embeddings
llm = ChatOpenAI(
    openai_api_key=key,
    model="gpt-3.5-turbo",
)
from collections import namedtuple

# Define a simple Document structure
Document = namedtuple(
    "Document", ["page_content", "lookup_str", "metadata", "lookup_index"]
)

csv_path = "./csvs/Marks.csv"
persist_directory_gpt3 = "./chroma_db_gpt3_split"


def csv_to_text_and_summary(csv_file_path):
    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Basic statistics for numerical columns
    numerical_summary = df.describe().transpose()

    # Categorical data summary
    categorical_summary = df.describe(include=["object", "category"]).transpose()

    # Combine summaries
    full_summary = f"Numerical Data Summary:\n{numerical_summary}\nCategorical Data Summary:\n{categorical_summary}"

    # Get the file name and prepare text file name
    file_name = csv_file_path.split("/")[-1]
    txt_file_name = file_name.rsplit(".", 1)[0] + ".txt"

    # Prepare the initial text document content
    text_doc = f"Uploaded CSV Dataset Summary:\n- File Name: {file_name}\n- Total Rows: {len(df)}\n- File Path: {csv_file_path}\n"
    text_doc += f"This CSV contains these column headers or attributes in comma separated form: {', '.join(df.columns)}\n"
    text_doc += "Summary generated from the dataset:\n"
    text_doc += (
        full_summary
        + "\nThis CSV File contains the data for each column as an array.\n\n"
    )

    # Adding row data for each column as an array with double newlines for separation
    for column in df.columns:
        column_data = df[column].tolist()
        text_doc += (
            f"{column}: {json.dumps(column_data)}\n\n"  # Double newline added here
        )

    # Save the document text to a file in ./docs/ directory
    docs_dir = "./docs/"
    os.makedirs(docs_dir, exist_ok=True)
    txt_file_path = os.path.join(docs_dir, txt_file_name)
    with open(txt_file_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(text_doc)

    print(f"Document saved to: {txt_file_path}")
    return txt_file_path


def split_text_into_documents(
    text, chunk_size=1000, chunk_overlap=200, docs_dir="./split_docs/"
):

    text_splitter = CharacterTextSplitter(
        separator="\n\n",  # Assuming each section is double-newline separated
        length_function=len,
        is_separator_regex=False,
    )

    documents = text_splitter.create_documents([text])
    print(documents)
    # print(len(documents))


def split_text_to_documents(file_path):
    # Read the text from the file
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Split the text based on double newlines
    sections = text.split("\n\n")

    # Initialize an empty list to store Document objects
    documents = []

    # Loop through each section and create a Document object
    for index, section in enumerate(sections):
        # Assuming the metadata and lookup_str are not directly available from this content
        # Adjust as necessary for your use case
        document = Document(
            page_content=section,
            lookup_str="",
            metadata={"source": file_path},
            lookup_index=index,
        )
        documents.append(document)

    return documents


# Usage example
if __name__ == "__main__":
    csv_path = (
        "./csvs/Marks.csv"  # This should be the path to your text file, not the CSV
    )
    txt_file_path = csv_to_text_and_summary(
        csv_path
    )  # Ensure this creates the desired txt file
    documents = split_text_to_documents(txt_file_path)  # Adjust the path if necessary
    for doc in documents:
        print(doc)


def test():
    # Assume csv_path is defined and valid
    csv_path = "./csvs/Marks.csv"  # Update this path
    txt_file_path = csv_to_text_and_summary(csv_path)

    # Assuming the text document is not too large to be loaded into memory at once
    with open(txt_file_path, "r", encoding="utf-8") as txt_file:
        text_doc = txt_file.read()

    documents = split_text_to_documents(txt_file_path)

    print(documents)


if __name__ == "__main__":
    test()
