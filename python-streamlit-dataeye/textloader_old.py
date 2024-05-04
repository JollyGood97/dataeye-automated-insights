from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders import TextLoader

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

langchain.debug = True

langchain.verbose = True


def csv_to_text(csv_file_path):
    # Extract the file name from the file path
    file_name = csv_file_path.split("/")[-1]
    txt_file_name = file_name.rsplit(".", 1)[0] + ".txt"

    # Initialize a dictionary to hold column data
    column_data = {}

    # Open the CSV file and read data
    with open(csv_file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        headers = reader.fieldnames
        for header in headers:
            column_data[header] = []

        row_count = 0
        for row in reader:
            for header in headers:
                column_data[header].append(row[header])
            row_count += 1

    # Structure the document text
    text_doc = f"Uploaded CSV Dataset Summary:\n- File Name: {file_name}\n- Total Rows: {row_count}\n- File Path: {csv_file_path}\n"
    text_doc += f"This CSV contains these column headers or attributes in comma separated form: {', '.join(headers)}\n"
    text_doc += "This CSV contains the data for each column as an array:\n"
    for header in headers:
        text_doc += f"{header}:{json.dumps(column_data[header])}\n"

        # Save the document text to a file in ./docs/ directory
    docs_dir = "./docs/"
    os.makedirs(docs_dir, exist_ok=True)  # Ensure the directory exists
    txt_file_path = os.path.join(docs_dir, txt_file_name)
    with open(txt_file_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(text_doc)

    print(f"Document saved to: {txt_file_path}")
    return txt_file_path


key = "YOURKEYHERE"
# Create embeddings
llm = ChatOpenAI(
    openai_api_key=key,
    model="gpt-3.5-turbo",
)

csv_path = "./csvs/Marks.csv"
persist_directory_gpt3 = "./chroma_db_gpt3_textloader"


def generate_matplotlib_code(insight, chart_type):
    prompt_template_insights = PromptTemplate(
        input_variables=["context"],
        # template="This context has only few data available from a larger dataset: {context}, shared with you to answer the following: {question}",
        # template="Given these data attributes of a larger dataset: {context}, {question}",
        template="  Given the data required for the chart as follows: {context}, {question}",
    )

    prompt = f"""
        
        Generate the Python Matplotlib code necessary to visualize the insight "{insight}" using a {chart_type}. 
        """

    # Set up the question-answering chain for insights
    qa_chain_insights = RetrievalQA.from_chain_type(
        llm,
        retriever=chroma_db.as_retriever(search_type="mmr"),
        chain_type_kwargs={"prompt": prompt_template_insights},
        # kwargs={"k": 5},
    )

    charts_result = qa_chain_insights({"query": prompt})
    print("charts_result", charts_result)


if __name__ == "__main__":
    txt_file_path = csv_to_text(csv_path)
    loader = TextLoader(txt_file_path)
    documents = loader.load()
    print(documents)
    # text-embedding-ada-002
    embeddings = OpenAIEmbeddings(openai_api_key=key)

    # Set up ChromaDB first time only
    # chroma_db = Chroma.from_documents(
    #     documents, embeddings, persist_directory=persist_directory_gpt3
    # )
    # chroma_db.persist()

    # load if already persisted
    chroma_db = Chroma(
        persist_directory=persist_directory_gpt3, embedding_function=embeddings
    )

    # Template for prompting for insights

    prompt_template_insights = PromptTemplate(
        input_variables=["context"],
        # template="This context has only few data available from a larger dataset: {context}, shared with you to answer the following: {question}",
        # template="Given these data attributes of a larger dataset: {context}, {question}",
        template=" {context}, {question}",
    )

    # Set up the question-answering chain for insights
    qa_chain_insights = RetrievalQA.from_chain_type(
        llm,
        retriever=chroma_db.as_retriever(search_type="mmr"),
        chain_type_kwargs={"prompt": prompt_template_insights},
        # kwargs={"k": 5},
    )

    detailed_prompt = """
    "Please analyze the data 
    to suggest three useful analytical insights. 
    For each insight, recommend a suitable type of visualization chart that a data analyst could use to represent the insight visually. 
    Please format your response as a JSON object array, where each object contains keys 'insight' and 'chart', 
    representing the insight and the suggested chart type respectively. 
    Do NOT focus on actual values; the intention is to understand the domain better. 
    For clarity, here is an example of the expected output format:

[
    {
        'insight': 'Trend of X',
        'chart': 'Line chart',
    }, ...
]

Please provide three analytical points along with the suggested visualization charts in the format shown above."

    """
    ## Otherwise LLM focuses on a single student in the retrieved context
    insights_result = qa_chain_insights({"query": detailed_prompt})

    # Deserialize the JSON string back into a Python list of dictionaries
    result = json.loads(insights_result["result"])
    print("Deserialized result:", result)
    print(type(result))

    # Iterate over each insight and generate Matplotlib code
    for insight_result in result:
        insight = insight_result["insight"]
        chart_type = insight_result["chart"]
        print("insight", insight)
        print("chart_type", chart_type)
