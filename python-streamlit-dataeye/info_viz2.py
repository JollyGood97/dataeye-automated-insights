from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
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
from langchain.globals import set_debug

key = "YOURKEYHERE"

# Create embeddings
llm = ChatOpenAI(
    openai_api_key=key,
    model="gpt-3.5-turbo",
)
# text-embedding-ada-002
embeddings = OpenAIEmbeddings(openai_api_key=key)


def get_csv_headers_and_dtypes(csv_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Get the headers (column names)
    headers = df.columns.tolist()

    # Get the data types of each column
    dtypes = df.dtypes.apply(lambda x: x.name).to_dict()

    return headers, dtypes


def get_insights():
    # Load and process the CSV data.
    loader = CSVLoader("./csvs/Marks.csv")
    documents = loader.load()
    # print(documents)
    persist_directory_gpt3 = "./chroma_db_gpt3_csv"

    # Set up ChromaDB
    # chroma_db = Chroma.from_documents(
    #     documents, embeddings, persist_directory=persist_directory_gpt3
    # )
    # chroma_db.persist()

    # load
    chroma_db = Chroma(
        persist_directory=persist_directory_gpt3, embedding_function=embeddings
    )

    csv_file_path = "./csvs/Marks.csv"

    # Get headers and data types
    headers, dtypes = get_csv_headers_and_dtypes(csv_file_path)

    # Print headers and data types
    print("Headers:", headers)
    print("Data Types:", dtypes)

    # Pass this information to the prompt
    dataset_attributes = {"headers": headers, "data_types": dtypes}

    # Convert dataset attributes to a string representation for the prompt
    attributes_string = ", ".join(
        [f"{header} ({dtype})" for header, dtype in dtypes.items()]
    )

    # Template for prompting for insights

    prompt_template_insights = PromptTemplate(
        input_variables=["context"],
        # template="This context has only few data available from a larger dataset: {context}, shared with you to answer the following: {question}",
        # template="Given these data attributes of a larger dataset: {context}, {question}",
        template="Given these first few rows of a larger csv dataset: {context}, {question}",
    )

    # Set up the question-answering chain for insights
    qa_chain_insights = RetrievalQA.from_chain_type(
        llm,
        retriever=chroma_db.as_retriever(),
        chain_type_kwargs={"prompt": prompt_template_insights},
    )

    detailed_prompt = """
    "Given a dataset with various attributes related to a specific domain, please analyze the data 
    to suggest two useful analytical insights. 
    For each insight, recommend a suitable type of visualization chart that a data analyst could use to represent the insight visually. 
    Please format your response as a JSON object array, where each object contains keys 'insight' and 'chart', 
    representing the insight and the suggested chart type respectively. 
    Do NOT focus on actual values; the intention is to understand the domain better. 
    For clarity, here is an example of the expected output format:

[
    {
        'insight': 'Trend of X',
        'chart': 'Line chart'
    }, ...
]

Please provide three analytical points along with the suggested visualization charts in the format shown above."

    """
    ## Otherwise LLM focuses on a single student in the retrieved context
    insights_result = qa_chain_insights({"query": detailed_prompt})

    print("full_response", insights_result)

    #     [
    #     {
    #         "insight": "Distribution of marks across different terms",
    #         "chart": "Box plot"
    #     },
    #     {
    #         "insight": "Correlation between marks in Math and marks in other subjects",
    #         "chart": "Scatter plot"
    #     },
    #     {
    #         "insight": "Comparison of average marks between different students",
    #         "chart": "Bar chart"
    #     }
    # ]

    # Assuming `insights_result` contains the insights and chart suggestions from the LLM
    for result in insights_result["result"]:
        insight = result["insight"]
        chart_type = result["chart"]
        code = generate_chart_code(insight, chart_type, headers, dtypes)
        execute_and_display_chart(code)
