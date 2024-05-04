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

# llm = Ollama(model="llama2")
key = "YOURKEYHERE"
llm = ChatOpenAI(
    openai_api_key=key,
    model="gpt-3.5-turbo",
)

import pandas as pd


def analyze_csv(csv_file_path):
    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Basic statistics: column names, count of non-null, mean, std, min, max for numerical columns
    summary = df.describe(include=[pd.np.number]).transpose().to_string()

    # Include information about categorical data
    categorical_summary = (
        df.describe(include=[object, "category"]).transpose().to_string()
    )

    # Combine summaries
    full_summary = f"Numerical Data Summary:\n{summary}\n\nCategorical Data Summary:\n{categorical_summary}"

    return full_summary


def get_chart_suggestions(llm, summary):
    prompt = f"Based on the following dataset summary, suggest 1-2 types of charts that could effectively visualize the data:\n{summary}"
    response = llm.generate(prompt)
    return response


def generate_visualization_code(llm, chart_suggestions):
    prompt = f"Generate Python matplotlib code to create the following charts for the dataset:\n{chart_suggestions}"
    code_response = llm.generate(prompt)
    return code_response


def execute_chart_code(code):
    # Execute the Python code to generate the chart
    exec(code)

    # Save the plot as an image in memory
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Clear the figure to free memory
    plt.clf()

    return image_base64


def generate_charts_from_csv(csv_file_path):
    # Step 1: Analyze CSV
    summary = analyze_csv(csv_file_path)

    # Step 2: Get chart suggestions based on the summary
    chart_suggestions = get_chart_suggestions(llm, summary)

    # Step 3: Generate visualization code based on the suggestions
    visualization_code = generate_visualization_code(llm, chart_suggestions)

    # Step 4: Execute the generated visualization code and get the image
    chart_image_base64 = execute_chart_code(visualization_code)

    return chart_image_base64
