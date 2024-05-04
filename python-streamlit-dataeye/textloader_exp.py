import ast
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader

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

output_parser = StrOutputParser()


langchain.debug = True

langchain.verbose = True

key = "YOURKEYHERE"
# Create embeddings
llm = ChatOpenAI(
    openai_api_key=key,
    model="gpt-4-turbo-preview",
)

csv_path = "./csvs/Marks.csv"
persist_directory_gpt3 = "./chroma_db_gpt3_textloader"


def csv_to_text_and_summary(csv_file_path):
    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Basic statistics for numerical columns
    numerical_summary = df.describe().transpose()

    # Categorical data summary
    categorical_summary = df.describe(include=["object", "category"]).transpose()

    # Combine summaries
    full_summary = f"Numerical Data Summary:\n{numerical_summary}\n\nCategorical Data Summary:\n{categorical_summary}"

    # Get the file name and prepare text file name
    file_name = csv_file_path.split("/")[-1]
    txt_file_name = file_name.rsplit(".", 1)[0] + ".txt"

    # Prepare the initial text document content
    text_doc = f"Uploaded CSV Dataset Summary:\n- File Name: {file_name}\n- Total Rows: {len(df)}\n- File Path: {csv_file_path}\n"
    text_doc += f"This CSV contains these column headers or attributes in comma separated form: {', '.join(df.columns)}\n"
    text_doc += "Summary generated from the dataset:\n"
    text_doc += full_summary + "\n"
    text_doc += "This CSV contains the data for each column as an array:\n"

    # Adding row data for each column as an array
    for column in df.columns:
        # Convert each column's data to a list and then to a JSON string for readability
        column_data = df[column].tolist()
        text_doc += f"{column}: {json.dumps(column_data)}\n"

    # Generate a prompt for LLM to get further analysis or suggestions
    # prompt = f"Provide a summary for a dataset with the following columns: {', '.join(df.columns)}."
    # prompt += f" The dataset includes numerical columns such as {', '.join(numerical_summary.index)} and categorical columns such as {', '.join(categorical_summary.index)}."
    # Assuming 'llm.invoke' is the method to get response from your LLM
    # llm_summary = llm.invoke(prompt)
    # print(llm_summary)

    # # Append LLM-generated summary to text document
    # text_doc += f"\nLLM-Generated Summary and Visualization Suggestions:\n{llm_summary}"

    # Save the document text to a file in ./docs/ directory
    docs_dir = "./docs/"
    os.makedirs(docs_dir, exist_ok=True)  # Ensure the directory exists
    txt_file_path = os.path.join(docs_dir, txt_file_name)
    with open(txt_file_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(text_doc)

    print(f"Document saved to: {txt_file_path}")
    return txt_file_path


def generate_matplotlib_code(insight, chart_type, chart_description, chroma_db):
    prompt_template_insights = PromptTemplate(
        input_variables=["context"],
        # template="This context has only few data available from a larger dataset: {context}, shared with you to answer the following: {question}",
        # template="Given these data attributes of a larger dataset: {context}, {question}",
        template="  Given the data required for the chart as follows: {context}, {question}",
    )

    prompt = f"""
        Based on the insight "{insight}" and the decision to use a {chart_type} as explained in the chart description: "{chart_description}",
        generate the Python Matplotlib code necessary to visualize the insight "{insight}" using a {chart_type}. 
        The code should include import statements for any required libraries, 
        and the plot should be fully labeled with a title, x-label, and y-label as appropriate. 
        Do NOT include plt.show() at the end. Please ensure the code is ready to run and
        effectively communicates the insight based on the provided chart description
        """

    # Set up the question-answering chain for insights
    qa_chain_insights = RetrievalQA.from_chain_type(
        llm,
        retriever=chroma_db.as_retriever(search_type="mmr"),
        chain_type_kwargs={"prompt": prompt_template_insights},
        # kwargs={"k": 5},
    )

    charts_result = qa_chain_insights({"query": prompt})

    # Extract the code from the charts_result
    matplotlib_code = charts_result["result"]

    # Return the extracted Matplotlib code
    return matplotlib_code


def execute_and_capture_chart(code):
    """
    Execute the given matplotlib code and capture the output as an image.

    Args:
    - code (str): Python code to execute, which generates a matplotlib figure.

    Returns:
    - PIL Image object of the matplotlib figure.
    """
    # Creating a custom namespace for exec to run in, importing common libraries
    namespace = {
        "plt": plt,
        "np": np,
        "io": io,
        "Image": Image,
    }

    # Execute the provided code in the defined namespace
    exec(code, namespace)

    # Save the matplotlib figure to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()  # Close the plot to free up memory
    buf.seek(0)

    # Return the image object
    return Image.open(buf)


def generate_insights_and_codes():
    txt_file_path = csv_to_text_and_summary(csv_path)
    loader = TextLoader(txt_file_path)
    documents = loader.load()

    # text-embedding-ada-002
    embeddings = OpenAIEmbeddings(openai_api_key=key, model="text-embedding-ada-002")

    # Set up ChromaDB first time only
    chroma_db = Chroma.from_documents(
        documents, embeddings, persist_directory=persist_directory_gpt3
    )
    chroma_db.persist()

    # load if already persisted
    # chroma_db = Chroma(
    #     persist_directory=persist_directory_gpt3, embedding_function=embeddings
    # )

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
        chain_type="map_reduce",
        retriever=chroma_db.as_retriever(),
        chain_type_kwargs={"prompt": prompt_template_insights},
        # kwargs={"k": 5},
    )

    detailed_prompt = """
    
"I have a dataset containing student marks across various subjects for multiple terms, with each row including the 
student's name, the subject, the term, and their score. 
I am looking for detailed insights from this data, alongside recommendations 
for the most appropriate visualizations to represent these insights clearly, 
and a brief description of why each visualization is suitable. 
Please analyze the dataset to identify interesting patterns, anomalies, or trends.
For each insight, recommend a visualization chart and provide a reasoning for its choice. 
Please format your response as a JSON object array, 
where each object contains keys 'insight', 'chart', and 'chart_description', 
representing the insight, the suggested chart type, and the reasoning for the chart choice respectively. 
For clarity, here is an example of the expected output format:

[
    {
        'insight': 'Bob has had the lowest marks for maths in Term 2 with a score of 45.',
        'chart': 'Bar chart',
        'chart_description': 'A bar chart comparing Bob's math scores across all terms to visualize his performance trend clearly highlights his performance relative to his peers and shows his progress or lack thereof over time.'
    },
    {
        'insight': 'Alice has shown the most improvement in Science from Term 1 to Term 3.',
        'chart': 'Line chart',
        'chart_description': 'A line chart plotting Alice's science scores across terms illustrates her upward trajectory in performance, making it easy to identify the trend of improvement over time.'
    },
    {
        'insight': 'The overall class performance in English has significantly declined from Term 1 to Term 3.',
        'chart': 'Line chart',
        'chart_description': 'A line chart with terms on the x-axis and average scores on the y-axis can clearly show the trend of declining performance, highlighting the need for interventions.'
    }
]

Please provide three analytical points along with the suggested visualization charts and the reasoning for each choice in the format shown above."


    """
    ## Otherwise LLM focuses on a single student in the retrieved context
    insights_result = qa_chain_insights({"query": detailed_prompt})
    response = str(insights_result)

    # Safely evaluate the string to a Python dictionary
    parsed_response = ast.literal_eval(response)

    # Access the 'result' key from the parsed response
    result_value = parsed_response["result"]
    print(result_value)
    # print(insights_result)


##detailed_prompt
# Deserialize the JSON string back into a Python list of dictionaries
# try:
#     result = json.loads(insights_result["result"])
#     print("Deserialized result:", result)
#     # print(type(result))
# except json.JSONDecodeError as e:
#     print("Parsing error:", e)
#     # This block catches parsing errors and prints them

# insights_and_codes = []
# # Iterate over each insight and generate Matplotlib code
# for insight_result in result:
#     insight = insight_result["insight"]
#     chart_type = insight_result["chart"]
#     chart_description = insight_result["chart_description"]
#     print("insight", insight)
#     print("chart_type", chart_type)

#     matplotlib_code = generate_matplotlib_code(
#         insight, chart_type, chart_description, chroma_db
#     )

#     # Generate the Matplotlib code for the current insight and chart type
#     matplotlib_code = (
#         matplotlib_code.replace("```python", "")
#         .replace("```", "")
#         .replace("plt.show()", "")  # Add this line to remove plt.show()
#         .strip()
#     )
#     print("matplotlib_code", matplotlib_code)
#     insights_and_codes.append((insight, matplotlib_code))

# return insights_and_codes


if __name__ == "__main__":
    generate_insights_and_codes()
