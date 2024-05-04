# from langchain.agents import Tool

# from langchain.agents import load_tools
# from langchain.agents import initialize_agent
# from langchain.agents import AgentType

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
import chromadb

set_debug(True)
# set_verbose(True)
# llm = Ollama(model="llama2")
key = "YOURKEYHERE"


def is_calculation_question(question):
    calculation_keywords = ["calculate", "average", "sum", "total"]
    return any(keyword in question.lower() for keyword in calculation_keywords)


# Create embeddings
llm = ChatOpenAI(
    openai_api_key=key,
    model="gpt-3.5-turbo",
)
# text-embedding-ada-002
embeddings = OpenAIEmbeddings(openai_api_key=key)


def transform_row_to_sentence(row):
    # Transform the row into a more natural sentence
    # TODO: Get the LLM to come up with the structure to convert a row into a sentence
    return f"{row['Student_Name']} in {row['Subject']} scored {row['Marks_out_of_100']} out of 100 in Term {row['Term']}."


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
    # # embeddings = OllamaEmbeddings(model="llama2")
    # document_sentences = [transform_row_to_sentence(doc) for doc in documents]

    # df = pd.read_csv("./csvs/Marks.csv")
    # document_sentences = [
    #     transform_row_to_sentence(row) for index, row in df.iterrows()
    # ]

    # query_result = embeddings.embed_documents(document_sentences)
    # print(query_result[:5])
    # embeddings.embed_documents(documents)

    # persist_directory = "./chroma_db"
    # persist_directory_gpt3 = "./chroma_db_gpt3"  # sentences
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

    # persistent_client = chromadb.PersistentClient()
    # collection_name = "Marks"
    # collection = persistent_client.get_or_create_collection(collection_name)
    # collection.add(documents=documents)
    # Initialize Chroma with the persistent client
    # chroma_db = Chroma(
    #     client=persistent_client,
    #     collection_name=collection_name,
    #     embedding_function=embeddings,
    # )

    # Count documents in the collection
    # print("There are", chroma_db._collection.count(), "documents in the collection.")

    # llm = Ollama(model="llama2")

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
    prompt = """Based on dataset attributes such as {attributes_string}, 
    what are 03 useful high-level analytical points that can we infer about the overall trends and patterns in the data? 
    Then suggest the types of visualization charts that would best represent these insights. 
    Provide your response in a structured JSON format with 'insights' and 'charts' as keys.
    Do NOT focus on individual values. 
    """

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

    # detailed_prompt = """Please provide up to three analytical points that could offer a comprehensive understanding of the overall data trends and patterns, "
    # "accompanied by suggestions for types of visualization charts that would best represent each point. "
    # "Format your response as a JSON object with keys 'insights' and 'charts', where 'insights' are generic analytical points applicable to the entire dataset "
    # "and 'charts' are lists of corresponding chart types. Do NOT focus on actual values."""

    # detailed_prompt = """
    # Please suggest three useful analytical points that could provide a comprehensive understanding of the given domain,
    # along with a suitable visualization chart that a data analyst could make for each point.
    # Format your response as a JSON object array, for example'.
    # Please do NOT focus on actual values, the given values are just for you to get an idea of the domain."""

    detailed_prompt = """
    "Given a dataset with various attributes related to a specific domain, please analyze the data 
    to suggest three useful analytical insights. 
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


# charts_result = qa_chain_chart({"query": "Chart"})
# full_response = charts_result["result"]
# Extract and execute the chart codes
# chart_codes = re.findall(r"```(.*?)```", full_response, re.DOTALL)
# images_base64 = []
# print("chart_codes", chart_codes)
# df = pd.read_csv("Marks.csv")
# if not chart_codes:
#     print("No chart code was generated.")
# else:
#     for python_code in chart_codes:
#         python_code = python_code.replace("python", "").strip()
#         python_code = python_code.strip().replace("plt.show()", "# plt.show()")
#         globals()["df"] = df

#         exec(python_code, globals())

#         # Convert plot to base64
#         buf = io.BytesIO()
#         plt.savefig(buf, format="png")
#         buf.seek(0)
#         images_base64.append(base64.b64encode(buf.read()).decode("utf-8"))
#         plt.clf()  # Clear the current figure

# return {"insights": insights_result, "charts": images_base64}


prompt_template_tailwind = PromptTemplate(
    input_variables=["insights", "charts_base64"],
    template=(
        "Given these insights: {insights} and chart images encoded in base64: {charts_base64}, "
        "create a JSON object for a React component. The object should have properties: 'heading', "
        "'content', 'conclusion', 'bgDivTailwindCSS', and 'chart'. Use appropriate Tailwind CSS classes "
        "for styling. Format the insights as bullet points in the 'content' property, and include the chart "
        "image. Ensure the JSON object follows this structure: \n\n"
        "const sample = [...]"
    ),
)

if __name__ == "__main__":
    # Get insights and charts
    insights_and_charts = get_insights()
    # insights = insights_and_charts["insights"]
    # charts_base64 = insights_and_charts["charts"]

    # tailwind_chain = LLMChain(
    #     llm=llm, prompt=prompt_template_tailwind, output_key="tailwind_json"
    # )

    # # Generate the Tailwind CSS JSON object
    # tailwind_json = tailwind_chain(
    #     {"insights": insights, "charts_base64": charts_base64}
    # )

    # print(tailwind_json)

    # prompt_template_chart = PromptTemplate(
    #     input_variables=["context"],
    #     template=f"Given this dataset: {{context}}, please generate Python code for the most suitable and insightful chart visualizations using matplotlib. DO NOT include plt.show(). Each chart should be a separate function returning a plt object. DO NOT include code explanations. Give the code directly. "
    #     "ALWAYS format the code within triple backticks at the start and end. "
    #     "ALWAYS define variables based on available data only before you use them in the code. AVOID syntax errors.",
    # )

    # qa_chain_chart = RetrievalQA.from_chain_type(
    #     llm,
    #     retriever=chroma_db.as_retriever(),
    #     chain_type_kwargs={"prompt": prompt_template_chart},
    # )

    #     detailed_prompt = """Given a dataset containing student performance across various subjects and terms, with columns for Student Name, Subject, Marks out of 100, and Term, I want meaningful insights to better understand patterns, trends, and areas for improvement. The dataset structure is as follows:
    #     - Student_Name: Name of the student
    #     - Subject: Academic subject
    #     - Marks_out_of_100: Score obtained by the student, out of a total of 100 marks
    #     - Term: The academic term (e.g., 1, 2)
    # Please suggest a variety of analytical points that could provide a comprehensive understanding of student performance."""
