import sqlite3
from langchain_community.utilities import sql_database
import getpass
import os
from langchain_community.llms import Ollama
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.agents.output_parsers import ReActSingleInputOutputParser


from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader, JSONLoader
from operator import itemgetter
import json
import ast
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.agents import tool

from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_openai_functions_agent,
    create_react_agent,
)
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
import json
from pathlib import Path
from pprint import pprint

langchain.debug = True

langchain.verbose = True

key = "YOURKEYHERE"
# Create embeddings
# Create embeddings
sqlllm = ChatOpenAI(openai_api_key=key, model="gpt-3.5-turbo", temperature=0)
# llm = Ollama(model="llama2")
# gpt-4-turbo-preview
llm = ChatOpenAI(openai_api_key=key, model="gpt-4-turbo-preview", temperature=0)
db = sql_database.SQLDatabase.from_uri("sqlite:///datastore.db")
output_parser = StrOutputParser()

csv_path = "./csvs/Marks.csv"
persist_directory_gpt3 = "./chroma_db_gpt3_textloader"
jsondoc = "insights_ganeesha.json"


def get_insights_info(detailed_prompt):
    # data = json.loads(Path(jsondoc).read_text())

    loader = JSONLoader(
        file_path=jsondoc, jq_schema=".messages[].content", text_content=False
    )

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
        template="Given this context: {context}, answer the question: {question}",
    )

    # Set up the question-answering chain for insights
    qa_chain_insights = RetrievalQA.from_chain_type(
        llm,
        retriever=chroma_db.as_retriever(search_type="mmr"),
        chain_type_kwargs={"prompt": prompt_template_insights},
        # kwargs={"k": 5},
    )

    insights_result = qa_chain_insights({"query": detailed_prompt})
    response = str(insights_result)

    # Safely evaluate the string to a Python dictionary
    parsed_response = ast.literal_eval(response)

    # Access the 'result' key from the parsed response
    result_value = parsed_response["result"]
    print(result_value)


def sql_agent(prompt: str) -> str:
    execute_query = QuerySQLDataBaseTool(db=db)

    write_query = create_sql_query_chain(sqlllm, db)
    answer_prompt = PromptTemplate.from_template(
        """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer: """
    )

    answer = answer_prompt | llm | StrOutputParser()
    chain = (
        RunnablePassthrough.assign(query=write_query).assign(
            result=itemgetter("query") | execute_query
        )
        | answer
    )

    result = chain.invoke({"question": prompt})
    return result


# def interact(prompt):
#     response = llm.invoke(prompt)
#     print(response)


# if __name__ == "__main__":
#     prompt = "Who scored highest marks for Math in Term 1?"
#     resp = sql_agent(prompt)
#     print(resp)
