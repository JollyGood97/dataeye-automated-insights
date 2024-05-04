import sqlite3
import uuid
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
from langchain_community.document_loaders import TextLoader
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
from sqlcreator import identify_categorical_columns, fetch_categorical_values
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI

# langchain.debug = True

# langchain.verbose = True
from langsmith.wrappers import wrap_openai
from langsmith import traceable
import os
from dotenv import load_dotenv

load_dotenv()


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Auto-trace LLM calls in-context
client = wrap_openai(openai.Client())

key = "YOURKEYHERE"
# Create embeddings
# Create embeddings
sqlllm = ChatOpenAI(openai_api_key=key, model="gpt-3.5-turbo", temperature=0)
# llm = ChatOpenAI(openai_api_key=key, model="gpt-3.5-turbo", temperature=0.7)

llm = ChatOpenAI(openai_api_key=key, model="gpt-4-turbo-preview", temperature=0.7)


# os.environ["OPENAI_API_KEY"] = getpass.getpass()
db = sql_database.SQLDatabase.from_uri("sqlite:///datastore.db")


def extract_and_validate_json(text, filename):
    try:
        # Extract JSON using regex, assuming it's enclosed in ```json ... ```
        json_str = re.search(r"```json(.*?)```", text, re.DOTALL).group(1).strip()

        # Try parsing the JSON to validate it
        json_obj = json.loads(json_str)

        # Save the validated JSON to a file
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(json_obj, f, ensure_ascii=False, indent=4)

        return json_obj

    except json.JSONDecodeError as e:
        return f"JSON decoding failed: {e}"

    except AttributeError:
        return "No JSON found in the text."

    except Exception as e:
        return f"An error occurred: {e}"


exp_prompt_template3 = """ 
    Here is the database schema: {context}.
    Here are the categorical columns and their values: {categorical_values}.
    Here is some information about the data (might or might not be available): {additional_info}.
Always use the "db-agent-tool" tool to query the database and get answers where required. DO NOT HALLUCINATE or try to make up your own database.
Avoid querying all the data as you have a token limit. You need to intelligently ask the questions from the "db-agent-tool".
Your ultimate goal is to uncover valuable insights that are specifically relevant to the domain of the dataset and identify critical factors and relationships that could influence decisions or 
operational effectiveness for the organization or entity.
    
    Please organize your insights and give the final output as a JSON array of insight objects. 
    You must yourself decide the number of insights required to uncover all possible insights from the data, so the JSON array will consist of that number of insight objects.

    Here's how you should proceed:

Step 1:  Use the "db-agent-tool" tool to analyze the database, paying special attention to identified table, its categorical columns and their values. 
Consider how these elements can interact and reveal trends, patterns, or anomalies and come up with a useful insight question.
   Step 2. Ask natural language questions one by one from the "db-agent-tool" tool to get data from the database to get the answers to your insight question. DONT try to make up or hallucinate answers on your own.
    Step 3. Interpret the tool's responses and summarize your findings.
    Step 4. Recommend the best chart type for visualizing each insight and provide the reasoning for choosing the chart type. Give all chart parameters required to make the chart in the future.
    Also add all the data required to generate the chart. Use the "db-agent-tool" tool to get the data required for the chart.
    Step 5. Finally, present your findings as a JSON array of comma separated objects as shown below. The array should consist of insight objects that delve into different aspects of the data. eg: If there is a column for marks and a column for student, consider marks wise, student wise.
    Here is an example JSON format for your final output:
    ```json[
    {{
        "insight_question": "The insight you explored goes here.  eg: What is the subject students struggle with?",
        "questions_and_answers_from_tool": 
        [ {{"question": "Your sub question 1 here. eg: What is the average marks for Math in Term 1?", 
        "answer": "Use the "db-agent-tool" tool and put its output here. eg: 52.5}}, 
        {{"question": "Your sub question 1 here. eg: What is the average marks for Math in Term 2?", 
        "answer": "Use the "db-agent-tool" tool and put its output here. eg: 62.5}},
        ... //continue adding your sub questions and the answers here],
        "summary": "Summary of the useful insight, after getting the answers through the db-agent-tool. Conclusion of the answers and any recommendations or improvements for the organization are stated here.",
        "assumption": "Any specific assumptions made for this analysis. eg: If dataset is on marks, marks scored below 50 is considered as fail. Leave this blank if none.",
        "visualization_chart": {{"chart_type": "Recommended chart type. eg: Bar Chart", 
        "reasoning": "Why this chart type is chosen for the insight.", "parameters": "Include all the parameters required to generate the chart here. 
        eg: x_axis, y_axis, title, colors.", "chart_data": "Include all the data required to generate the chart here." }},
         
    }},  ... //continue adding insight objects here. Minimum number of insight objects should be 03.]
    ```
    """
## we could add more prompts for additional information given by user as input. Here is some more info about the dataset.


@tool("db-agent-tool")
def sqlAgentTool(question: str) -> str:
    """Use this tool to query data. Given a single natural language question as input, executes an SQL query to perform on the table and returns the answer as a Python List of Tuples in string format"""

    try:
        execute_query = QuerySQLDataBaseTool(db=db)
        write_query = create_sql_query_chain(sqlllm, db)
        chain = write_query | execute_query
        response = chain.invoke({"question": question})
        print("response from db-agent-tool", response)
        return response
    except Exception as e:
        print(f"An error occurred: {e}")
        return str(e)


@traceable  # Auto-trace this function
def generate_insights(additional_info=" "):
    prompt_template_agent = PromptTemplate.from_template(exp_prompt_template3)

    tools = [sqlAgentTool]
    llm_with_tools = llm.bind_tools(tools)

    # run()
    tables = db.run("SELECT name FROM sqlite_master WHERE type='table'")

    tables = ast.literal_eval(tables)
    for table in tables:

        table_name = table[0]

        structure = db.run(f"PRAGMA table_info('{table_name}')")

        # table_info = db.get_table_info(table_names=[table_name])
        print("table_info:", table_name)

        categorical_columns = identify_categorical_columns(db, table_name)
        categorical_values = fetch_categorical_values(
            db, table_name, categorical_columns
        )

        categorical_values = json.dumps(categorical_values, indent=2)

        formatted_prompt = prompt_template_agent.format(
            context=structure,
            categorical_values=categorical_values,
            additional_info=additional_info,
        )

        pr = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a highly skilled data analyst with access to a powerful tool called 'db-agent-tool',
                    which can query a database directly based on natural language questions. Remember, you don't know SQL and cannot manually write SQL queries.
                    Instead, always use the 'db-agent-tool' tool to generate and execute SQL queries for you.
                    Before answering any question related to the database, make sure to use the 'db-agent-tool' tool to fetch the relevant data.
                    NEVER hallucinate or come up with a database on your own.


                """,
                ),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                ),
                "context": lambda x: structure,
                "categorical_values": lambda x: categorical_values,
                "additional_info": lambda x: additional_info,
            }
            | pr
            | llm_with_tools
            | OpenAIToolsAgentOutputParser()
        )

        agent_executor = AgentExecutor(
            agent=agent, tools=tools, verbose=True, return_intermediate_steps=True
        )

        a = agent_executor.invoke({"input": formatted_prompt})

        unique_id = str(uuid.uuid4())
        insights_filename = f"insights_{unique_id}.json"

        with open(f"insights_{unique_id}.txt", "w", encoding="utf-8") as file:
            file.write(str(a))

        print("Output written to txt")
        output = a["output"]
        # Convert string escape sequences into actual escape characters
        formatted_output = bytes(output, "utf-8").decode("unicode_escape")

        # Now write this formatted output to a text file
        with open(f"insights_{unique_id}.txt", "w", encoding="utf-8") as file:
            file.write(formatted_output)

        print(f"Formatted output written to insights_{unique_id}.txt")

        json_data = extract_and_validate_json(output, insights_filename)

        print(f"Output written to {insights_filename}")
        return json_data


# if __name__ == "__main__":
#     generate_insights()
