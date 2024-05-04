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
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI

langchain.debug = True

langchain.verbose = True


key = "YOURKEYHERE"
# Create embeddings
# Create embeddings
sqlllm = ChatOpenAI(openai_api_key=key, model="gpt-3.5-turbo", temperature=0)

llm = ChatOpenAI(openai_api_key=key, model="gpt-4-turbo-preview", temperature=0)


# os.environ["OPENAI_API_KEY"] = getpass.getpass()
db = sql_database.SQLDatabase.from_uri("sqlite:///datastore.db")

# connection = sqlite3.connect("test.db")
# cursor = connection.cursor()

# print(db.dialect)
# print(db.get_usable_table_names())
# print(db.run("SELECT * FROM MARKS LIMIT 10;"))

#  query_results = []
#     for question in insight['sub_questions']:
#         response = chain.invoke({"question": question})
#         print(f"Query response for '{question}': {response}")


def no_execute(prompt):

    chain = create_sql_query_chain(llm, db)

    response = chain.invoke({"question": prompt})
    # SQL Query
    print(response)

    print(db.run(response))


@tool("db-agent-tool")
def sqlAgentTool(question: str) -> str:
    """Use this tool to query data. Given a single natural language question as input, this tool can execute one SQL query to perform on the table and returns the answer as a Python List of Tuples in string format"""

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


def interact(context, categorical_values):
    prompt_template = PromptTemplate.from_template(
        """
As a data analyst, you're presented with the following database schema and a sample of the data: {context}. Additionally, you have information about categorical columns and their respective values: {categorical_values}. Your goal is to uncover valuable insights and identify areas for improvement within this dataset for an organization.

Step 1: Analyze the provided schema and data samples, paying special attention to the categorical columns and their values. Consider how these elements can interact and reveal trends, patterns, or anomalies.

Step 2: Based on your analysis, brainstorm and list potential insights that could be valuable to the organization. Think about trends over time, comparisons between categories, or any specific characteristics indicated by the categorical values.

Step 3: For each potential insight, formulate a set of detailed questions that would allow further exploration into the data. These questions should guide the retrieval of additional data through SQL queries, providing the necessary information to validate or investigate the insights further.

Please organize your work into a JSON array of insight objects. Each object should include:
- An 'insight_question' detailing the overarching inquiry or observation.
- A list of 'sub_questions' that specify the data queries needed to explore the insight question further.
- The 'assumption' made, if any, to clarify the context or premise of the insight.
- The recommended 'visualization_chart' to best present the findings, along with a brief 'reasoning' for its choice.

Remember, the focus should be on generating insights that can lead to actionable intelligence for the organization, utilizing both the categorical data and other relevant data points from the schema and samples provided.

Example format for a single insight object (repeat this format for each insight you identify):
{{
    "insight_question": "What insight are we exploring?",
    "sub_questions": [
        "Detailed question 1 to explore the insight?",
        "Detailed question 2 to explore the insight?"
    ],
    "assumption": "Any specific assumptions made for this analysis.",
    "visualization_chart": "Recommended chart type",
    "reasoning": "Why this chart type is chosen for the insight."
}}

Remember, aim for a minimum of 10 insightful questions that delve into different aspects of the data.
"""
    )

    formatted_prompt = prompt_template.format(
        context=context, categorical_values=categorical_values
    )
    response = llm.invoke(formatted_prompt)
    # response = response.content.replace("\n", "")
    # print(response)
    return response


def sql_agent(prompt: str) -> str:
    execute_query = QuerySQLDataBaseTool(db=db)

    write_query = create_sql_query_chain(llm, db)
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
    print(result)
    print(type(result))


def fetch_categorical_values(db, table_name, categorical_columns):
    categorical_values = {}
    for column in categorical_columns:
        values_query = f"SELECT DISTINCT `{column}` FROM `{table_name}`"
        values_str = db.run(values_query)
        try:
            # Ensure the result is properly formatted and evaluated
            values = ast.literal_eval(values_str)
            # Extract the distinct values and store them
            categorical_values[column] = [value[0] for value in values]
        except Exception as e:
            print(f"Error processing values for {column}: {e}")
            categorical_values[column] = []

    return categorical_values


def identify_categorical_columns(db, table_name, threshold=10):
    categorical_columns = []
    structure_str = db.run(f"PRAGMA table_info('{table_name}')")
    try:
        structure = ast.literal_eval(structure_str)
    except (ValueError, SyntaxError) as e:
        print(f"Error evaluating table structure: {e}")
        return []

    for column in structure:
        column_name = column[1]  # Extract the column name
        try:
            unique_count_str = db.run(
                f"SELECT COUNT(DISTINCT `{column_name}`) FROM `{table_name}`"
            )
            # Ensure the result is properly formatted and evaluated
            unique_count_result = ast.literal_eval(unique_count_str)
            # Extract the first element (the count) and ensure it's an integer
            unique_count = int(unique_count_result[0][0])
        except (ValueError, SyntaxError, TypeError) as e:
            print(f"Error processing unique count for {column_name}: {e}")
            continue

        if unique_count <= threshold:
            categorical_columns.append(column_name)

    return categorical_columns


def extract_json(text):
    try:
        # Use regex to find JSON part between ```json ``` and remove it
        json_str = re.search(r"```json(.*?)```", text, re.DOTALL).group(1).strip()
        # Load it to check if it's valid JSON
        json_obj = json.loads(json_str)
        # Save it to a JSON file
        with open("insights_ganeesha.json", "w", encoding="utf-8") as f:
            json.dump(json_obj, f, ensure_ascii=False, indent=4)
        return "Valid JSON extracted and saved to insights_ganeesha.json"
    except json.JSONDecodeError as e:
        return f"Invalid JSON found: {e}"
    except Exception as e:
        return f"An error occurred: {e}"


if __name__ == "__main__":
    # b = run.invoke("What is the average score for English?")
    exp_prompt_template = """
Please uncover valuable insights and identify areas for improvement within this database for an organization.
Always use the "db-agent-tool" tool to query the database and get answers where required.
Step 1:  Use the "db-agent-tool" tool to analyze the database, paying special attention to identified table, its categorical columns and their values. Consider how these elements can interact and reveal trends, patterns, or anomalies.

Step 2: Based on your analysis, brainstorm and list potential insights that could be valuable to the organization. Think about trends over time, comparisons between categories, or any specific characteristics indicated by the categorical values.

Step 3: For each potential insight, formulate a set of detailed questions that would allow further exploration into the data. These questions should guide the retrieval of additional data through SQL queries, providing the necessary information to validate or investigate the insights further.

Please organize your work into a JSON array of insight objects. Each object should include:
- An 'insight_question' detailing the overarching inquiry or observation.
- "questions_and_answers_from_tool": Array of Json objects of natural language questions you need to ask the db-agent-tool and the response you get from the db-agent tool you must include in the answer. Include as many questions as required to answer the insight_question.
- The 'assumption' made, if any, to clarify the context or premise of the insight.
- The recommended 'visualization_chart' to best present the findings, along with a brief 'reasoning' for its choice.

Remember, the focus should be on generating insights that can lead to actionable intelligence for the organization, utilizing both the categorical data and other relevant data points from the schema and samples provided.

Example format for a single insight object (repeat this format for each insight you identify):
{{
    "insight_question": "What insight are we exploring?",
    "questions_and_answers_from_tool": [ {{"question": "What is the average of X?", "answer":"Y"}}],
    "assumption": "Any specific assumptions made for this analysis.",
    "visualization_chart": "Recommended chart type",
    "reasoning": "Why this chart type is chosen for the insight."
}}

Aim for a minimum of 03 insightful questions that delve into different aspects of the data.
"""

    exp_prompt_template2 = """ 
    Here is the database schema: {context}.
    Here are the categorical columns and their values: {categorical_values}.
Always use the "db-agent-tool" tool to query the database and get answers where required. DO NOT HALLUCINATE or try to make up your own database.
Avoid querying all the data as you have a token limit. You need to intelligently ask the questions from the "db-agent-tool".
    Your task is to uncover 03 valuable insights and identify areas for improvement if any for the organization. 
    Please organize your insights and give the final output as a JSON array of 03 insight objects. 

    Here's how you should proceed:

Step 1:  Use the "db-agent-tool" tool to analyze the database, paying special attention to identified table, its categorical columns and their values. 
Consider how these elements can interact and reveal trends, patterns, or anomalies and come up with a useful insight question.
   Step 2. Ask natural language questions one by one from the "db-agent-tool" tool to get data from the database to get the answers to your insight question. DONT try to make up or hallucinate answers on your own.
    Step 3. Interpret the tool's responses and summarize your findings.
    Step 4. Recommend the best chart type for visualizing each insight and provide the reasoning for choosing the chart type. Give all chart parameters required to make the chart in the future.
    Also add all the data required to generate the chart. Use the "db-agent-tool" tool to get the data required for the chart.
    Step 5. Finally, present your findings as a JSON array of comma separated objects as shown below. The array should consist of 03 insight objects that delve into different aspects of the data.
    Here is an example JSON format for your final output:
    ```json[
    {{
        "insight_question": "The insight you explored goes here.  eg: What is the average marks for Math?",
        "questions_and_answers_from_tool": 
        [ {{"question": "Your sub question 1 here. eg: What is the average marks for Math in Term 1?", 
        "answer": "Use the "db-agent-tool" tool and put its output here. eg: 52.5}}, ... //continue adding your sub questions and the answers here],
        "summary": "Summary of the useful insight, after getting the answers through the db-agent-tool. Conclusion of the answers and any recommendations or improvements for the organization are stated here.",
        "assumption": "Any specific assumptions made for this analysis.",
        "visualization_chart": {{"chart_type": "Recommended chart type. eg: Bar Chart", 
        "reasoning": "Why this chart type is chosen for the insight.", "parameters": "Include all the parameters required to generate the chart here. 
        eg: x_axis, y_axis, title, colors.", "chart_data": "Include all the data required to generate the chart here." }},
        
        
    }},  ... //continue adding insight objects here]
    ```
    

    """

    exp_prompt_template3 = """ 
    Here is the database schema: {context}.
    Here are the categorical columns and their values: {categorical_values}.
Always use the "db-agent-tool" tool to query the database and get answers where required. DO NOT HALLUCINATE or try to make up your own database.
Avoid querying all the data as you have a token limit. You need to intelligently ask the questions from the "db-agent-tool".
    Your task is to uncover valuable insights and identify areas for improvement if any for the organization. 
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
        "insight_question": "The insight you explored goes here.  eg: What is the average marks for Math?",
        "questions_and_answers_from_tool": 
        [ {{"question": "Your sub question 1 here. eg: What is the average marks for Math in Term 1?", 
        "answer": "Use the "db-agent-tool" tool and put its output here. eg: 52.5}}, 
        {{"question": "Your sub question 1 here. eg: What is the average marks for Math in Term 2?", 
        "answer": "Use the "db-agent-tool" tool and put its output here. eg: 52.5}},
        ... //continue adding your sub questions and the answers here],
        "summary": "Summary of the useful insight, after getting the answers through the db-agent-tool. Conclusion of the answers and any recommendations or improvements for the organization are stated here.",
        "assumption": "Any specific assumptions made for this analysis. eg: If dataset is on marks, marks scored below 50 is considered as fail. Leave this blank if none.",
        "visualization_chart": {{"chart_type": "Recommended chart type. eg: Bar Chart", 
        "reasoning": "Why this chart type is chosen for the insight.", "parameters": "Include all the parameters required to generate the chart here. 
        eg: x_axis, y_axis, title, colors.", "chart_data": "Include all the data required to generate the chart here." }},
         
    }},  ... //continue adding insight objects here]
    ```
    """
    ## we could add more prompts for additional information given by user as input. Here is some more info about the dataset.
    prompt_template_agent = PromptTemplate.from_template(exp_prompt_template3)

    tools = [sqlAgentTool]
    llm_with_tools = llm.bind_tools(tools)

    # run()
    tables = db.run("SELECT name FROM sqlite_master WHERE type='table'")

    tables = ast.literal_eval(tables)
    for table in tables:

        table_name = table[0]

        structure = db.run(f"PRAGMA table_info('{table_name}')")

        table_info = db.get_table_info(table_names=[table_name])

        categorical_columns = identify_categorical_columns(db, table_name)
        categorical_values = fetch_categorical_values(
            db, table_name, categorical_columns
        )

        categorical_values = json.dumps(categorical_values, indent=2)

        print(categorical_values)
        formatted_prompt = prompt_template_agent.format(
            context=structure, categorical_values=categorical_values
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

        pr2 = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a highly skilled data analyst with access to a powerful tool called 'db-agent-tool',
                    which can query a database directly based on natural language questions. Remember, you don't know SQL and cannot manually write SQL queries.
                    Instead, always use the 'db-agent-tool' tool to generate and execute SQL queries for you.
                    Before answering any question related to the database, make sure to use the 'db-agent-tool' tool to fetch the relevant data.
                    NEVER hallucinate or come up with a database on your own.
                    
                           Use the following format:

                Question: the input question you must answer
                Thought: you should always think about what to do
                Action: the action to take, should be [{tool_names}]
                Action Input: the input to the action
                Observation: the result of the action
                ... (this Thought/Action/Action Input/Observation can repeat N times)
                Thought: I now know the final answer
                Final Answer: the final answer to the original input question

                Begin!

                Question: {input}

                """,
                ),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        # pr.format_messages(
        #     context=structure,
        #     categorical_values=categorical_values,
        # )
        response_schemas = [
            ResponseSchema(
                name="insight_question",
                description="The main question or observation being explored.",
            ),
            ResponseSchema(
                name="summary",
                description="A summary of the insight derived from the data.",
            ),
            ResponseSchema(
                name="assumption",
                description="Any assumptions made during the analysis.",
            ),
            ResponseSchema(
                name="visualization_chart",
                description="The type of chart recommended for presenting the findings.",
            ),
            ResponseSchema(
                name="reasoning",
                description="Justification for selecting the chart type.",
            ),
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        # agent = create_openai_functions_agent(llm, tools, pr)
        # https://github.com/SandyShah/langchain_agents/blob/main/medium_post.ipynb - try with llama
        tool_names = ", ".join([t.name for t in tools])
        llm_with_stop = llm.bind(stop=["\nObservation"])
        # output_parser = ReActSingleInputOutputParser()
        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                ),
                "context": lambda x: structure,
                "categorical_values": lambda x: categorical_values,
                # "tool_names": lambda x: tool_names,
            }
            | pr
            | llm_with_tools
            # | llm_with_stop
            # | output_parser
            | OpenAIToolsAgentOutputParser()
        )
        # agent = create_react_agent(llm, tools, pr2)

        agent_executor = AgentExecutor(
            agent=agent, tools=tools, verbose=True, return_intermediate_steps=True
        )

        a = agent_executor.invoke({"input": formatted_prompt})

        print(type(a))
        with open("insights_ganeesha.txt", "w", encoding="utf-8") as file:
            file.write(str(a))

        print("Output written to insights_ganeesha.txt")

        output = a["output"]

        # Convert string escape sequences into actual escape characters
        formatted_output = bytes(output, "utf-8").decode("unicode_escape")

        # Now write this formatted output to a text file
        with open("output_formatted_ganeesha.txt", "w", encoding="utf-8") as file:
            file.write(formatted_output)

        print("Formatted output written to output_formatted_ganeesha.txt")

        extract_json(output)

        # Remove the potential markdown JSON indicators if present.
        output = output.replace("```json", "").replace("```", "").strip()

        # Parse the JSON string into a Python dictionary.
        try:
            data = json.loads(output)
        except json.JSONDecodeError as e:
            print("Failed to decode JSON:", e)
        else:
            # Now write the Python dictionary as a properly formatted JSON file.
            with open("insights_ganeesha.json", "w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=4)

            print("Output written to insights_ganeesha.json")

        # interact(table_info, categorical_values)
        # todo: improve prompt's json output. should be array of comma separated json objects. convert file to json not txt.
        # For chart params give structured properties like chart type, axis, title, colors.
        # Call the chart generation code now. in visualizer.py. have templates all possible chart types.
        # Or get LLM itself to generate and execute the chart code as done in text_loader.py
