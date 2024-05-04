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

langchain.debug = True

langchain.verbose = True


key = "YOURKEYHERE"

# Create embeddings
llm = ChatOpenAI(
    openai_api_key=key,
    model="gpt-3.5-turbo",
)
# text-embedding-ada-002
embeddings = OpenAIEmbeddings(openai_api_key=key)
detailed_analysis_request = (
    "Please provide a detailed analysis including statistical measures such as mean, "
    "median, and standard deviation where relevant. Compare the performances of "
    "students across different subjects and terms. If possible, suggest reasons for "
    "any trends or patterns observed."
)


def generate_embeddings(document_texts, embeddings):
    return embeddings.embed_documents(document_texts)


def generate_insights(chroma_db, llm, header, first_row):
    header_str = ", ".join(header)
    first_row_str = ", ".join([f"{key}: {value}" for key, value in first_row.items()])

    # question = (
    #     "You are a data analyst. Your task is to analyze the given data and "
    #     "provide detailed insights. Look for trends, patterns, outliers, and any "
    #     "interesting correlations. Consider each aspect of the data. Highlight key findings in bullet points."
    # )
    question = "What are some analytical questions we can ask to gain useful insights? List the questions as 5 bullet points. Consider all attributes of the header row."

    prompt_template_insights = PromptTemplate(
        input_variables=["context"],
        template=f"Given a dataset with the main attributes: {header_str}, and an example data record: {first_row_str}. Using the following context data: {{context}}, answer the given question: {question}.",
        # template="Use this data to accomplish given task: {context}. Task: {question}"
    )

    qa_chain_insights = RetrievalQA.from_chain_type(
        llm,
        retriever=chroma_db.as_retriever(),
        chain_type_kwargs={"prompt": prompt_template_insights},
    )
    result = qa_chain_insights({"query": question})
    print("result", result)
    return result


def generate_charts(chroma_db, llm, df):
    prompt_template_chart = PromptTemplate(
        input_variables=["context"],
        template=f"Given this dataset: {{context}}, please generate Python code for the most suitable and insightful chart visualizations using matplotlib...",
    )
    qa_chain_chart = RetrievalQA.from_chain_type(
        llm,
        retriever=chroma_db.as_retriever(),
        chain_type_kwargs={"prompt": prompt_template_chart},
    )
    charts_result = qa_chain_chart({"query": "Chart"})
    return extract_and_execute_chart_codes(charts_result["result"], df)


def extract_and_execute_chart_codes(full_response, df):
    chart_codes = re.findall(r"```(.*?)```", full_response, re.DOTALL)
    images_base64 = []
    if not chart_codes:
        print("No chart code was generated.")
    else:
        for python_code in chart_codes:
            python_code = (
                python_code.replace("python", "")
                .strip()
                .replace("plt.show()", "# plt.show()")
            )
            globals()["df"] = df
            exec(python_code, globals())
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            images_base64.append(base64.b64encode(buf.read()).decode("utf-8"))
            plt.clf()
    return images_base64


def generate_infographics_template(insights, charts_base64, llm):
    prompt_template_tailwind = PromptTemplate(
        input_variables=["insights", "charts_base64"],
        template="Given these insights: {insights} and chart images encoded in base64: {charts_base64}, create a JSON object...",
    )
    tailwind_chain = LLMChain(
        llm=llm, prompt=prompt_template_tailwind, output_key="tailwind_json"
    )
    return tailwind_chain({"insights": insights, "charts_base64": charts_base64})


# def transform_row_to_sentence(row):

#     return f"{row['Student_Name']} in {row['Subject']} scored {row['Marks_out_of_100']} out of 100 in Term {row['Term']}."


def apply_transformation_template(row, transformation_function_name):
    # Convert the DataFrame row to a dictionary (if not already one)
    row_dict = row if isinstance(row, dict) else row.to_dict()

    # Call the transformation function
    return globals()[transformation_function_name](row_dict)


def load_csv_with_pandas(csv_path):
    return pd.read_csv(csv_path)


def load_and_process_csv(df, transformation_template):
    transformed_documents = [
        apply_transformation_template(row, transformation_template)
        for _, row in df.iterrows()
    ]
    return transformed_documents


def generate_transformation_logic(llm, header, first_row):
    formatted_header = ", ".join(header)
    formatted_first_row = ", ".join(
        [f"{key}: {value}" for key, value in first_row.items()]
    )
    example_sentence = f"{first_row['Student_Name']} in {first_row['Subject']} scored {first_row['Marks_out_of_100']} out of 100 in Term {first_row['Term']}."

    # Create a prompt template
    template_str = (
        "Given a CSV dataset with the following header: {formatted_header}, "
        "and its first row as a dictionary: {formatted_first_row}, "
        "write a Python function named 'transform_row_to_sentence' that takes a row dictionary as input "
        "and returns a sentence. The function should dynamically use the column names and their corresponding values. "
        "For example, the function should look like:\n\n"
        "def transform_row_to_sentence(row):\n"
        "    # Your code here\n\n"
        "And if a row is for example: {formatted_first_row}, the function should return: "
        "'{example_sentence}'. AVOID code explanations or descriptions. Directly give the code."
    )

    prompt_template = PromptTemplate.from_template(template_str)

    # Generate the prompt
    prompt = prompt_template.format(
        formatted_header=formatted_header,
        formatted_first_row=formatted_first_row,
        example_sentence=example_sentence,
    )

    # Generate the response from the LLM
    response = llm.invoke(prompt)
    print("response: ", response.content)
    # Extract the function definition from the response
    function_code = response.content

    # Define the function in the script
    exec(function_code, globals())

    # Return the function name (assuming the function is named 'transform_row_to_sentence')
    return "transform_row_to_sentence"


# response:  content='def transform_row_to_sentence(row):\n    sentence = f"{row[\'Student_Name\']} in {row[\'Subject\']} scored {row[\'Marks_out_of_100\']} out of 100 in Term {row[\'Term\']}."\n    return sentence'
def save_transformed_sentences_to_file(transformed_sentences, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        for sentence in transformed_sentences:
            file.write(sentence + "\n")


def load_text_as_documents(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    return documents


persist_directory_gpt3 = "./chroma_db_gpt3_sentences"


def main():
    csv_path = "./csvs/Marks.csv"
    df = load_csv_with_pandas(csv_path)

    # Extracting header and first row
    header = df.columns.tolist()
    first_row = df.iloc[0].to_dict()
    # load from disk
    chroma_db = Chroma(
        persist_directory=persist_directory_gpt3, embedding_function=embeddings
    )

    insights_result = generate_insights(chroma_db, llm, header, first_row)

    print(insights_result)


if __name__ == "__main__":
    main()
# df = pd.read_csv(csv_path)
# charts_base64 = generate_charts(chroma_db, llm, df)
# tailwind_json = generate_infographics_template(insights_result["result"], charts_base64, llm)


# Generate transformation logic and get the function name
# transformation_function_name = generate_transformation_logic(llm, header, first_row)

# Load and process CSV with the transformation function
# transformed_sentences = load_and_process_csv(df, transformation_function_name)

# Save transformed sentences to a text file
# text_file_path = 'transformed_sentences.txt'
# save_transformed_sentences_to_file(transformed_sentences, text_file_path)

# Load the text file as documents
# transformed_documents = load_text_as_documents(text_file_path)

# Use transformed_documents with Chroma
# chroma_db = Chroma.from_documents(transformed_documents, embeddings, persist_directory=persist_directory_gpt3)
# chroma_db.persist()
