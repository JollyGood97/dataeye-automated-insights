import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from fpdf import FPDF
from langchain_openai import ChatOpenAI

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
from langchain.prompts import PromptTemplate

langchain.debug = True
# model="gpt-3.5-turbo",
langchain.verbose = True

from langsmith.wrappers import wrap_openai
from langsmith import traceable
import os
from dotenv import load_dotenv

load_dotenv()


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Auto-trace LLM calls in-context
client = wrap_openai(openai.Client())


key = "YOURKEYHERE"
# gpt-4-turbo-preview
llm = ChatOpenAI(openai_api_key=key, model="gpt-4-turbo-preview", temperature=0)

# llm = ChatOpenAI(openai_api_key=key, model="gpt-3.5-turbo", temperature=0)

# with open("insights_ganeesha.json", "r", encoding="utf-8") as file:
#     json_data = json.load(file)


@traceable  # Auto-trace this function
def generate_matplotlib_code(insight_info):
    prompt = """
    You are an expert in generating python code.
    Please generate executable Python code using Matplotlib to create a chart based on the given data. 
    The code must be plain Python without markdown, comments, or additional characters that would prevent it 
    from running as is. Here's the data:

    {insight_info}

    The code should:
    - Import all necessary libraries.
    - Use the provided data to plot the chart, without adding any extra data.
    - Label axes appropriately and set the title of the chart.
    - Define suitable colors for the chart elements if colors are not provided.
    - Do not display the chart. Specifically, do not include plt.show() in the code.
    - Ensure the code is complete and can be executed immediately in a Python environment without modifications.
    - Avoid explanations. 
    """
    prompt_template = PromptTemplate.from_template(prompt)
    formatted_prompt = prompt_template.format(insight_info=insight_info)
    response = llm.invoke(formatted_prompt)

    code = (
        response.content.replace("```python", "")
        .replace("```", "")
        .replace("plt.show()", "")
        .strip()
    )
    return code


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


def generate_visualizations(insights_data):
    charts = []
    insights_text = []
    for insight in insights_data:
        insight_string = json.dumps(insight).strip("{}")
        matplotlib_code = generate_matplotlib_code(insight_string)
        chart_image = execute_and_capture_chart(matplotlib_code)
        charts.append(chart_image)
        insights_text.append(
            {
                "insight_question": insight["insight_question"],
                "summary": insight["summary"],
                "assumption": insight["assumption"],
            }
        )

    return charts, insights_text


import tempfile


def create_pdf_report(insights_data, charts):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for insight, chart_image in zip(insights_data, charts):
        pdf.multi_cell(0, 10, insight["insight_question"])
        pdf.multi_cell(0, 10, insight["summary"])
        pdf.multi_cell(0, 10, insight["assumption"])

        # Save the chart to a temporary file
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".png")
        chart_image.save(tmp_path, format="PNG")

        # Add the image to the PDF and then delete the temporary file
        pdf.image(tmp_path, x=10, y=pdf.get_y(), w=180, type="PNG")
        os.close(tmp_fd)
        os.remove(tmp_path)
        pdf.ln(105)

    # Instead of saving the PDF to a file, return the PDF bytes
    pdf_content = pdf.output(dest="S").encode("latin1")
    return pdf_content


# if __name__ == "__main__":
#     charts, insight_text = generate_visualizations(json_data)
#     print("insights", insight_text)
#     print(type(insight_text))
#     # print(charts, insight_text)
#     create_pdf_report(insight_text, charts)
