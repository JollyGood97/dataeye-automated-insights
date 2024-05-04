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

key = "YOURKEYHERE"
# Create embeddings
llm = ChatOpenAI(openai_api_key=key, model="gpt-4-turbo-preview", temperature=0)

with open("insights_netflix.json", "r", encoding="utf-8") as file:
    json_data = json.load(file)


class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Insights Report", 0, 1, "C")

    def chapter_title(self, title):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, 0, 1, "L")
        self.ln(10)

    def chapter_body(self, body):
        self.set_font("Arial", "", 12)
        self.multi_cell(0, 10, body)
        self.ln()


def generate_matplotlib_code(insight_info):
    prompt = """
    Generate Python code using Matplotlib to create a chart based on the given data object:

    Data:
   {insight_info}

    The code should:
    - Import all necessary libraries.
    - Use the data provided to plot the chart. You must go through the entire given data to figure out the values for the chart.
    NEVER hallucinate data. Only use the data provided.
    - Label axes appropriately.
    - Set the title of the chart.
    - Use the provided colors for the chart elements. If colors are not provided, you may define suitable colors.
    - Make sure the code can be executed immediately without the need for any undefined variables.
      Do NOT include plt.show() at the end. Please ensure the code is ready to run and
        effectively communicates the insight based on the provided chart description
s
    """
    prompt_template = PromptTemplate.from_template(prompt)
    formatted_prompt = prompt_template.format(insight_info=insight_info)
    response = llm.invoke(formatted_prompt)
    #
    print(type(response))
    print(response)
    return response.content


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


# Generate PDF
pdf = PDF()
pdf.add_page()

for index, insight in enumerate(json_data):
    # Add insight question
    pdf.chapter_title(insight["insight_question"])

    # Add summary and assumption
    pdf.chapter_body(insight["summary"])
    if insight["assumption"]:
        pdf.chapter_body("Assumption: " + insight["assumption"])

    # Generate and save the chart

    insight_string = json.dumps(insight).strip("{}")

    # print(insight_string)

    matplotlib_code = generate_matplotlib_code(insight_string)
    # Generate the Matplotlib code for the current insight and chart type
    matplotlib_code = (
        matplotlib_code.replace("```python", "")
        .replace("```", "")
        .replace("plt.show()", "")  # Add this line to remove plt.show()
        .strip()
    )
    image = execute_and_capture_chart(matplotlib_code)
    temp_chart_filename = f"temp_chart_{index}.png"
    image.save(temp_chart_filename)

    # Add chart reasoning
    # Check if visualization_chart is a list or a dictionary
    visualization_info = insight["visualization_chart"]
    reasoning_texts = []  # Initialize an empty list to hold reasoning text(s)

    # If it's a dictionary, wrap it in a list for consistent processing
    if isinstance(visualization_info, dict):
        visualization_info = [visualization_info]

    # Iterate through the list (whether originally a list or now a list of one)
    for viz in visualization_info:
        reasoning_texts.append(viz.get("reasoning", "No reasoning provided."))

    # Join all reasoning texts separated by a newline (or choose another separator as needed)
    reasoning_text = "\n".join(reasoning_texts)

    # Add chart reasoning to the PDF
    pdf.chapter_body(f"Reasoning for suggested chart type: {reasoning_text}")

    # Insert the chart image
    pdf.image(temp_chart_filename, x=None, y=None, w=180, h=100)

    pdf.add_page()

pdf.output("Insights_Report.pdf")
