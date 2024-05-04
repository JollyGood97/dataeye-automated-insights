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

langchain.debug = True

langchain.verbose = True

key = "YOURKEYHERE"
# Create embeddings
llm = ChatOpenAI(
    openai_api_key=key,
    model="gpt-3.5-turbo",
)

csv_path = "./csvs/Marks.csv"
persist_directory_gpt3 = "./chroma_db_gpt3_textloader"

# if __name__ == "__main__":
