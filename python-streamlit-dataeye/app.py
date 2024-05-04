from flask import Flask, request, jsonify
import requests
import io
import base64
import re
import matplotlib.pyplot as plt
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route("/")
def hello_world():
    return "Hello, World!"
# write some code in python using matplotlib.pyplot to create a simple graph


@app.route("/post-endpoint", methods=["POST"])
def handle_post():
    data = request.json
   
    print(data)
    # properties = dir(request)
    # for prop in properties:
    #     print(prop)

    return "Data received", 200  

@app.route("/chat", methods=["POST"])
def chat():

    data_from_frontend = request.json
    prompt = data_from_frontend.get("prompt")

    ollama_data = {
        "model": "llama2",
        "prompt": prompt,
        "stream": False  # Set to False for a single response
    }

  
    ollama_url = "http://127.0.0.1:11434/api/generate"

    try:
  
        ollama_response = requests.post(ollama_url, json=ollama_data)

   
        if ollama_response.status_code == 200:
            response_data = ollama_response.json()
            model_response = response_data.get("response")
            return jsonify({"response": model_response})

        else:
            return jsonify({"error": "Error communicating with Ollama API"}), 500

    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 500

@app.route("/generate-vis", methods=["POST"])
def vis():
    data_from_frontend = request.json
    prompt = data_from_frontend.get("prompt")

    llama_data = {
        "model": "llama2",
        "prompt": prompt,
        "stream": False
    }

    llama_url = "http://127.0.0.1:11434/api/generate"

    try:
        llama_response = requests.post(llama_url, json=llama_data)

        if llama_response.status_code == 200:
            response_data = llama_response.json()
            full_response = response_data.get("response")

      
            # Code block is enclosed in triple backticks
            python_code = re.search(r"```(.*?)```", full_response, re.DOTALL).group(1).strip()
            python_code = python_code.replace("plt.show()", "# plt.show()")

            exec(python_code)

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')

            return jsonify({"image": image_base64})

        else:
            return jsonify({"error": "Error communicating with LLM API"}), 500

    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(debug=True)
