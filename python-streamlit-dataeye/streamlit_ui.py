import streamlit as st
import os
from data_prepper import DataPrepper
import tempfile
import traceback

# Provide your OpenAI API key here
openai_api_key = "YOURKEYHERE"


def main():
    st.title("CSV Import for Insight Generation")

    # Initialize DataPrepper with the OpenAI API key
    data_prepper = DataPrepper(openai_api_key=openai_api_key)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            # Process the CSV file and store the embeddings
            with st.spinner("Processing data and storing embeddings..."):
                data_prepper.preprocess_and_store(tmp_file_path)
                st.success("Data processing complete and embeddings stored!")
        except Exception as e:
            st.error("An error occurred:")
            st.text(traceback.format_exc())  # This will print the full traceback
        finally:
            # Delete the temporary file
            os.remove(tmp_file_path)


if __name__ == "__main__":
    main()
