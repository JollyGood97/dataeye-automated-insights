import streamlit as st
from interact import interact

import streamlit as st
import openai

# Streamlit UI
st.title("LLM Chat")

# User input
user_input = st.text_input("Type your message here:")

if st.button("Send"):
    if user_input:  # Check if there's user input
        # Get response from the model
        model_response = interact(user_input)

        # Display the response
        st.write("GPT-3.5-turbo:", model_response)
    else:
        st.write("Please enter a message.")
