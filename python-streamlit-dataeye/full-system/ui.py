import json
import time
import streamlit as st
from sqlcreator import csv_to_sqlite, delete_table, view_all_tables, view_table_schema
from insights_agent import generate_insights
from viz import generate_visualizations, create_pdf_report
from nli import sql_agent
import pandas as pd
import os
import uuid


# Function to generate a unique table name if not provided
def generate_unique_table_name():
    return "table_" + str(uuid.uuid4()).replace("-", "_")


# Title
st.title("Data Eye: Automating Insights & Visualizations")

# Sidebar for deletion and viewing options
st.sidebar.title("Database Actions")
table_name_to_delete = st.sidebar.selectbox(
    "Select a table to delete:", [""] + view_all_tables(return_tables=True)
)
if st.sidebar.button("Delete Table"):
    if table_name_to_delete:
        delete_table(table_name_to_delete)
        st.sidebar.success(f"Table {table_name_to_delete} deleted successfully.")
    else:
        st.sidebar.error("Please select a table to delete.")

if st.sidebar.button("View All Tables"):
    st.sidebar.write(view_all_tables(return_tables=True))

table_name_to_view = st.sidebar.selectbox(
    "Select a table to view schema:", [""] + view_all_tables(return_tables=True)
)
if st.sidebar.button("View Table Schema"):
    if table_name_to_view:
        st.sidebar.write(view_table_schema(table_name_to_view, return_schema=True))
    else:
        st.sidebar.error("Please select a table to view its schema.")

# Main page for CSV upload
st.header("Upload CSV to Database")
uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
table_name = st.text_input("Enter table name (leave blank for auto-generated name):")

if st.button("Upload CSV(s)"):
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)

            # If table name is empty, generate a unique one
            if not table_name:
                table_name = generate_unique_table_name()

            # Convert DataFrame to SQL
            csv_to_sqlite(df, table_name)
            st.success(
                f"CSV '{uploaded_file.name}' uploaded to table '{table_name}' successfully."
            )
    else:
        st.error("Please upload at least one CSV file.")

# Sidebar section for Generate actions
st.sidebar.title("Automate Insights")
dataset_description = st.sidebar.text_area("Enter a description for the dataset:")

if st.sidebar.button("Generate Insights"):
    # Start the timer
    start_time = time.time()
    with st.spinner("Generating insights... Please wait."):
        insights_data = generate_insights(dataset_description)
        # Store the insights JSON in the session state
        st.session_state["insights_data"] = insights_data
        # Stop the timer after the function execution
        end_time = time.time()
    elapsed_time = int(end_time - start_time)
    st.success(f"Insights generated in {elapsed_time} seconds!")

if st.sidebar.button("View Insights"):
    # This button will display the generated insights
    if "insights_data" in st.session_state:
        st.header("Insights")
        for insight in st.session_state["insights_data"]:
            st.subheader(insight["insight_question"])
            for qa in insight["questions_and_answers_from_tool"]:
                st.write(f"Q: {qa['question']}")
                st.write(f"A: {qa['answer']}")
            st.write(f"Summary: {insight['summary']}")
    else:
        st.error("No insights to display. Please generate insights first.")


if st.sidebar.button("Generate Visualizations"):
    if "insights_data" in st.session_state:
        # Start the timer
        start_time = time.time()
        with st.spinner("Generating visualizations... Please wait."):
            charts, insights_text = generate_visualizations(
                st.session_state["insights_data"]
            )
            # Stop the timer after the function execution
            end_time = time.time()
        elapsed_time = int(end_time - start_time)
        st.success(f"Visualizations generated in {elapsed_time} seconds!")

        # Display insights and charts in the UI
        for insight_dict, chart in zip(insights_text, charts):
            st.write(insight_dict["insight_question"])
            st.write(insight_dict["summary"])
            st.write(insight_dict["assumption"])
            st.image(chart, use_column_width=True)

            # Save charts and insights in session state
        st.session_state["charts"] = charts
        st.session_state["insights_text"] = insights_text

        # Button to download PDF report
        if st.button("Download PDF Report"):
            pdf = create_pdf_report(st.session_state["insights_data"], charts)
            st.download_button(
                label="Download PDF",
                data=pdf,
                file_name="Insights_Report.pdf",
                mime="application/pdf",
            )
    else:
        st.error("No insights to visualize. Please generate insights first.")

if st.sidebar.button("View Visualizations"):
    if "charts" in st.session_state and "insights_text" in st.session_state:
        # Display insights and charts in the UI
        for insight_dict, chart in zip(
            st.session_state["insights_text"], st.session_state["charts"]
        ):
            st.write(insight_dict["insight_question"])
            st.write(insight_dict["summary"])
            st.write(insight_dict["assumption"])
            st.image(chart, use_column_width=True)
    else:
        st.error("No visualizations available. Please generate visualizations first.")


st.sidebar.title("Ask a Question")
user_query = st.sidebar.text_input(
    "What questions would you like to know ask your data?"
)
if st.sidebar.button("Submit Query"):
    if user_query:
        with st.spinner("Fetching answer..."):
            # Assuming sql_agent returns a response as a string
            answer = sql_agent(user_query)
            st.text("Answer:")
            st.write(answer)
    else:
        st.sidebar.error("Please enter a question to get an answer.")

if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
