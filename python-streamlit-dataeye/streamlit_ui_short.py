import streamlit as st
from textloader_exp import generate_insights_and_codes, execute_and_capture_chart

# Generate insights and their Matplotlib codes
insights_and_codes = generate_insights_and_codes()

st.title("Insights and Visualizations")

for insight, code in insights_and_codes:
    st.subheader(f"Insight: {insight}")

    # Generate the chart image
    chart_image = execute_and_capture_chart(code)

    # Display the chart image in Streamlit
    st.image(chart_image, caption=f"Visualization for: {insight}")
