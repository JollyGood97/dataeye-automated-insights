import streamlit as st
import langchain_helper as lch

st.title("Data Insights & Infographics")

response = lch.get_insights()

if "insights" in response:
    st.write("Insights:", response["insights"])

if "charts" in response:
    for idx, img_base64 in enumerate(response["charts"]):
        img_url = "data:image/png;base64," + img_base64
        st.image(img_url, caption=f"Chart {idx+1}", use_column_width=True)
