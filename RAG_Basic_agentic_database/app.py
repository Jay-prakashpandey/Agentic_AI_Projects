import streamlit as st
from agent import analyst_chat

st.title("BigQuery Agentic Data Analyst 🤖")
st.write("Dataset: `bigquery-public-data.thelook_ecommerce`")

user_q = st.text_input("Ask a question about ecommerce data:")

if user_q:
    with st.spinner("Thinking..."):
        result = analyst_chat(user_q)

    st.subheader("Generated SQL")
    st.code(result["sql"], language="sql")

    if "error" in result:
        st.error(result["error"])
    else:
        st.subheader("Query Results (first 10 rows)")
        st.dataframe(result["data"])

        st.subheader("Summary")
        st.write(result["summary"])