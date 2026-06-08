from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

st.header("Reasearch Tool")

user_input = st.text_input("Ask Amything...")

if st.button("summarize"):
    result = model.invoke(user_input)
    st.write(result.content)