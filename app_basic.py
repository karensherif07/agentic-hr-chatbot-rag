import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

st.title("Basic Chatbot")

llm = ChatGroq(
    model="llama-3.1-8b-instant"
)
user_input = st.chat_input("Ask something")

if user_input:
    st.chat_message("user").write(user_input)

    response = llm.invoke(user_input)

    st.chat_message("assistant").write(response.content)