port numpy as np
import pandas as pd
import os
import json
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from groq.api import Groq
import streamlit as st

pdf_path = "/kaggle/input/ai-assignment/pdf_data.json"
try:
    with open(pdf_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        document_text = data[0].get("text", "") if isinstance(data[0], dict) else ""
    elif isinstance(data, dict):
        document_text = data.get("text", "")
    else:
        raise ValueError("Unexpected JSON structure: expected a list or dict.")
except Exception as e:
    raise FileNotFoundError(f"Error reading JSON file: {e}")

if not document_text.strip():
    raise ValueError("The document text is empty. Please check the file content.")

sections = document_text.split("\n\n")
if not sections:
    raise ValueError("No sections found in the document text. Please check the formatting.")

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
try:
    embeddings_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
except Exception as e:
    raise RuntimeError(f"Error loading embedding model: {e}")

try:
    vector_store = FAISS.from_texts(texts=sections, embedding=embeddings_model)
except Exception as e:
    raise RuntimeError(f"Error creating FAISS vector store: {e}")

prompt = """ You are a chatbot designed to answer questions about my following CV: {retrieved_data}

User Query: {user_query} """

try:
    client = Groq(api_key="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")
except Exception as e:
    raise RuntimeError(f"Error initializing Groq client: {e}")

def process_query(user_query):
    try:
        retrieved_docs = vector_store.similarity_search(user_query, k=3)
        retrieved_data = "\n".join([doc.page_content for doc in retrieved_docs])
        formatted_prompt = prompt.format(retrieved_data=retrieved_data, user_query=user_query)

        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": formatted_prompt}],
            temperature=0.5,
            max_tokens=500,
            top_p=0.85,
            stream=False,
            stop=None,
        )
        response = completion.choices[0].message.content
        return response
    except Exception as e:
        return f"Error processing query: {e}"

st.title("CV Chatbot")
st.write("Use this chatbot to ask questions about my CV.")

user_query = st.text_area("Ask a Question", placeholder="Enter your question about the CV...")
if st.button("Ask"):
    if user_query.strip():
        response = process_query(user_query)
        st.text_area("Response", value=response, height=200, disabled=True)
    else:
        st.error("Please enter a question.")

st.markdown("""<h3>Disclaimer</h3>
The chatbot provides responses based on my CV content. Verify critical details independently.
"", unsafe_allow_html=True)
