# Loading the environment
from dotenv import load_dotenv
load_dotenv()

# Importing necessary libraries
import streamlit as st
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# Initializing model
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please set a valid GOOGLE_API_KEY environment variable.")

genai.configure(api_key=api_key)
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.4, max_tokens=512)

# Function to get response from Gemini model
def get_gemini_response(question):
    prompt = [
        {"role": "system", "content": "You are an AI assistant who is there to help the user answer all of their questions and help them in their daily life while answering with utmost joy."},
        {"role": "user", "content": question}
    ]
    response = model.invoke(prompt)
    return response.content

# Initializing the Streamlit app
st.set_page_config(page_title="Q&A Chatbot")
st.header("Gemini LLM Application")

# Setting up the GUI
input = st.text_input("Input", key="input")  # Takes input
submit = st.button("Ask The Question")  # Submit button

# After pressing the submit button
if submit:
    if input != "":  # Ensure the input is not empty
        response = get_gemini_response(input)
        st.subheader("The response is :")
        st.write(response)
    else:
        st.error("Please enter a question before submitting.")