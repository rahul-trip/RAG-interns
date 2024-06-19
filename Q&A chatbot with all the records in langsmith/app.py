from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
import streamlit as st 
import os

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")
chat = model.start_chat(history=[])

def get_gemini_response(questions):
    response = chat.send_message(questions,stream=True)
    return response


st.set_page_config("Q&A Chatbot with history")
st.header("LLM application with gemini")

if "chat_history" not in st.session_state:
    st.session_state['chat_history'] =[]


input = st.text_input("Input",key = "input")
submit = st.button("Ask the question")

if submit and input:
    response = get_gemini_response(input)
    ## Add query and response to session chat history 
    st.session_state['chat_history'].append(("you",input))
    st.subheader("The Response is :")
    for chunk in response:
        st.write(chunk.text)
        st.session_state['chat_history'].append(("BOT",chunk.text))
st.subheader("The chat history is :")

for role , text in st.session_state['chat_history']:
    st.write(f"{role}:{text}")
