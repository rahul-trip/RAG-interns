# Loading the environment
from dotenv import load_dotenv
load_dotenv()

# Importing necessary libraries
import streamlit as st
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
import base64

# Initializing model
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please set a valid GOOGLE_API_KEY environment variable.")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")



def get_gemini_response(input_text, image):
    if input_text != "":
        response = model.generate_content([input_text,image])
    else:
        response = model.generate_content(image)
    
    return response.text

# Initializing the Streamlit app
st.set_page_config(page_title="image chatbot")
st.header("Gemini LIM Application")
input_text = st.text_input("Input", key="input")

upload_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image = None
if upload_file is not None:
    image = Image.open(upload_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

submit = st.button("Tell me about the image")

if submit and image:
    response = get_gemini_response(input_text, image)
    st.subheader("The information about the image is down below:")
    st.write(response)
elif submit:
    st.error("Please upload an image before submitting.")
