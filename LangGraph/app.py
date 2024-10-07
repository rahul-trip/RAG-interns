import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langchain.agents import create_structured_chat_agent, AgentExecutor, tool
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import PDFMinerLoader , CSVLoader
import tempfile
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import uuid
import chromadb
from langchain_core.prompts import PromptTemplate
from PIL import Image
import numpy as np
# Load environment variables
load_dotenv()
# Get API keys from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# Configure the Google API key
genai.configure(api_key=GOOGLE_API_KEY)
# Define chat prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Respond to the human as helpfully and accurately as possible. You have access to the following tools:\n\n{tools}\n\n in which there is tool to have a general conversation if it is not able to answer the question you can utilize all the other tools to find the answer ,and one of them uses vectorstore to searh the document.Use a JSON blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).\n\nValid 'action' values: 'Final Answer' or {tool_names}\n\nProvide only ONE action per $JSON_BLOB, as shown:\n\n```\n{{\n  \"action\": $TOOL_NAME,\n  \"action_input\": $INPUT\n}}\n```\n\nFollow this format:\n\nQuestion: input question to answer\nThought: consider previous and subsequent steps\nAction:\n```\n$JSON_BLOB\n```\nObservation: action result\n... (repeat Thought/Action/Observation N times)\nThought: I know what to respond\nAction:\n```\n{{\n  \"action\": \"Final Answer\",\n  \"action_input\": \"Final response to human\"\n}}\n\nBegin! Reminder to ALWAYS respond with a valid JSON blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:\n```$JSON_BLOB```\nthen Observation"),
    ("placeholder", "{chat_history}"),
    ("human", "{input}\n\n{agent_scratchpad}\n(reminder to respond in a JSON blob no matter what)")
])
img_model = genai.GenerativeModel("gemini-1.5-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma("Documents",embedding_function=embeddings,persist_directory="./document_db")
db_path = "./img_db"
embedding_function = OpenCLIPEmbeddingFunction()
data_loader = ImageLoader()
client = chromadb.PersistentClient(path=db_path)
collection = client.get_or_create_collection(
    name='img_data',
    embedding_function=embedding_function,
    data_loader=data_loader)  
@tool
def tavily_search(query):
    """Fetches search results from Tavily based on the query input."""
    from langchain_community.tools.tavily_search import TavilySearchResults
    tavily_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=2)
    return tavily_tool.invoke(query["title"])
@tool
def vectorstore_search(query):
    """Fetches search result from vectorstore based on query input."""
    retriver = vectorstore._similarity_search_with_relevance_scores(query=query["title"])
    return retriver
@tool
def image_chat(query):
    """Chats with the given image embedding from the image store."""
    # Perform the image query, expecting embeddings stored in 'data'
    result = collection.query(query_texts=[query["title"]], include=['data'], n_results=1)
    # Extract and process the image embeddings
    processed_images = []
    for img_array in result['data'][0]: 
        img = Image.fromarray(np.uint8(img_array))# Convert the NumPy array to a PIL image  
        processed_images.append(img)  # Store the image
    # Define a prompt for interacting with the image embedding
    prompt_template_for_img = PromptTemplate(
        template="""You are a helpful AI assistant / image chatbot with extensive knowledge
        who helps the user to answer their queries using the image  as reference.
        Answer:"""
    )
    formatted_prompt = prompt_template_for_img.format()
    if processed_images:
        response = img_model.generate_content([query["title"], processed_images[0], formatted_prompt])
    else:
        response = "No images found in the result."
    return response.text
# Initialize embeddings and LLM
llm = ChatGoogleGenerativeAI(
    api_key=GOOGLE_API_KEY,
    model="gemini-1.5-flash",
    max_retries=5,
)
# Define tools list using the @tool decorated function directly
tools = [tavily_search,vectorstore_search,image_chat]
# Create the tool-calling agent using the direct @tool integration
agent = create_structured_chat_agent(llm, tools, prompt)
# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=5)
# Function to handle search queries using the agent executor
def search_the_web_or_vector_store(query):
    # Retrieve the chat history from session state
    chat_history = st.session_state.chat_history
    # Prepare the chat history for the agent executor
    agent_chat_history = []
    for entry in chat_history:
        if 'human' in entry:
            agent_chat_history.append(HumanMessage(content=entry['human']))
        elif 'ai' in entry:
            agent_chat_history.append(AIMessage(content=entry['ai']))
    # Call the agent executor with the current query and the chat history
    result = agent_executor.invoke({
        "input": query,
        "chat_history": agent_chat_history  
        # Pass the chat history to the agent
    })
    return result['output']
#General Chat
def general_chat(query):
        chat_history = st.session_state.chat_history
        prompt="""You are a helpful AI assistant with extensive knowledge
            who helps the user to answer their queries and keeping chat history as refrence : {chat_history}
            if you dont know the answer you suggest the user to utilize vector / web search button."""
        formatted_prompt = prompt.format(chat_history=chat_history)
        response = img_model.generate_content([query["title"],formatted_prompt])
        return response.text
#Function to handle submit
def handle_submit(user_input):
    if user_input:
        # Add human message to the session state
        st.session_state.chat_history.append({'human': user_input})
        # Get the bot's response using search_the_web
        bot_output = search_the_web_or_vector_store(query= user_input)
        # Add AI response to the session state
        st.session_state.chat_history.append({'ai': bot_output})
#Function to handle chat
def handle_chat(user_input):
    if user_input:
        # Add human message to the session state
        st.session_state.chat_history.append({'human': user_input})
        # Get the bot's response using search_the_web
        bot_output = general_chat(query= user_input)
        # Add AI response to the session state
        st.session_state.chat_history.append({'ai': bot_output})
#Function to load pdf
def pdf_loader(files):
    loader = PDFMinerLoader(files)
    docs = loader.load()
    return docs[0]
#Function to load csv
def csv_loader(files):
    loader = CSVLoader(files)
    docs = loader.load()
    return docs[0]
#Function to generate unique id
def generate_unique_int_id(existing_ids):
    """Generates a random, unique integer ID based on UUID."""
    while True:
        new_id = uuid.uuid4().int >> 64
        if new_id not in existing_ids:
            return new_id
#Function to load image
def img_loader(files):
            existing_ids = set()
            unique_id = str(generate_unique_int_id(existing_ids))
            return collection.add(ids=[unique_id],uris=[files])
# Set up Streamlit app configuration
st.set_page_config(page_title="Agent with History", layout="wide")
# Streamlit UI Title with Header Styling
st.markdown("<h1 style='text-align: center; color: #4B72E0;'>Intelligent Chatbot with File Upload</h1>", unsafe_allow_html=True)
st.write("---")  # Separator
# Sidebar for File Upload
st.sidebar.header("Upload Documents/Images")
files = st.sidebar.file_uploader("Upload your files here", type=['pdf', 'jpg', 'jpeg', 'png', 'csv'], accept_multiple_files=True)
# Progress bar
if files:
    upload_progress = st.sidebar.progress(0)
# Initialize chat history and document in session state if not present
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
# Function to handle file upload and save files
def save_file_by_type(files):
    progress_val = 0
    for i, file in enumerate(files):
        if file.type == "application/pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_file_path = tmp_file.name
                pdf_data = pdf_loader(tmp_file_path)
                vectorstore.add_documents([pdf_data])
        elif file.type in ["image/jpeg", "image/png"]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg" if file.type == "image/jpeg" else ".png") as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_file_path = tmp_file.name
                img_loader(tmp_file_path)
        elif file.type == "text/csv":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_file_path = tmp_file.name
                csv_data = csv_loader(tmp_file_path)
                vectorstore.add_documents([csv_data])
        else:
            st.warning("Unsupported file format.")
        progress_val = (i + 1) / len(files)
        upload_progress.progress(progress_val)  # Update progress bar
    upload_progress.empty()  # Clear progress bar when done
    st.sidebar.success("Files uploaded and processed successfully!")
def display_chat():
    st.markdown("<h3 style='color: #4B72E0;'>Chat History</h3>", unsafe_allow_html=True)
    for message in st.session_state['chat_history']:
        if 'human' in message:
            st.markdown(f"<div style='background-color: #DCF8C6; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>"
                        f"<strong>You:</strong> {message['human']}</div>", unsafe_allow_html=True)
        elif 'ai' in message:
            st.markdown(f"<div style='background-color: #E4E6EB; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>"
                        f"<strong>Bot:</strong> {message['ai']}</div>", unsafe_allow_html=True)
# Save uploaded files
if files:
    save_file_by_type(files)
# Form for user input and message submission
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your message:")
    col1,col2 = st.columns([1,10])
    with col1:
        submit_button = st.form_submit_button(label="Chat")
        if submit_button:
            if user_input:
                handle_chat(user_input)
                display_chat()
    with col2:
        search_button = st.form_submit_button(label="Search VectorDB/Web", help="Click to send your query")
        if search_button:
            if user_input:
                handle_submit(user_input)
                display_chat()          
# Clear chat button
if st.button("Clear Chat"):
    st.session_state['chat_history'].clear()