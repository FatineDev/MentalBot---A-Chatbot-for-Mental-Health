
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
import os

# Load the environment variables
load_dotenv()  # This function loads environment variables from a .env file located in the same directory as the script
huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load the Hugging Face embedding model
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load the corpus
with open("corpus.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f.readlines()]

# Generate embeddings with Sentence-BERT
embeddings = embedding_model.encode(documents, convert_to_numpy=True)

# Convert to numpy array
embeddings = np.array(embeddings, dtype=np.float32)

# Create a FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Initialize the Hugging Face generation model
llm = HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-alpha",
    model_kwargs={"temperature": 0.7, "max_length": 300},
    huggingfacehub_api_token=huggingfacehub_api_token
)


def search(query, index, documents, k=1):   # k : number of documents
    # Transform the user's query into a vector representation
    query_embedding = embedding_model.encode(query, convert_to_numpy=True).astype(np.float32).reshape(1, -1)

    # FAISS search
    distances, indices = index.search(query_embedding, k)

    # Retrieve the top k documents
    results = [documents[i] for i in indices[0]]
    return results

def generate_answer(query, document):
    prompt = (
        f"Question : {query}\n"
        f"Informations : {document}\n"
        "Give us a proper answer :"
    )
    response = llm.invoke(prompt)
    
    # Extract the answer after "Give us a proper answer:"
    if isinstance(response, str):
        if "Give us a proper answer :" in response:
            return response.split("Give us a proper answer :", 1)[1].strip()
        else:
            return response.strip()
    elif isinstance(response, list) and len(response) > 0:
        return response[0].strip()
    else:
        return "Can u rephrase your request ?"
    

# Streamlit Interface
st.markdown(
    """
    <style>
        h1 {
            background-color: #A9DBF1;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Mental AI-Bot - Your mental Health Assistant")
st.write("I am here to assist you in your mental health. Feel free to ask me whatever you think about !")


# Initialize the message history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display all the historical messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])


# User input field with chat_input
user_query = st.chat_input("Type your request :")


if user_query:
    # Search and generate response
    doc_found = search(user_query, index, documents, k=1)
    response = generate_answer(user_query, doc_found[0])

    # Add the user message
    st.chat_message('user').markdown(user_query) # If I don't add the following line, every time I type text, it gets updated in the conversation!
    # Store the user_query in state
    st.session_state.messages.append({'role':'user', 'content': user_query})
    # Search and generate response
    st.chat_message('assistant').markdown(response)
    st.session_state.messages.append({'role':'assistant', 'content':response})

    # Refresh the page to display the new message
    st.rerun()