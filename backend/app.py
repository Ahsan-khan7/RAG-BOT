import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings # Using HuggingFace for open-source embeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceHub # For a free LLM, or use OpenAI/Cohere
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.schema import AIMessage, HumanMessage

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app) # Enable CORS for frontend communication

# --- Configuration ---
UPLOAD_FOLDER = 'uploaded_pdfs'
VECTOR_DB_DIR = 'vector_store'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# HuggingFace API Token (replace with your actual token or use OpenAI/other)
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HUGGINGFACEHUB_API_TOKEN:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not set. Please set it in .env file.")

# LLM setup (Using a suitable open-source model from HuggingFace Hub)
# You might need to experiment with models for best performance and speed.
# Example: 'google/flan-t5-large', 'HuggingFaceH4/zephyr-7b-beta', 'mistralai/Mistral-7B-Instruct-v0.1'
# For demonstration, we'll use a relatively small one for faster inference.
# For production, consider larger, more capable models or self-hosting.
llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.1, "max_length": 512})

# Embedding model (using a local HuggingFace embedding model)
# This downloads the model locally. For better performance, use a strong embedding model.
# 'sentence-transformers/all-MiniLM-L6-v2' is a good balance of size and quality.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Global variables for vector store and conversation chain
vector_store = None
conversation_history = [] # Stores HumanMessage and AIMessage objects for LangChain memory

# --- Helper Functions ---
def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def get_vector_store(chunks):
    global vector_store
    if vector_store is None:
        # Create a new Chroma vector store from the documents
        vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=VECTOR_DB_DIR)
        print(f"Created new vector store with {len(chunks)} chunks.")
    else:
        # Add new documents to existing vector store (more complex, requires Chroma's add_documents)
        # For simplicity here, if a new PDF is uploaded, we'll recreate the store.
        # In a real-world app, you'd manage multiple document sets or add to an existing one.
        vector_store.add_documents(chunks)
        print(f"Added {len(chunks)} chunks to existing vector store.")
    return vector_store

def get_rag_chain():
    global vector_store, conversation_history
    if vector_store is None:
        return None # No documents loaded yet

    # Define the conversational retriever
    retriever = vector_store.as_retriever()

    # Prompt for history-aware retrieval
    # This prompt helps the LLM decide if it needs to rephrase the query given chat history.
    history_aware_retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation. Only return the search query and nothing else.")
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, history_aware_retriever_prompt)

    # Prompt for answer generation
    # This prompt provides the context and chat history to the LLM for generating the final answer.
    rag_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("system", "Given the above conversation and the following retrieved context, answer the user's question. If you don't know the answer, say 'I don't have enough information to answer that.'.\n\nRetrieved context:\n{context}")
    ])
    
    # Create the RAG chain
    rag_chain = create_retrieval_chain(history_aware_retriever, rag_prompt | llm)
    
    return rag_chain

# --- API Endpoints ---
@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'pdf_file' not in request.files:
        return jsonify({"error": "No PDF file part"}), 400
    
    file = request.files['pdf_file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith('.pdf'):
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        try:
            chunks = process_pdf(filepath)
            get_vector_store(chunks) # This will create or update the vector store
            global conversation_history # Clear history on new document upload
            conversation_history = []
            return jsonify({"message": f"PDF '{file.filename}' uploaded and processed successfully!"}), 200
        except Exception as e:
            return jsonify({"error": f"Error processing PDF: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file type. Please upload a PDF."}), 400

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')

    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    if vector_store is None:
        return jsonify({"error": "No PDF uploaded yet. Please upload a document first."}), 400

    try:
        rag_chain = get_rag_chain()
        
        # Add human message to history
        conversation_history.append(HumanMessage(content=user_message))

        # Invoke the RAG chain with current input and history
        # The 'input' to the chain is the latest user message.
        # The 'chat_history' is passed implicitly to create_history_aware_retriever and rag_prompt.
        response = rag_chain.invoke({"input": user_message, "chat_history": conversation_history})
        
        # The 'response' object from create_retrieval_chain usually has a 'answer' key.
        bot_response = response.get("answer", "I could not generate a response.")
        
        # Add AI message to history
        conversation_history.append(AIMessage(content=bot_response))
        
        return jsonify({"response": bot_response}), 200
    except Exception as e:
        print(f"Error during chat: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/')
def serve_frontend():
    return send_from_directory('../frontend/public', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('../frontend/public', filename)

if __name__ == '__main__':
    # Initialize a dummy vector store if none exists for testing without upload
    # This will prevent errors if you try to chat before uploading
    if not os.path.exists(os.path.join(VECTOR_DB_DIR, "index")):
        print("No existing vector store found. Please upload a PDF to initialize it.")
    
    app.run(debug=True, port=5000)
