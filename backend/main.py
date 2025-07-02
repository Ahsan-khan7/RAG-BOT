import os
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain Imports
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferWindowMemory, CombinedMemory, ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# For production, consider using a more robust vector store like Pinecone, Qdrant, etc.
# For simplicity, we'll use a local ChromaDB instance.

load_dotenv() # Load environment variables from .env file

# --- Configuration ---
VECTOR_DB_DIR = "./chroma_db"
DOCUMENT_DIR = "./documents" # Directory where your source documents will be
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
K_RETRIEVE = 4 # Number of relevant chunks to retrieve
MEMORY_WINDOW_SIZE = 5 # Number of recent turns to keep in short-term memory
LLM_MODEL_NAME = "gpt-3.5-turbo" # Or "gpt-4", "llama3", etc.
EMBEDDING_MODEL_NAME = "text-embedding-ada-002" # For OpenAI embeddings

# --- Pydantic Models for API Request/Response ---
class DocumentInput(BaseModel):
    file_content: str
    file_name: str

class ChatRequest(BaseModel):
    user_message: str
    session_id: str # To maintain separate conversations for different users

class ChatResponse(BaseModel):
    ai_response: str
    source_documents: List[Dict[str, Any]] # To show sources for RAG

# --- Global Components (Initialized once) ---
app = FastAPI(title="Advanced RAG Chatbot API")

# Store conversation history per session
session_memories: Dict[str, ChatMessageHistory] = {}

# Initialize Embeddings
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

# Initialize LLM
llm = ChatOpenAI(model_name=LLM_MODEL_NAME, temperature=0.7)

# Initialize Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    is_separator_regex=False,
)

# Function to get or create a vector store
def get_vector_store():
    # In a real production system, you'd likely connect to a persistent
    # remote vector database. For this demo, we'll use a local ChromaDB.
    # We could also load from a pre-existing directory if documents were ingested previously.
    if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
        print(f"Loading existing ChromaDB from {VECTOR_DB_DIR}")
        return Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
    else:
        print("Creating new empty ChromaDB (no documents ingested yet).")
        return Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)

vector_store = get_vector_store()
retriever = vector_store.as_retriever(search_kwargs={"k": K_RETRIEVE})

# --- RAG Chain with Memory ---
# Advanced Prompt for RAG with Chat History
_template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
If the follow up question directly relates to the retrieved context (and not past conversation), you can directly use the follow up question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_template(_template)

# Contextualize question based on chat history
# This is a crucial step for memory in RAG, ensuring retrieval is relevant to the ongoing conversation.
_contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
        ("ai", "."),
    ]
)
contextualize_q_chain = CONDENSE_QUESTION_PROMPT | llm # Use the LLM to rephrase

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in session_memories:
        session_memories[session_id] = ChatMessageHistory()
    return session_memories[session_id]


# Main RAG prompt
QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Keep the answer concise and to the point.\n\nRetrieved Context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

# Define the RAG chain
def create_rag_chain_with_memory(session_id: str):
    # Short-term memory (windowed)
    conversational_memory = ConversationBufferWindowMemory(
        k=MEMORY_WINDOW_SIZE,
        memory_key="chat_history",
        return_messages=True,
        chat_memory=get_session_history(session_id)
    )

    # Note: For more advanced long-term memory (summarization/vectorization of history),
    # you would integrate additional components here, potentially
    # a separate chain that processes and stores summaries or
    # uses a vector store for chat history embeddings.

    # This chain handles both conversational memory and retrieval
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=conversational_memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        rephrase_question_llm=llm # Use LLM to rephrase follow-up questions
    )
    return rag_chain


# --- API Endpoints ---

@app.post("/ingest_document/")
async def ingest_document(doc_input: DocumentInput):
    """
    Ingests a document into the vector database.
    In a real app, this might be an admin endpoint or part of a document upload service.
    """
    try:
        # Create a directory to store documents if it doesn't exist
        os.makedirs(DOCUMENT_DIR, exist_ok=True)
        file_path = os.path.join(DOCUMENT_DIR, doc_input.file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(doc_input.file_content)

        # Load and split the document
        # For different file types, you'd use appropriate LangChain document loaders (e.g., PyPDFLoader)
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file_path)
        documents = loader.load()
        texts = text_splitter.split_documents(documents)

        # Add to vector store
        vector_store.add_documents(texts)
        vector_store.persist() # Save the vector store to disk
        print(f"Document '{doc_input.file_name}' ingested successfully.")
        return {"message": f"Document '{doc_input.file_name}' ingested successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {str(e)}")

@app.post("/chat/", response_model=ChatResponse)
async def chat_with_rag(chat_request: ChatRequest):
    """
    Handles user chat messages, retrieves context, and generates a response.
    """
    try:
        session_id = chat_request.session_id
        user_message = chat_request.user_message

        # Ensure a chain with memory is created for the session
        rag_chain = create_rag_chain_with_memory(session_id)

        # Invoke the RAG chain
        # The chain automatically handles adding the user message to memory
        # and retrieving relevant documents based on the rephrased question.
        result = rag_chain.invoke({"question": user_message})

        ai_response = result["answer"]
        source_docs = []
        # Extract source documents (metadata might vary based on your loader/vector store)
        if "source_documents" in result:
            for doc in result["source_documents"]:
                source_docs.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                })

        # LangChain's ConversationalRetrievalChain automatically adds the interaction to memory.
        # You can inspect session_memories[session_id].messages to see the history.

        return ChatResponse(ai_response=ai_response, source_documents=source_docs)
    except Exception as e:
        print(f"Error during chat: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# --- Run the FastAPI App (for development) ---
if __name__ == "__main__":
    import uvicorn
    # To run: uvicorn main:app --reload --port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
