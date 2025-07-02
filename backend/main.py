import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment
from dotenv import load_dotenv
load_dotenv()

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    with open(f"./{file.filename}", "wb") as f:
        f.write(contents)

    loader = PyPDFLoader(file.filename)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local("faiss_index")

    return {"message": "Index built and saved."}

@app.get("/ask")
async def ask(question: str):
    embeddings = OpenAIEmbeddings()
    store = FAISS.load_local("faiss_index", embeddings)

    llm = OpenAI(temperature=0)
    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=store.as_retriever())
    answer = qa.run(question)
    return {"question": question, "answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
