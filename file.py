import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings

def load_files(file_path):
    print("it might take some time please be patient!! ")
    docs = []
    for file in os.listdir(file_path):
        if file.endswith('.pdf'):
            full_path = os.path.join(file_path, file)
            loader = PyPDFLoader(full_path)
            docs.extend(loader.load())
    return docs

def chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return text_splitter.split_documents(docs)

def vectordb(chunks_list):
    if not chunks_list:
        raise ValueError("No document chunks to embed.")
    
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(
        documents=chunks_list,
        embedding=embedding,
        persist_directory="data/db"
    )
    db.persist()
    return db

def chatbot_chain(vectordb):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro",
                                  temperature=0,
                                  Google_KEY = os.getenv("gemini_api_key"))
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return chain


if __name__ == "__main__":
    folder_path = "downloads"
    docs = load_files(folder_path)
    
    # Check if documents were loaded
    if not docs:
        print("No PDF documents found in the data folder")
        exit(1)
    
    print(f"Loaded {len(docs)} documents")
    
    chunk_texts = chunks(docs)
    print(f"Created {len(chunk_texts)} chunks")
    
    # Fixed: Pass chunk_texts instead of docs
    vectordb1 = vectordb(chunk_texts)
    chain = chatbot_chain(vectordb1)
    
    response = chain({"query": "What is the main topic of the document?"})
    print("Answer:", response["result"])
    print("Source Documents:", response["source_documents"])