from langchain_community.document_loaders import Docx2txtLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import pickle
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv()

DB_PATH = "faiss_index"

def load_files(folder_path):
    docs = []
    print(f"Loading documents from {folder_path} ...")
    for file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file)
        if file.lower().endswith(".pdf"):
            loader = PyPDFLoader(full_path)
            docs.extend(loader.load())
        elif file.lower().endswith(".txt"):
            loader = TextLoader(full_path, encoding="utf-8")
            docs.extend(loader.load())
        elif file.lower().endswith(".docx"):
            loader = Docx2txtLoader(full_path)
            docs.extend(loader.load())
    print(f"Total documents loaded: {len(docs)}")
    return docs

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    print("Splitting documents into chunks...")
    chunks = splitter.split_documents(docs)
    print(f"Total chunks created: {len(chunks)}")
    return chunks

def save_faiss(vectordb, path):
    vectordb.save_local(path)
    print(f"FAISS index saved at {path}")

def load_faiss(path, embeddings):
    print(f"Loading FAISS index from {path} ...")
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

def create_or_load_vectordb(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(DB_PATH) and os.listdir(DB_PATH) and chunks is None:
        vectordb = load_faiss(DB_PATH, embeddings)
    else:
        print("Creating new FAISS index from documents...")
        vectordb = FAISS.from_documents(chunks, embeddings)
        save_faiss(vectordb, DB_PATH)
    return vectordb

def chatbot_chain(vectordb):
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="deepseek-r1-distill-llama-70b",
        temperature=0,
        max_tokens=None,
        reasoning_format="parsed",
        timeout=None,
        max_retries=2
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful assistant. Answer the question ONLY based on the provided context.
If the answer is not contained within the context, say: "Sorry, I don't know the answer to that."
Context:
{context}
Question:
{question}

and explain everything and every seections in a clearn and detailed manner aside from the answer.
"""
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )
    return chain

if __name__ == "__main__":
    folder_path = os.path.abspath(r"C:\Users\iaksh\Desktop\ml workspacec\RAG CHATBOT\downloads")  # Change path if needed

    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        vectordb = create_or_load_vectordb(None)
    else:
        documents = load_files(folder_path)
        document_chunks = chunk_documents(documents)
        vectordb = create_or_load_vectordb(document_chunks)

    chain = chatbot_chain(vectordb)

    while True:
        user_question = input("Bhai kya kaam hai tereko?.. (type 'exit' to quit) ")
        if user_question.strip().lower() == "exit":
            print("Chalo bhai, bye!")
            break

        response = chain.invoke({"query": user_question})
        print("\nAnswer:\n", response["result"], "\n")
