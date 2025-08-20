from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
    print("Bhai baat saamjh ni chalra ye")
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv()
db = "vectordb"
def load_files(file_path):
    docs = []
    for file in os.listdir(file_path):
        full_path = os.path.join(file_path, file)
        if file.endswith('.pdf'):
            loader = PyPDFLoader(full_path)
            docs.extend(loader.load())

        elif file.lower().endswith('.txt'):
            loader = TextLoader(full_path, encoding="utf-8")
            docs.extend(loader.load())
        
        elif file.lower().endswith('.docx'):
            loader = Docx2txtLoader(full_path)
            docs.extend(loader.load())
    print("loadhogs")

    return docs

def chunks(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    print("Chunks hogis")
    return splitter.split_documents(docs)


def create_or_load_vectordb(chunks):
    print("kuch toh hora hai :) ")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Embeddings hogis")
    if os.path.exists(db) and chunks is None:
        print("Loading existing database...")
        return Chroma(persist_directory=db, embedding_function=embeddings)
    else:
        print("Creating new database...")
        vectordb = Chroma.from_documents(chunks, embedding_function=embeddings, persist_directory=db)
        try:
            vectordb.persist()
        except AttributeError:
            pass
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
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PromptTemplate(
                input_variables=["context", "question"],
                template="""You are a helpful assistant. Answer the question based on the provided context.\n\nContext: {context}\n\nQuestion: {question}\nAnswer 
                and also answer the questions in a short but precise way and make sure to add bullet points for questions that have a list in it or write a short
                 paragraph on it :"""
            )
        }
    )
    
    return chain

if __name__ == "__main__":
    if os.path.exists(db):
        vectordb1 = create_or_load_vectordb(None)
    else:
        #folder_path = input("Enter the folder path containing your documents: ").strip()
        file = "/test1"
        docs = load_files(file)
        chunk_texts = chunks(docs)
        vectordb1 = create_or_load_vectordb(chunk_texts)

    chain = chatbot_chain(vectordb1)
    user= input("Bhai kya kaam hai tereko?.. ")
    response = chain.invoke({"query": f"{user}"})

    print("Answer:", response["result"])
