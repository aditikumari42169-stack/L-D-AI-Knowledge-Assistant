import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from config import GOOGLE_API_KEY, PDF_FOLDER, FAISS_INDEX_PATH

def load_documents():
    documents = []

    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, file)

            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            documents.extend(docs)

    print(f"Loaded {len(documents)} pages.")

    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    return chunks

def get_embeddings():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    return embeddings

def create_vector_store(chunks):
    embeddings = get_embeddings()

    vector_db = FAISS.from_documents(
        chunks,
        embeddings
    )

    vector_db.save_local(FAISS_INDEX_PATH)

    print(f"FAISS index saved at: {FAISS_INDEX_PATH}")

def build_index():
    print("Loading documents...")
    documents = load_documents()

    print("Splitting documents...")
    chunks = split_documents(documents)

    print("Creating vector database...")
    create_vector_store(chunks)

    print("Knowledge Base Ready!")

def load_vector_store():
    embeddings = get_embeddings()

    vector_db = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_db

def ask_ai(question):
    vector_db = load_vector_store()

    docs = vector_db.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-lite-latest",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )

    prompt = f"""
You are an AI Knowledge Assistant for the Learning & Development department.

Use ONLY the information provided below.

Knowledge Base:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    # Extract only the text from the AIMessage
    if hasattr(response, "content"):
        print(type(response.content)) 
        content = response.content

        # If content is a list (new Gemini format)
        if isinstance(content, list):
            answer = ""
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    answer += item.get("text", "")
            return answer

        # If content is already a string
        return str(content)

    return str(response)