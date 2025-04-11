import os

os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="policy_acts",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

for filename in os.listdir("data"):
    if filename.endswith(".pdf"):  # filter PDFs
        filepath = os.path.join("data", filename)
        print("Processing:", filepath)
        loader = PyPDFLoader(filepath)

        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)
        vector_store.add_documents(documents=all_splits)