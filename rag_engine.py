import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import config

class RAGEngine:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
        self.vector_store = None
        self._load_or_create_index()

    def _load_or_create_index(self):
        if os.path.exists(config.FAISS_INDEX_PATH):
            self.vector_store = FAISS.load_local(
                config.FAISS_INDEX_PATH, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        else:
            # Initialize with default empty placeholder
            init_doc = [Document(page_content="Enterprise AI Research Base.", metadata={"source": "system"})]
            self.vector_store = FAISS.from_documents(init_doc, self.embeddings)
            self.vector_store.save_local(config.FAISS_INDEX_PATH)

    def ingest_pdf(self, file_path: str):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(documents)
        
        self.vector_store.add_documents(chunks)
        self.vector_store.save_local(config.FAISS_INDEX_PATH)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        return self.vector_store.similarity_search(query, k=k)