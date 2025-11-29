import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from .agentic_chunking import AgenticChunker

load_dotenv()


class DocumentIngestionPipeline:
    
    def __init__(
        self,
        docs_path: str = "workfiles",
        vector_store_path: str = "data/vector_db",
        collection_name: str = "documents",
        chunk_size: int = 1000
    ):
        self.docs_path = docs_path
        self.vector_store_path = vector_store_path
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        
        # OPTION 1: Agentic Chunking (AI-powered, uses Groq tokens)
        # self.chunker = AgenticChunker(chunk_size=chunk_size)
        
        # OPTION 2: Simple Recursive Chunking (No LLM, unlimited & free!)
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        self.chunker = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
    def load_single_file(self, file_path: str) -> dict:
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == ".txt":
            loader = TextLoader(file_path)
            documents = loader.load()
            content = documents[0].page_content
            
        elif file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            content = "\n\n".join([page.page_content for page in pages])
            
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        return {
            'content': content,
            'metadata': {
                'source': file_path,
                'filename': Path(file_path).name,
                'file_type': file_extension
            }
        }
    
    def create_vector_store(self, chunks: list[dict], unique_id: str = None):
        
        # Prepare texts and metadatas for ChromaDB
        texts = [chunk['content'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        # Add unique_id to all metadata if provided
        if unique_id:
            for metadata in metadatas:
                metadata['unique_id'] = unique_id
        
        # Determine collection name
        collection = f"{self.collection_name}_{unique_id}" if unique_id else self.collection_name
        
        # Create vector store
        vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            persist_directory=self.vector_store_path,
            collection_name=collection,
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        return vectorstore
    
    def ingest_file(self, file_path: str, unique_id: str = None):
        
        # Step 1: Load file
        document = self.load_single_file(file_path)
        
        # Step 2: Chunk with simple text splitting
        text_chunks = self.chunker.split_text(document['content'])
        
        # Add metadata to chunks
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunks.append({
                'content': chunk_text,
                'metadata': {
                    **document['metadata'],
                    'chunk_index': i,
                    'chunk_length': len(chunk_text)
                }
            })
        
        # Step 3: Store
        self.create_vector_store(chunks, unique_id)
    
    def load_existing_vectorstore(self, unique_id: str = None):
        collection = f"{self.collection_name}_{unique_id}" if unique_id else self.collection_name
        
        vectorstore = Chroma(
            persist_directory=self.vector_store_path,
            embedding_function=self.embeddings,
            collection_name=collection,
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        count = vectorstore._collection.count()
        
        return vectorstore
