import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from .agentic_chunking import AgenticChunker
import fitz  
import pdfplumber
import pytesseract
from PIL import Image
import io
import cv2
import numpy as np

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
        
    def _extract_tables(self, file_path: str) -> str:
        
        table_content = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    if tables:
                        for table_idx, table in enumerate(tables):
                            if table and len(table) > 0:
                                headers = table[0]
                                rows = table[1:]
                                table_md = f"\n**Table {table_idx + 1} on Page {page_num + 1}:**\n"
                                if headers:
                                    table_md += "| " + " | ".join(str(h or "") for h in headers) + " |\n"
                                    table_md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                                for row in rows:
                                    table_md += "| " + " | ".join(str(cell or "") for cell in row) + " |\n"
                                table_content += table_md + "\n"
        except Exception as e:
            print(f"Warning: Could not extract tables: {e}")
        return table_content
    
    def _extract_images_with_ocr(self, file_path: str) -> str:
        image_descriptions = ""
        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                images = page.get_images(full=True)
                if images:
                    for img_idx, img in enumerate(images):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        if base_image:
                            image_bytes = base_image["image"]
                            image = Image.open(io.BytesIO(image_bytes))
                            
                            description = f"Image {img_idx + 1} on page {page_num + 1}: Visual element (chart, diagram, or photo)."
                            
                            try:
                                ocr_text = pytesseract.image_to_string(image).strip()
                                if ocr_text:
                                    description += f" OCR Text: {ocr_text}"
                            except Exception:
                                pass
                            
                            image_descriptions += f"\n{description}\n"
            doc.close()
        except Exception as e:
            print(f"Warning: Could not extract images: {e}")
        return image_descriptions
    

    def _load_image_file(self, file_path: str) -> dict:
        try:
            pil_img = Image.open(file_path).convert("RGB")

            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            h, w = gray.shape
            if min(h, w) < 1000:
                gray = cv2.resize(gray, None, fx=2.0, fy=2.0,
                                interpolation=cv2.INTER_CUBIC)

            gray = cv2.medianBlur(gray, 3)
            _, thresh = cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            preprocessed = Image.fromarray(thresh)

            config = "--psm 4"  

            ocr_text = ""
            try:
                ocr_text = pytesseract.image_to_string(preprocessed, config=config)
                ocr_text = ocr_text.strip()
            except Exception as e:
                print(f"Warning: OCR failed for image {file_path}: {e}")

            description = (
                f"Image file: {Path(file_path).name}. "
                f"This is a user-uploaded invoice or document image."
            )

            if ocr_text:
                content = description + "\n\nOCR Text:\n" + ocr_text
            else:
                content = description + "\n\nOCR Text: <none detected>"

            return {
                "content": content,
                "metadata": {
                    "source": file_path,
                    "filename": Path(file_path).name,
                    "file_type": Path(file_path).suffix.lower(),
                    "has_tables": False,
                    "has_images": True,
                },
            }

        except Exception as e:
            raise ValueError(f"Failed to load image file {file_path}: {e}")

    def load_single_file(self, file_path: str) -> dict:
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == ".txt":
            loader = TextLoader(file_path)
            documents = loader.load()
            content = documents[0].page_content
            return {
                'content': content,
                'metadata': {
                    'source': file_path,
                    'filename': Path(file_path).name,
                    'file_type': file_extension
                }
            }
            
        elif file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            content = "\n\n".join([page.page_content for page in pages])
            
            table_content = self._extract_tables(file_path)
            image_descriptions = self._extract_images_with_ocr(file_path)
            
            full_content = content
            if table_content:
                full_content += "\n\n--- TABLES ---\n" + table_content
            if image_descriptions:
                full_content += "\n\n--- IMAGES ---\n" + image_descriptions
            
            return {
                'content': full_content,
                'metadata': {
                    'source': file_path,
                    'filename': Path(file_path).name,
                    'file_type': file_extension,
                    'has_tables': bool(table_content.strip()),
                    'has_images': bool(image_descriptions.strip())
                }
            }
        
        elif file_extension in [".jpg", ".jpeg", ".png"]:
            return self._load_image_file(file_path)
            
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    def create_vector_store(self, chunks: list[dict], unique_id: str = None):
        
        texts = [chunk['content'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        if unique_id:
            for metadata in metadatas:
                metadata['unique_id'] = unique_id
        
        collection = f"{self.collection_name}_{unique_id}" if unique_id else self.collection_name
        
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
        document = self.load_single_file(file_path)
        text_chunks = self.chunker.split_text(document['content'])
        
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
