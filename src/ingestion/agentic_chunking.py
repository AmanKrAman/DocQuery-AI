"""
Agentic Chunking - AI-powered intelligent text splitting
Uses LLM to understand context and split at natural boundaries
"""

from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()


class AgenticChunker:
    def __init__(self, model_name="llama-3.3-70b-versatile", chunk_size=500):
       
        self.llm = ChatGroq(
            model=model_name,
            temperature=0, 
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        self.chunk_size = chunk_size
        
    def chunk_text(self, text: str) -> list[str]:
        print(f"🤖 Using AI to chunk text ({len(text)} chars)...")
        
        # If text is too large (>10000 chars), pre-split by paragraphs
        max_size = 10000
        if len(text) > max_size:
            print(f"⚠️ Text too large, pre-splitting into sections...")
            # Split by double newlines (paragraphs)
            paragraphs = text.split('\n\n')
            
            # Group paragraphs into ~10k char sections
            sections = []
            current_section = ""
            
            for para in paragraphs:
                if len(current_section) + len(para) < max_size:
                    current_section += para + "\n\n"
                else:
                    if current_section:
                        sections.append(current_section.strip())
                    current_section = para + "\n\n"
            
            if current_section:
                sections.append(current_section.strip())
            
            print(f"📄 Split into {len(sections)} sections for AI processing")
            
            # Process each section with AI
            all_chunks = []
            for i, section in enumerate(sections):
                print(f"   Processing section {i+1}/{len(sections)}...")
                section_chunks = self._ai_chunk_section(section)
                all_chunks.extend(section_chunks)
            
            print(f"✅ Created {len(all_chunks)} intelligent chunks")
            return all_chunks
        else:
            # Small enough for direct AI processing
            return self._ai_chunk_section(text)
    
    def _ai_chunk_section(self, text: str) -> list[str]:
        """Chunk a single section using AI"""
        prompt = f"""You are an expert at splitting text into logical chunks.

Rules:
- Each chunk should be around {self.chunk_size} characters (flexible)
- Split at natural topic boundaries (paragraphs, sections, topics)
- Keep related information together
- Don't break sentences mid-way
- Put "<<<SPLIT>>>" marker between chunks

Text to chunk:
{text}

Return the EXACT same text with <<<SPLIT>>> markers where you want to split.
Don't add any extra commentary, just the text with markers.
"""
        
        response = self.llm.invoke(prompt)
        marked_text = response.content
        
        chunks = marked_text.split("<<<SPLIT>>>")
        
        clean_chunks = []
        for chunk in chunks:
            cleaned = chunk.strip()
            if cleaned and len(cleaned) > 20:  
                clean_chunks.append(cleaned)
        
        return clean_chunks
    
    def chunk_document(self, document_content: str, metadata: dict = None) -> list[dict]:
        chunks = self.chunk_text(document_content)
        
        result = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                'content': chunk,
                'metadata': {
                    **(metadata or {}),
                    'chunk_index': i,
                    'chunk_length': len(chunk)
                }
            }
            result.append(chunk_data)
        
        return result
