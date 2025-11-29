# DocQuery AI

AI-powered document search system. Upload documents and get intelligent answers to your questions using advanced retrieval techniques.

## 🎥 Demo Video

<video width="100%" controls>
  <source src="rag_doc_search.webm" type="video/webm">
  Your browser does not support the video tag.
</video>

## Setup

1. **Install dependencies:**
```bash
python -m venv rag_venv
source rag_venv/bin/activate
pip install -r requirements.txt
```

2. **Add API key:**
Create `.env` file:
```
GROQ_API_KEY=your_key_here
```
Get free key at: https://console.groq.com/keys

3. **Run server:**
```bash
python main.py
```
Server: http://localhost:5000  
Docs: http://localhost:5000/docs

## Usage

**Upload file:**
```bash
POST /upload
# Form-data: key="file", type=File
# Returns: unique_id
```

**Ask question:**
```bash
POST /query
{
  "question": "Your question here?",
  "unique_id": "your_unique_id"
}
```

## Tech Stack

- FastAPI, ChromaDB, LangChain
- HuggingFace embeddings (local)
- Groq LLM (free tier)

