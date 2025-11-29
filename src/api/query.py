from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from src.retrieval.advanced_retrieval import DocumentRetriever
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    unique_id: str
    use_advanced: bool = True  


@router.post("/query")
async def query_documents(request: QueryRequest):
    try:
        if not request.question:
            raise HTTPException(status_code=400, detail="Question is required")
        
        if not request.unique_id:
            raise HTTPException(status_code=400, detail="unique_id is required")
        
        retriever = DocumentRetriever()
        
        if request.use_advanced:
            
            results = retriever.advanced_search(
                query=request.question,
                unique_id=request.unique_id,
                top_k=5,
                use_multi_query=False,  
                use_reranker=True
            )
        else:
            results = retriever.vector_search(
                query=request.question,
                unique_id=request.unique_id,
                k=5
            )
        
        # Use smaller/faster model to save tokens
        llm = ChatGroq(
            model="llama-3.1-8b-instant",  
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        context = "\n\n".join([
            f"Document {i+1}:\n{doc.page_content}"
            for i, (doc, score) in enumerate(results)
        ])
        
        prompt = f"""You are a helpful AI assistant. Answer the question based on the context provided.

Context:
{context}

Question: {request.question}

Instructions:
- Answer based ONLY on the context provided
- If the answer is not in the context, say "I don't have enough information to answer that."
- Be concise and accurate
- Cite which document you used if relevant

Answer:"""
        
        response = llm.invoke(prompt)
        answer = response.content
        
        sources = []
        for i, (doc, score) in enumerate(results):
            sources.append({
                "document": i + 1,
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "score": float(score),
                "metadata": doc.metadata
            })
        
        return JSONResponse(content={
            "question": request.question,
            "unique_id": request.unique_id,
            "answer": answer,
            "sources": sources,
            "retrieval_method": "advanced" if request.use_advanced else "basic",
            "num_sources": len(sources)
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
