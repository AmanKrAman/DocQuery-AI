"""
Advanced Retrieval Techniques for RAG
Includes: Hybrid Search, Reranking, RRF, Multi-Query
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder
import os
from dotenv import load_dotenv

load_dotenv()


class DocumentRetriever:
    """
    Advanced document retrieval with multiple strategies
    """
    
    def __init__(self, vector_store_path: str = "data/vector_db", collection_name: str = "documents"):
        self.vector_store_path = vector_store_path
        self.collection_name = collection_name
        
        # Initialize embeddings (same as ingestion)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize reranker (cross-encoder for better accuracy)
        print("Loading reranker model...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Initialize LLM for multi-query
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
    
    def load_vectorstore(self, unique_id: str):
        """Load vector store for specific user"""
        collection = f"{self.collection_name}_{unique_id}"
        
        vectorstore = Chroma(
            persist_directory=self.vector_store_path,
            embedding_function=self.embeddings,
            collection_name=collection,
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        return vectorstore
    
    # 1. BASIC VECTOR SEARCH
    def vector_search(self, query: str, unique_id: str, k: int = 10):
        """
        Basic semantic search using embeddings
        
        Args:
            query: User's question
            unique_id: User's document ID
            k: Number of results to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        print(f"🔍 Vector search for: {query}")
        
        vectorstore = self.load_vectorstore(unique_id)
        results = vectorstore.similarity_search_with_score(query, k=k)
        
        print(f"✅ Found {len(results)} results")
        return results
    
    # 2. MULTI-QUERY RETRIEVAL
    def generate_multiple_queries(self, query: str, num_queries: int = 3) -> list[str]:
        """
        Generate multiple variations of the query using LLM
        
        Why: Different phrasings catch different relevant docs
        """
        print(f"🤖 Generating {num_queries} query variations...")
        
        prompt = f"""You are an AI assistant. Generate {num_queries} different versions of this question to retrieve relevant documents.

Original question: {query}

Provide {num_queries} alternative questions that mean the same thing but use different wording.
Format: Just list the questions, one per line, no numbering.

Alternative questions:"""
        
        response = self.llm.invoke(prompt)
        queries = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
        
        # Add original query
        all_queries = [query] + queries[:num_queries-1]
        
        print(f"📝 Generated queries:")
        for i, q in enumerate(all_queries, 1):
            print(f"   {i}. {q}")
        
        return all_queries
    
    def multi_query_search(self, query: str, unique_id: str, k: int = 5):
        """
        Search using multiple query variations and combine results
        
        Better than single query - catches more relevant docs
        """
        print(f"\n{'='*60}")
        print(f"MULTI-QUERY RETRIEVAL")
        print(f"{'='*60}")
        
        # Generate query variations
        queries = self.generate_multiple_queries(query)
        
        # Search with each query
        all_results = {}
        vectorstore = self.load_vectorstore(unique_id)
        
        for q in queries:
            results = vectorstore.similarity_search_with_score(q, k=k)
            for doc, score in results:
                # Use document content as key to deduplicate
                doc_key = doc.page_content
                if doc_key not in all_results or score < all_results[doc_key][1]:
                    all_results[doc_key] = (doc, score)
        
        # Convert back to list and sort by score
        final_results = sorted(all_results.values(), key=lambda x: x[1])
        
        print(f"✅ Multi-query found {len(final_results)} unique results")
        return final_results[:k]
    
    # 3. RERANKER
    def rerank_results(self, query: str, results: list, top_k: int = 5):
        """
        Rerank results using cross-encoder for better accuracy
        
        Why: Initial search is fast but approximate. Reranker is slower but more accurate.
        """
        print(f"\n🔄 Reranking {len(results)} results...")
        
        # Prepare pairs for reranker
        pairs = [[query, doc.page_content] for doc, _ in results]
        
        # Get reranker scores
        scores = self.reranker.predict(pairs)
        
        # Combine with original docs
        reranked = [(results[i][0], float(scores[i])) for i in range(len(results))]
        
        # Sort by reranker score (higher is better)
        reranked = sorted(reranked, key=lambda x: x[1], reverse=True)
        
        print(f"✅ Reranked! Top score: {reranked[0][1]:.4f}")
        
        return reranked[:top_k]
    
    # 4. RECIPROCAL RANK FUSION (RRF)
    def reciprocal_rank_fusion(self, result_lists: list[list], k: int = 60):
        """
        Combine multiple ranked lists using RRF
        
        Formula: RRF_score = sum(1 / (k + rank))
        
        Why: Different retrieval methods find different docs. RRF combines them smartly.
        """
        print(f"\n🔀 Applying Reciprocal Rank Fusion...")
        
        # Track scores for each document
        doc_scores = {}
        
        for result_list in result_lists:
            for rank, (doc, _) in enumerate(result_list, start=1):
                doc_key = doc.page_content
                
                # RRF formula
                rrf_score = 1.0 / (k + rank)
                
                if doc_key not in doc_scores:
                    doc_scores[doc_key] = {'doc': doc, 'score': 0}
                
                doc_scores[doc_key]['score'] += rrf_score
        
        # Sort by RRF score
        final_results = sorted(
            [(v['doc'], v['score']) for v in doc_scores.values()],
            key=lambda x: x[1],
            reverse=True
        )
        
        print(f"✅ RRF combined {len(result_lists)} result lists into {len(final_results)} unique docs")
        
        return final_results
    
    # 5. COMPLETE ADVANCED RETRIEVAL
    def advanced_search(self, query: str, unique_id: str, top_k: int = 5, use_multi_query: bool = True, use_reranker: bool = True):
        """
        Complete advanced retrieval pipeline:
        1. Multi-query retrieval (optional)
        2. Vector search
        3. Reranking (optional)
        4. RRF to combine (if multi-query used)
        
        Args:
            query: User's question
            unique_id: User's document ID
            top_k: Final number of results
            use_multi_query: Whether to use multi-query retrieval
            use_reranker: Whether to rerank results
            
        Returns:
            List of (document, score) tuples
        """
    
        print(f"SEARCH: {query}")
        
        print(f"Multi-query: {use_multi_query} | Reranker: {use_reranker}")
        
        if use_multi_query:
            # Multi-query approach
            queries = self.generate_multiple_queries(query, num_queries=3)
            vectorstore = self.load_vectorstore(unique_id)
            
            result_lists = []
            for q in queries:
                results = vectorstore.similarity_search_with_score(q, k=10)
                result_lists.append(results)
            
            # Combine with RRF
            combined_results = self.reciprocal_rank_fusion(result_lists)
            
            # Take top results for reranking
            results_to_rerank = combined_results[:20]
        else:
            # Single query
            results_to_rerank = self.vector_search(query, unique_id, k=20)
        
        # Rerank if enabled
        if use_reranker:
            final_results = self.rerank_results(query, results_to_rerank, top_k=top_k)
        else:
            final_results = results_to_rerank[:top_k]
        
        print(f"\n✅ Final results: {len(final_results)} documents")
        return final_results
