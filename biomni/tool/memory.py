import os
import subprocess
from datetime import datetime
from typing import List, Dict, Any
from langchain_classic.memory import VectorStoreRetrieverMemory
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Configuration
def _get_qdrant_path():
    """Resolve the Qdrant database path."""
    # Try environment variable first
    biomni_path = os.getenv("BIOMNI_DATA_PATH") or os.getenv("BIOMNI_PATH")
    
    if not biomni_path:
        # Fallback: look for biomni_data relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Check up to 3 levels up
        search_path = current_dir
        for _ in range(3):
            candidate = os.path.join(search_path, "biomni_data")
            if os.path.isdir(candidate):
                biomni_path = candidate
                break
            search_path = os.path.dirname(search_path)
    
    if not biomni_path:
        # Default to absolute path relative to execution if all else fails, 
        # but use a fixed location to avoid CWD issues
        # Try to find the project root (Biomni_HITS)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        biomni_path = os.path.join(project_root, "biomni_data")
    
    db_path = os.path.join(biomni_path, "qdrant_db")
    return db_path

def _get_gemini_api_key():
    """Retrieve Gemini API key from environment or bash_profile."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    "source ~/.bash_profile 2>/dev/null && echo $GEMINI_API_KEY",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.stdout.strip():
                api_key = result.stdout.strip()
                os.environ["GEMINI_API_KEY"] = api_key
        except Exception as e:
            print(f"Note: Could not load GEMINI_API_KEY from bash_profile: {e}")
    return api_key

QDRANT_PATH = _get_qdrant_path()
COLLECTION_NAME = "biomni_conversations"
EMBEDDING_MODEL = "models/embedding-001"

def get_memory() -> VectorStoreRetrieverMemory:
    """
    Initializes and returns the VectorStoreRetrieverMemory with Qdrant and Gemini embeddings.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(QDRANT_PATH), exist_ok=True)

    # Get API Key
    api_key = _get_gemini_api_key()
    if not api_key:
        print("Warning: GEMINI_API_KEY not found. Embedding operations may fail.")

    # Initialize Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key
    )

    # Initialize Qdrant Client
    client = QdrantClient(path=QDRANT_PATH)
    
    # Check if collection exists, if not create it
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

    # Initialize Vector Store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    # Initialize Retriever with more results for better recall
    # Increased to 20 to ensure we capture all relevant conversations
    # (Some conversations may rank lower but still be relevant)
    retriever = vector_store.as_retriever(search_kwargs={"k": 20})

    # Initialize Memory
    memory = VectorStoreRetrieverMemory(retriever=retriever)
    
    return memory

def save_conversation(input_text: str, output_text: str) -> None:
    """
    Saves the conversation interaction (input and output) to the vector store memory.
    Includes detailed timestamp (year, date, time) in both content and metadata for LLM to understand when conversations occurred.
    
    Args:
        input_text (str): The user's input message.
        output_text (str): The agent's response message.
    """
    try:
        # Get current datetime with detailed breakdown
        now = datetime.now()
        year = now.year
        date_str = now.strftime("%Y-%m-%d")  # e.g., "2025-12-01"
        time_str = now.strftime("%H:%M:%S")  # e.g., "14:30:45"
        datetime_str = now.strftime("%Y-%m-%d %H:%M:%S")  # Full datetime for display
        timestamp_float = now.timestamp()  # For easy sorting
        
        # Create document with timestamp information in content so LLM can see it
        # Format: Include time information at the start so it's visible to the LLM
        content = f"[Date: {date_str}, Time: {time_str}, Year: {year}]\ninput: {input_text}\n\noutput: {output_text}"
        
        doc = Document(
            page_content=content,
            metadata={
                "year": year,
                "date": date_str,
                "time": time_str,
                "datetime": datetime_str,
                "timestamp": now.isoformat(),
                "timestamp_float": timestamp_float,
                "input": input_text,
                "output": output_text
            }
        )
        
        # Get vector store and add document directly
        memory = get_memory()
        vector_store = memory.retriever.vectorstore
        
        # Add document to vector store
        vector_store.add_documents([doc])
        
        print(f"Conversation saved to memory ({datetime_str}): {input_text[:30]}... -> {output_text[:30]}...")
    except Exception as e:
        print(f"Error saving conversation to memory: {e}")

def retrieve_past_conversations(query: str) -> str:
    """
    Retrieves relevant past conversation history based on the query.
    Use this tool when the user asks about previous interactions, their name, personal details provided earlier,
    or context from past conversations.
    
    IMPORTANT: When the user asks "what did we talk about?" or "previous conversations", 
    use the user's ACTUAL question or keywords from their question as the query parameter.
    For example, if user asks "what did we discuss about admet?", use "admet" as the query.
    If user asks "what is my name?", use "name" or the user's actual question as the query.
    Do NOT use generic phrases like "previous conversation history" - use specific keywords from the user's question.
    
    Args:
        query (str): The query string to search for in the conversation history. 
                    Should be specific keywords or the user's actual question, not generic phrases.
        
    Returns:
        str: A formatted string containing relevant past conversations.
    """
    print(f"DEBUG: retrieve_past_conversations called with query: '{query}'")
    try:
        memory = get_memory()
        # The memory.load_memory_variables method retrieves relevant docs based on the input key
        # usually it expects a dict, but VectorStoreRetrieverMemory uses the value associated with input_key to search
        # Default input_key is "memory" or similar? No, VectorStoreRetrieverMemory defaults: input_key="input"
        
        # We can also use the retriever directly from the vector store if we want more control,
        # but let's use the memory interface or the retriever.
        
        # Accessing the retriever directly might be cleaner for a "tool" response
        retriever = memory.retriever
        vector_store = retriever.vectorstore
        
        # If query is too generic, try to extract keywords or use the query as-is
        # Use vector_store directly for semantic search
        docs = vector_store.similarity_search(query, k=20)
        
        if not docs:
            return "No relevant past conversations found."
        
        # Sort documents by timestamp (most recent first, or oldest first based on preference)
        # Extract timestamp from metadata and sort
        docs_with_timestamps = []
        for doc in docs:
            metadata = doc.metadata or {}
            timestamp_float = metadata.get("timestamp_float", 0)
            docs_with_timestamps.append((timestamp_float, doc))
        
        # Sort by timestamp (oldest first for chronological order)
        docs_with_timestamps.sort(key=lambda x: x[0])
        sorted_docs = [doc for _, doc in docs_with_timestamps]
        
        result = "Found relevant past conversations (in chronological order):\n\n"
        for i, doc in enumerate(sorted_docs, 1):
            metadata = doc.metadata or {}
            # Try to get formatted datetime, fallback to timestamp or content
            datetime_str = metadata.get("datetime")
            if not datetime_str:
                datetime_str = metadata.get("timestamp", "Unknown time")
            
            # Format: Year, Date, Time for LLM clarity
            year = metadata.get("year", "Unknown")
            date = metadata.get("date", "Unknown")
            time = metadata.get("time", "Unknown")
            
            result += f"--- Conversation {i} (Year: {year}, Date: {date}, Time: {time}) ---\n{doc.page_content}\n\n"
        
        print(f"DEBUG: Retrieved {len(sorted_docs)} conversations for query: '{query}' (sorted chronologically)")
        return result
    except Exception as e:
        return f"Error retrieving past conversations: {e}"

