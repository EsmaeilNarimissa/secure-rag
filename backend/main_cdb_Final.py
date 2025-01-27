import os
import sys
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
import chromadb
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import torch

# Load environment variables
load_dotenv()

# Enhanced prompt template for structured, detailed answers
CUSTOM_PROMPT = """You are an expert scientific assistant specializing in polymer science and rheology.
Given the following context and question, provide a comprehensive answer that follows this structure:

1. DEFINITION & OVERVIEW:
   - Clear definition of the main concept
   - Brief historical context if relevant

2. TECHNICAL DETAILS:
   - Key components and mechanisms
   - Mathematical or theoretical foundations
   - Important parameters and variables

3. APPLICATIONS & SIGNIFICANCE:
   - Practical applications
   - Advantages and limitations
   - Relationship to other models or concepts

For follow-up questions:
- Avoid repeating details already covered unless necessary for clarity.
- Focus on providing new insights, deeper explanations, or additional context related to the follow-up question.

Use information strictly from the provided context. If the context lacks sufficient detail,
explicitly indicate this and do not provide information based on assumptions or unrelated concepts.

Context: {context}
Question: {input}

When responding:
- Start with a clear, concise definition
- Use technical terms appropriately
- Highlight key relationships and dependencies
- Maintain scientific accuracy
- Include specific examples from the sources

Answer: """


def get_available_models(provider: str) -> dict:
    """
    Get available models for the specified provider.
    Returns a dictionary with model keys and their descriptions.
    """
    models = {
        "openai": {
            "GPT-4 Turbo": "Most capable model with exceptional accuracy and clarity. Excels in complex reasoning, mathematical understanding, and precise technical explanations.",
            "GPT-4o": "Advanced model delivering detailed analysis with strong technical precision and conceptual understanding. Ideal for in-depth technical tasks.",
            "GPT-4o-mini": "Efficient and cost-effective model balancing speed with accuracy. Perfect for quick, reliable responses while maintaining high quality.",
        },
        "ollama": {
            "llama3.2-vision": "Advanced LLaMA model with vision capabilities, offering strong performance for multimodal tasks and complex reasoning",
            "deepseek-r1:14b": "Reliable open-source LLaMA model with balanced performance for general text processing and analysis"
        }
    }
    return models.get(provider, {})


def validate_model_availability(provider: str, model: str) -> bool:
    """Validate if the selected model is available."""
    return model in get_available_models(provider)


def select_model(provider: str) -> str:
    """
    Interactive model selection based on provider.
    Returns the selected model name.
    """
    models = get_available_models(provider)
    
    print(f"\nAvailable {provider.upper()} models:")
    print("=" * 50)
    
    # Display models with descriptions
    for i, (model, description) in enumerate(models.items(), 1):
        print(f"{i}. {model:<20} - {description}")
    
    while True:
        try:
            choice = input(f"\nSelect model (1-{len(models)}): ").strip()
            if choice.lower() == 'exit':
                sys.exit(0)
            
            choice = int(choice)
            if 1 <= choice <= len(models):
                return list(models.keys())[choice - 1]
            else:
                print(f"Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("Please enter a valid number")


def get_llm_provider(provider: str, temperature: float = 0.1, model: str = None):
    """
    Dynamically initialize the LLM provider (OpenAI or Ollama) with selected model.
    """
    if provider.lower() not in ["openai", "ollama"]:
        raise ValueError("Unsupported provider. Choose 'openai' or 'ollama'.")
    
    # Remove the model selection from here since it's handled in main
    if model is None:
        raise ValueError("Model must be specified")
    
      
    if provider.lower() == "openai":
        # Map friendly names to actual OpenAI model IDs
        model_mapping = {
            "GPT-4 Turbo": "gpt-4-turbo",
            "GPT-4o": "gpt-4o",
            "GPT-4o-mini": "gpt-4o-mini",
         }
        actual_model = model_mapping.get(model, model)
        return ChatOpenAI(temperature=temperature, model=actual_model)
    else:  # ollama
        return ChatOllama(temperature=temperature, model=model)


def format_sources(docs: List[Document]) -> str:
    """Format source documents for display with improved filtering"""
    source_info = []
    seen_sources = set()  # Track unique sources
    
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'Unknown')
        
        # Skip if:
        # - it's a references page
        # - source already seen
        # - content too short
        # - content is mostly numbers or special characters
        if ('reference' in doc.page_content.lower() or 
            source in seen_sources or 
            len(doc.page_content.strip()) < 50 or
            sum(c.isalpha() for c in doc.page_content) < len(doc.page_content) * 0.3):
            continue
            
        seen_sources.add(source)
        
        # Clean and truncate excerpt
        excerpt = doc.page_content.replace('\n', ' ').strip()
        excerpt = ' '.join(excerpt.split())  # Remove multiple spaces
        excerpt = excerpt[:150] + "..." if len(excerpt) > 150 else excerpt
        
        source_info.append(f"\nSource {len(source_info) + 1}:")
        source_info.append(f"📄 File: {os.path.basename(source)}")
        source_info.append(f"📑 Page: {page}")
        source_info.append(f"💡 Excerpt: {excerpt}\n")
    
    return "\n".join(source_info) if source_info else "No relevant sources found."


def adapt_query_with_context(query: str, chat_history: List[Dict[str, Any]]) -> str:
    """
    Refine the query dynamically by incorporating context from the last 3 recent interactions.
    """
    if chat_history:
        # Extract recent topics from the last 3 interactions
        recent_topics = " | ".join(
            entry["human_input"] for entry in chat_history[-3:]  # Use up to the last 3 interactions
        )
        # Adapt the query with the recent context
        query = f"{query} (focused on {recent_topics})"
    return query


def query_database(query: str, chat_history: List[Dict[str, Any]] = [], filter_metadata: dict = None, llm_provider: str = "openai", model: str = None):
    """Query the vector database interactively with support for multiple LLM providers."""

    # Limit chat history to the last 10 interactions
    if len(chat_history) > 10:
        chat_history = chat_history[-10:]  # Retain only the last 10 entries

    # Refine the query dynamically using the most recent interaction
    query = adapt_query_with_context(query, chat_history)

    # Initialize LLM dynamically based on provider and model
    llm = get_llm_provider(llm_provider, temperature=0.1, model=model)

    # Define paths
    persist_directory = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "ChromaDB"))
    collection_name = "pes-review-hmmsf-pdf-BGE"  # Updated to match ingestion collection name

    # Initialize Chroma
    client = chromadb.PersistentClient(path=persist_directory)
    
    # Initialize embeddings - Choose between BGE and OpenAI
    # Option 1: BGE Embeddings (Currently active)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # # Option 2: OpenAI Embeddings (Commented out)
    # embeddings = OpenAIEmbeddings()
    
    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    # Use a broader rephrase prompt
    rephrase_prompt = PromptTemplate(
        template="Rephrase the following query into a clear and standalone question, ensuring it is context-independent and easy to understand:\n\nOriginal Query: {input}\n\nRephrased Query:",
        input_variables=["input"]
    )

    # Create custom prompt
    prompt = PromptTemplate(
        template=CUSTOM_PROMPT,
        input_variables=["context", "input"]
    )

    # Create document chain
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )

    # Configure base retriever with metadata filtering
    base_retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 30,  # Increased from 5 to get more context
            "filter": filter_metadata if filter_metadata else None
        }
    )

    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=base_retriever,
        prompt=rephrase_prompt
    )

    # Create retrieval chain with history awareness
    retrieval_chain = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=combine_docs_chain
    )

    # Execute query with chat history
    print(f"\n🔍 Query: '{query}'")
    if filter_metadata:
        print(f"📋 Filters applied: {filter_metadata}")
    if chat_history:
        print(f"💬 Using chat history with {len(chat_history)} previous interactions")

    result = retrieval_chain.invoke({
        "input": query,
        "chat_history": chat_history
    })
    
    # Process and display results
    if "answer" in result:
        formatted_sources = []
        for doc in result.get("context", []):
            formatted_sources.append({
                'file': os.path.basename(doc.metadata.get('source', 'Unknown')),
                'page': doc.metadata.get('page', 'Unknown'),
                'excerpt': doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
            })

        return {
            'answer': result["answer"],
            'sources': formatted_sources,
            'stats': {
                'sources_used': len(result.get('context', [])),
                'unique_docs': len(set(doc.metadata.get('source') for doc in result.get('context', [])))
            }
        }
    else:
        raise ValueError("No answer found in result")


if __name__ == "__main__":
    print("Enhanced Scientific Document Retrieval System with Memory")
    print("=====================================================")
    
    # Initialize chat history
    chat_history = []

    # Select the LLM provider
    while True:
        provider = input("Select LLM provider (openai/ollama): ").strip().lower()
        if provider in ["openai", "ollama"]:
            break
        print("Invalid provider. Please choose 'openai' or 'ollama'")

    # Get model selection once
    model = select_model(provider)
    
    try:
        # Interactive question loop
        while True:
            query = input("\nEnter your question (or type 'exit' to end): ").strip()
            if query.lower() in {"stop", "quit", "exit"}:
                print("\nGoodbye!")
                break

            # Pass the selected model explicitly to query_database
            result = query_database(
                query=query, 
                chat_history=chat_history, 
                llm_provider=provider,
                model=model
            )
            if "answer" in result:
                chat_history.append({
                    "human_input": query,
                    "ai_response": result["answer"]
                })
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
