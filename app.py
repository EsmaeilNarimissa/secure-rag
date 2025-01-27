import streamlit as st
import logging
from backend.main_cdb_Final import query_database, get_available_models

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def initialize_session_state():
    """Initialize session state variables."""
    logging.info("Initializing session state")
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        logging.info("Chat history initialized")
    if 'current_model' not in st.session_state:
        st.session_state.current_model = None
        logging.info("Current model initialized")
    if 'filter_metadata' not in st.session_state:
        st.session_state.filter_metadata = {}
        logging.info("Filter metadata initialized")
    logging.info("Session state initialization complete")

def main():
    logging.info("Starting application")
    # Initialize session state
    initialize_session_state()

    # Configure app layout and theme
    logging.info("Configuring app layout and theme")
    st.set_page_config(
        page_title="Local Secure Document Retrieval",
        page_icon="🔍",
        layout="wide"
    )

    # Apply dark theme styles
    logging.info("Applying dark theme styles")
    st.markdown("""
        <style>
        .stApp { background-color: #1E1E1E; color: #FFFFFF; }
        .stTextArea textarea { background-color: #2D2D2D; color: #FFFFFF; }
        .stButton button { background-color: #0E86D4; color: white; }
        div[data-testid="stExpander"] { background-color: #2D2D2D; border: none; }
        .source-box { background-color: #363636; padding: 10px; border-radius: 5px; margin: 5px 0; }
        </style>
    """, unsafe_allow_html=True)

    # Title and description
    logging.info("Displaying title and description")
    st.markdown("""
    # 🔍 Local Secure Document Retrieval
    This system uses advanced AI to search through scientific documents and provide detailed answers with citations from the source material.
    """)

    # Sidebar for model settings
    with st.sidebar:
        logging.info("Setting up sidebar")
        st.markdown("## Model Settings")
        
        # Provider selection
        provider = st.selectbox("Choose Provider", ["openai", "ollama"], key="provider")
        logging.info(f"Selected provider: {provider}")

        if provider:
            # Get available models from backend
            try:
                models = get_available_models(provider)
                available_models = list(models.keys())
                logging.info(f"Available models: {available_models}")
                
                if available_models:
                    selected_model = st.selectbox(
                        "Select a Model", 
                        available_models,
                        help="Hover over each model to see its description"
                    )
                    # Show model description
                    if selected_model:
                        st.info(models[selected_model])
                    st.session_state.current_model = selected_model
                    logging.info(f"Selected model: {selected_model}")
                else:
                    st.error("No models available for selected provider")
            except Exception as e:
                logging.error(f"Error fetching models: {e}")
                st.error("Failed to fetch available models")

        # Advanced settings expander
        with st.expander("Advanced Settings"):
            logging.info("Displaying advanced settings")
            st.number_input("Temperature", 0.0, 1.0, 0.1, 0.1, help="Controls randomness in responses")

    # Chat interface
    chat_container = st.container()

    # Display chat history
    with chat_container:
        for chat in st.session_state.chat_history:
            with st.container():
                # User question
                st.markdown(
                    f"""<div style='background-color: #2D2D2D; padding: 15px; border-radius: 10px; margin: 10px 0;'>
                        <span style='color: #9E9E9E;'>Question:</span>
                        <p style='color: #FFFFFF;'>{chat.get('question', '')}</p>
                    </div>""",
                    unsafe_allow_html=True
                )
                
                # Assistant response
                st.markdown(
                    f"""<div style='background-color: #363636; padding: 15px; border-radius: 10px; margin: 10px 0;'>
                        <span style='color: #4FB0FF;'>Answer:</span>
                        <p style='color: #FFFFFF;'>{chat.get('answer', '')}</p>
                    </div>""",
                    unsafe_allow_html=True
                )
                
                # Sources with detailed information
                if 'sources' in chat and chat['sources']:
                    stats = chat.get('stats', {})
                    unique_docs = stats.get('unique_docs', 0)
                    total_chunks = len(chat['sources'])
                    
                    logging.info(f"Displaying sources for question: '{chat.get('question', '')}'")
                    logging.info(f"Found {total_chunks} relevant chunks from {unique_docs} documents")
                    
                    with st.expander(f"📚 Supporting Evidence ({total_chunks} relevant excerpts from {unique_docs} documents)"):
                        for i, source in enumerate(chat['sources'], 1):
                            # Extract filename without path
                            filename = source['file'].split('/')[-1]
                            
                            # Format the source box
                            st.markdown(
                                f"""<div class='source-box' style='margin-bottom: 15px;'>
                                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                                        <div style='color: #4FB0FF; font-weight: bold;'>
                                            Source {i} of {total_chunks}
                                        </div>
                                        <div style='color: #9E9E9E; font-size: 0.9em;'>
                                            {filename} (Page {source['page']})
                                        </div>
                                    </div>
                                    <div style='background-color: #2D2D2D; padding: 12px; border-radius: 5px; margin-top: 8px;'>
                                        <span style='color: #E0E0E0; font-size: 0.95em;'>{source['excerpt']}</span>
                                    </div>
                                </div>""",
                                unsafe_allow_html=True
                            )
                            logging.info(f"Displayed source {i}: {filename} (Page {source['page']})")

    # Query input
    logging.info("Displaying query input")
    st.markdown("### Ask Your Question")
    query = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="Type your question here..."
    )

    # Submit button
    if st.button("Submit Query", type="primary"):
        if query and query.strip():
            logging.info(f"Processing query: {query}")
            with st.spinner("🤖 Processing your query..."):
                try:
                    # Create chat history format expected by the backend
                    formatted_history = [
                        {"human_input": msg.get("question", ""),
                         "ai_response": msg.get("answer", "")}
                        for msg in st.session_state.chat_history
                    ]
                    logging.info(f"Chat history length: {len(formatted_history)}")

                    # Query the database with formatted chat history
                    logging.info("Sending query to database")
                    results = query_database(
                        query=query,
                        chat_history=formatted_history,
                        llm_provider=provider.lower(),
                        model=st.session_state.current_model
                    )
                    logging.info("Query processed successfully")
                    
                    # Store in chat history with consistent structure
                    chat_entry = {
                        "question": query,
                        "answer": results.get("answer", ""),
                        "sources": results.get("sources", []),
                        "stats": results.get("stats", {})
                    }
                    logging.info(f"Sources found: {len(chat_entry['sources'])}")
                    
                    # Update chat history
                    st.session_state.chat_history.append(chat_entry)
                    logging.info("Chat history updated")
                    
                    # Rerun to update display
                    logging.info("Triggering page rerun")
                    st.rerun()
                    
                except Exception as e:
                    logging.error(f"Error processing query: {str(e)}", exc_info=True)
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    logging.info("Application startup")
    main()
