import streamlit as st
import os
from pdf2image import convert_from_bytes
from PIL import Image
import tempfile
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional, List, Any, Tuple
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from langchain.text_splitter import CharacterTextSplitter
from datetime import datetime
from google import genai
from google.genai import types
import logging
import uuid
import psycopg2
from psycopg2.extras import execute_values
from PyPDF2 import PdfReader

#==============================================================================
# Application Configuration
#==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)

for lib in ['watchdog', 'httpcore', 'matplotlib', 'PIL', 'streamlit', 'httpx', 'google']:
    logging.getLogger(lib).setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.info("Application started")

#==============================================================================
# API Configuration and Initialization
#==============================================================================

class Settings(BaseSettings):
    # Google API configuration
    google_api_key: Optional[str] = Field(None, env="GOOGLE_API_KEY")
    # Database configuration
    db_name: str = Field(..., env="DB_NAME")
    db_user: str = Field(..., env="DB_USER")
    db_password: str = Field(..., env="DB_PASSWORD")
    db_host: str = Field(..., env="DB_HOST")
    db_port: str = Field(..., env="DB_PORT")

    def initialize_client(self):
        if self.google_api_key:
            logger.debug("Initializing Google API client...")
            return genai.Client(api_key=self.google_api_key)
        logger.error("Google API key not found!")
        return None

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = False

try:
    settings = Settings()
    if not settings.google_api_key:
        logger.error("Google API key not found. Please add it to the .env file.")
        st.error("Google API key not found. Please add it to the .env file.")
        st.stop()
    client = settings.initialize_client()
except Exception as e:
    logger.error(f"Settings error: {str(e)}")
    st.error(f"Settings error: {str(e)}")
    st.stop()

#==============================================================================
# Model Configuration and Constants
#==============================================================================

MODEL_CONFIGS = {
    'response': {
        'name': 'gemini-2.0-flash',
    },
    'ocr': {
        'name': 'gemini-2.0-flash'
    },
    'embedding': {
        'name': "text-embedding-004"
    }
}

#==============================================================================
# Session State Management
#==============================================================================

# Initialize session state variables
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = str(uuid.uuid4())
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
if "chunked_documents" not in st.session_state:
    st.session_state.chunked_documents = []
if "index" not in st.session_state:
    st.session_state.index = None
if "bm25" not in st.session_state:
    st.session_state.bm25 = None
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = []

# Create temporary directory for file processing
temp_dir = tempfile.mkdtemp()
successful_files = []

#==============================================================================
# Document Processing and Utility Functions
#==============================================================================

def pdf_to_images_generator(pdf_input):
    """Convert PDF to PIL Image objects one by one (generator) from file object or file path."""
    try:
        logger.debug("Converting PDF to images (generator)...")
        if isinstance(pdf_input, str):  # If input is file path
            with open(pdf_input, 'rb') as file:
                pdf_bytes = file.read()
        else:  # If input is file object
            pdf_bytes = pdf_input.read()
        # Use convert_from_bytes with first_page/last_page to avoid loading all pages at once
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_input if isinstance(pdf_input, str) else pdf_input)
        num_pages = len(reader.pages)
        for page_num in range(1, num_pages + 1):
            images = convert_from_bytes(pdf_bytes, fmt='png', first_page=page_num, last_page=page_num)
            for img in images:
                yield img
    except Exception as e:
        logger.error(f"PDF conversion error: {str(e)}")
        st.error(f"PDF conversion error: {str(e)}")
        return

def get_embedding(text):
    try:
        logger.debug("Generating embedding for text...")
        result = client.models.embed_content(
            model=MODEL_CONFIGS['embedding']['name'],
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=4)
        )
        values = next((emb[1] for embedding in result.embeddings for emb in embedding if emb[0] == 'values'), None)
        logger.info("Embedding successfully generated")
        return values
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        print(f"Embedding error: {e}")
        return None

def process_document(file_path: str, use_ocr: bool = False, embedding_model_name: str = MODEL_CONFIGS['embedding']['name']) -> Tuple[List[str], Any, Any]:
    """Process a PDF document and create search indices, memory-efficiently."""
    try:
        logger.info(f"Processing document: {file_path}")
        documents = []
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        progress_bar = st.progress(0, text="Processing pages...")

        if use_ocr:
            # OCR processing path
            images_gen = pdf_to_images_generator(file_path)
            page_count = 0
            for i, image in enumerate(images_gen):
                page_count += 1
                document = extract_text_with_ocr(image)
                del image  # Free memory
                if document:
                    documents.append(document)
                else:
                    st.warning(f"OCR failed for page {i+1}")
                progress_bar.progress(page_count / total_pages, text=f"OCR processed {page_count}/{total_pages} pages")
        else:
            # Standard PDF processing path
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    documents.append(text)
                else:
                    st.warning(f"No text extracted from page {i+1}")
                progress_bar.progress((i + 1) / total_pages, text=f"Processed {i+1}/{total_pages} pages")

        progress_bar.empty()
        if not documents:
            raise Exception("No text extracted from document")

        # Rest of the processing remains the same
        combined_text = "\n".join(documents)
        text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
        chunked_documents = text_splitter.split_text(combined_text)
        embeddings = []
        batch_size = 2
        for i in range(0, len(chunked_documents), batch_size):
            batch = chunked_documents[i:i + batch_size]
            batch_embeddings = [get_embedding(chunk) for chunk in batch]
            embeddings.extend(filter(None, batch_embeddings))
        if not embeddings:
            raise Exception("No embeddings could be generated")
        embeddings = np.array(embeddings).astype(np.float32)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        logger.info("FAISS index successfully created")
        tokenized_docs = [doc.split() for doc in chunked_documents]
        bm25 = BM25Okapi(tokenized_docs)
        logger.info("Document successfully processed")
        return chunked_documents, index, bm25
    except Exception as e:
        logger.error(f"Document processing error: {str(e)}")
        st.error(f"Document processing failed: {str(e)}")
        return None, None, None

def create_message(role: str, content: str, metadata: dict = None) -> dict:
    """Create a message with role, content, and optional metadata."""
    message = {
        "role": role,
        "content": content,
        "metadata": metadata or {}
    }
    return message

def hybrid_search(query: str, k: int = 5):
    """Perform hybrid search using both BM25 and vector similarity."""
    try:
        chunked_documents = st.session_state.chunked_documents
        index = st.session_state.index
        bm25 = st.session_state.bm25
        bm25_scores = bm25.get_scores(query.split())
        query_embedding = get_embedding(query)
        if query_embedding is None:
            raise Exception("Failed to generate query embedding")
        query_embedding = np.array([query_embedding]).astype(np.float32)
        distances, indices = index.search(query_embedding, len(chunked_documents))
        vector_scores = 1 / (distances[0] + 1e-8)
        epsilon = 1e-8
        alpha = 0.5
        bm25_scores = np.clip((bm25_scores - np.min(bm25_scores)) / (np.ptp(bm25_scores) + epsilon), 0, 1)
        vector_scores = np.clip((vector_scores - np.min(vector_scores)) / (np.ptp(vector_scores) + epsilon), 0, 1)
        combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores
        top_k_indices = np.argsort(combined_scores)[::-1][:k]
        return [chunked_documents[i] for i in top_k_indices]
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        st.error(f"Search failed: {str(e)}")
        return []

#==============================================================================
# Model Response Generation
#==============================================================================

def get_gemini_response(query: str, context_docs: List[Any]) -> str:
    try:
        context = "\n".join([doc for doc in context_docs])
        formatted_prompt = f"""
        Based on the following context information, please answer the question.

        Context:
        {context}

        Question:
        {query}

        Instructions:
        - Answer only based on the provided context
        - If the answer cannot be found in the context, say so
        - Be concise and specific
        - Use the same language as the question
        """
        client = genai.Client(api_key=settings.google_api_key)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            response = client.models.generate_content(
                model=MODEL_CONFIGS['response']['name'],
                contents=formatted_prompt,
                config=types.GenerateContentConfig(
                    system_instruction="""You are a professional AI assistant. Answer questions based solely on the provided context. 
                    If the context doesn't contain relevant information, state that clearly. Use clear language and organize your response well.""",
                    max_output_tokens=1024,
                    temperature=0.5,
                ),
            )
            message_placeholder.markdown(response.text)
        return response.text
    except Exception as e:
        error_msg = f"Failed to generate response: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        return error_msg

def extract_text_with_ocr(image: Image.Image, ocr_model_name: str = MODEL_CONFIGS['ocr']['name']) -> str:
    try:
        prompt = """You are an OCR assistant. Carefully extract all text from the provided images as if you are describing them to a visually impaired person (e.g., Image: In this picture, 8 people are posed hugging each other). Use markdown formatting effectively:
        - Use # and ## for headings
        - Use - for unordered lists
        - Use 1. for ordered lists
        - Use *italic* and **bold** for emphasis
        - Use [text](URL) for links
        - Use markdown table formatting for tables
        For non-text elements, describe them as Image: Brief description. Maintain logical flow and separate sections with --- when needed to improve readability. Be precise and thorough in transcribing all content.
        Never skip any content! Convert the document as it is, creatively using markdown to reproduce it as faithfully as possible. Translate the text from the images sequentially without omissions. 
        Separate different images with ---. Do not add comments or summaries — start immediately with the first image.
        """
        response = client.models.generate_content(
            model=ocr_model_name,
            contents=[
                prompt,
                image
            ]
        )
        return response.text
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        return f"Image processing error: {str(e)}"

def hyde_retrieve(query: str) -> str:
    try:
        prompt = f"""
        # Task
        Create a hypothetical document that contains a plausible answer to the given question.
        
        # Important Rules
        - Even if you're not sure if the answer is correct, generate a coherent document containing relevant information
        - The document should reflect information that might be found in a real database
        - Don't make up specific facts, but use general knowledge
        - Ideally, the answer should contain information that could be found in real documents from a search engine
        - Don't cite any sources in your answer
        - Limit your answer to about 150 words
        - IMPORTANT: Respond in the SAME LANGUAGE as the question. If the question is in Turkish, Spanish, French, etc., your hypothetical document must be in that same language
        - Match the style, terminology and formality level of technical domains in the query
        
        # Question
        {query}
        
        # Hypothetical Document
        """
        client = genai.Client(api_key=settings.google_api_key)
        response = client.models.generate_content(
            model=MODEL_CONFIGS['response']['name'],
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens= 100,
            ),
        )   
        return response.text
    except Exception as e:
        error_msg = f"Failed to hyde retrieve generate response: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        return error_msg

#==============================================================================
# PostgreSQL Database Connection
#==============================================================================

def get_db_connection():
    """Create and return a database connection"""
    try:
        conn = psycopg2.connect(
            dbname=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
            host=settings.db_host,
            port=settings.db_port
        )
        conn.autocommit = True
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        return None

def init_db():
    """Initialize database tables"""
    try:
        conn = get_db_connection()
        if not conn:
            return
        with conn.cursor() as cur:
            # Chat history table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                session_id TEXT,
                user_message TEXT,
                ai_response TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # Vector embeddings table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS vector_embeddings (
                id SERIAL PRIMARY KEY,
                document_id TEXT,
                chunk_text TEXT,
                embedding FLOAT[],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # Chat sessions table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                session_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_archived BOOLEAN DEFAULT FALSE
            )
            """)
        conn.close()
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")

def save_chat_history(session_id, user_message, ai_response):
    """Save chat interaction to database"""
    try:
        conn = get_db_connection()
        if not conn:
            return
        with conn.cursor() as cur:
            cur.execute("""
            INSERT INTO chat_sessions (session_id, session_name)
            VALUES (%s, %s)
            ON CONFLICT (session_id) DO UPDATE
            SET last_updated = CURRENT_TIMESTAMP
            """, (session_id, f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"))
            cur.execute("""
            INSERT INTO chat_history (session_id, user_message, ai_response)
            VALUES (%s, %s, %s)
            """, (session_id, user_message, ai_response))
        conn.close()
        logger.info("Chat history saved successfully")
    except Exception as e:
        logger.error(f"Error saving chat history: {str(e)}")

def save_embeddings(document_id, chunks_with_embeddings):
    """Save document chunks and their embeddings"""
    try:
        conn = get_db_connection()
        if not conn:
            return
        with conn.cursor() as cur:
            data = [(document_id, chunk, embedding.tolist()) 
                   for chunk, embedding in chunks_with_embeddings]
            execute_values(cur, """
            INSERT INTO vector_embeddings (document_id, chunk_text, embedding)
            VALUES %s
            """, data, template="(%s, %s, %s)")
        conn.close()
        logger.info(f"Embeddings saved successfully for document {document_id}")
    except Exception as e:
        logger.error(f"Error saving embeddings: {str(e)}")

def get_vector_embeddings():
    """Fetch vector embeddings from database"""
    try:
        conn = get_db_connection()
        if not conn:
            return []
        with conn.cursor() as cur:
            cur.execute("""
            SELECT document_id, chunk_text, embedding, created_at 
            FROM vector_embeddings 
            ORDER BY created_at DESC
            LIMIT 100
            """)
            rows = cur.fetchall()
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"Error fetching vector embeddings: {str(e)}")
        return []

def get_chat_history(session_id=None):
    """Fetch chat history from database for a specific session or all sessions"""
    try:
        conn = get_db_connection()
        if not conn:
            return []
        with conn.cursor() as cur:
            if session_id:
                cur.execute("""
                SELECT session_id, user_message, ai_response, timestamp 
                FROM chat_history 
                WHERE session_id = %s
                ORDER BY timestamp ASC
                """, (session_id,))
            else:
                cur.execute("""
                SELECT session_id, user_message, ai_response, timestamp 
                FROM chat_history 
                ORDER BY timestamp DESC
                LIMIT 50
                """)
            rows = cur.fetchall()
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"Error fetching chat history: {str(e)}")
        return []

def get_chat_sessions():
    """Fetch all chat sessions from database"""
    try:
        conn = get_db_connection()
        if not conn:
            return []
        with conn.cursor() as cur:
            cur.execute("""
            SELECT session_id, session_name, created_at, last_updated
            FROM chat_sessions
            WHERE is_archived = FALSE
            ORDER BY last_updated DESC
            """)
            rows = cur.fetchall()
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"Error fetching chat sessions: {str(e)}")
        return []

def delete_chat_session(session_id):
    """Archive a chat session (soft delete)"""
    try:
        conn = get_db_connection()
        if not conn:
            return False
        with conn.cursor() as cur:
            cur.execute("""
            UPDATE chat_sessions
            SET is_archived = TRUE
            WHERE session_id = %s
            """, (session_id,))
        conn.close()
        logger.info(f"Chat session {session_id} archived successfully")
        return True
    except Exception as e:
        logger.error(f"Error archiving chat session: {str(e)}")
        return False

def update_session_name(session_id, new_name):
    """Update the name of a chat session"""
    try:
        conn = get_db_connection()
        if not conn:
            return False
        with conn.cursor() as cur:
            cur.execute("""
            UPDATE chat_sessions
            SET session_name = %s
            WHERE session_id = %s
            """, (new_name, session_id))
        conn.close()
        logger.info(f"Session name updated for {session_id}")
        return True
    except Exception as e:
        logger.error(f"Error updating session name: {str(e)}")
        return False

# Initialize database tables when app starts
init_db()

#==============================================================================
# Chat Session Management Functions
#==============================================================================

def create_new_session():
    """Create a new chat session and set it as active"""
    session_id = str(uuid.uuid4())
    session_name = f"{datetime.now().strftime('%Y-%m-%d %H:%M')}"
    # Create session in database
    conn = get_db_connection()
    if conn:
        with conn.cursor() as cur:
            cur.execute("""
            INSERT INTO chat_sessions (session_id, session_name)
            VALUES (%s, %s)
            """, (session_id, session_name))
        conn.close()
    # Update session state
    st.session_state.active_session_id = session_id
    st.session_state.chat_sessions[session_id] = {"name": session_name, "messages": []}
    # Clear current messages
    st.session_state.messages = []
    logger.info(f"Created new chat session: {session_id}")
    return session_id, session_name

def load_session(session_id):
    """Load a chat session and set it as active"""
    if session_id not in st.session_state.chat_sessions:
        # Fetch from database
        messages = []
        history = get_chat_history(session_id)
        for _, user_msg, ai_resp, timestamp in history:
            messages.append(create_message("user", user_msg, {"timestamp": timestamp.isoformat()}))
            messages.append(create_message("assistant", ai_resp, {"timestamp": timestamp.isoformat()}))
        # Get session name
        conn = get_db_connection()
        session_name = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        if conn:
            with conn.cursor() as cur:
                cur.execute("""
                SELECT session_name FROM chat_sessions WHERE session_id = %s
                """, (session_id,))
                result = cur.fetchone()
                if result:
                    session_name = result[0]
            conn.close()
        st.session_state.chat_sessions[session_id] = {
            "name": session_name,
            "messages": messages
        }
    # Set as active session
    st.session_state.active_session_id = session_id
    st.session_state.messages = st.session_state.chat_sessions[session_id]["messages"]
    logger.info(f"Loaded chat session: {session_id}")
    return session_id

def display_message(message: dict):
    """Display a chat message with enhanced formatting."""
    with st.chat_message(message["role"]):
        st.markdown(f"{message['content']}")

#==============================================================================
# User Interface Components
#==============================================================================

with st.sidebar:    
    # New chat button at the top
    if st.button("➕ New Chat", key="new_chat", use_container_width=True):
        create_new_session()
        st.rerun()
    # Document Upload in expander
    with st.expander("📄 Document Upload", expanded=False):
        uploaded_files = st.file_uploader("Choose your files", accept_multiple_files=True, type=["pdf"])
        if uploaded_files:
            col1, col2 = st.columns([0.7, 0.3])
            with col1:
                process_button = st.button("Process Documents", use_container_width=True)
            with col2:
                use_ocr = st.checkbox("Use OCR", value=False, help="Enable Multimodal OCR for scanned documents")
            if process_button:
                with st.spinner("Processing documents..."):
                    for uploaded_file in uploaded_files:
                        # Save file to temp directory
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        # Process document
                        try:
                            chunked_docs, index, bm25 = process_document(file_path, use_ocr=use_ocr)
                            st.session_state.chunked_documents = chunked_docs
                            st.session_state.index = index
                            st.session_state.bm25 = bm25
                            st.session_state.documents_processed = True
                            successful_files.append(uploaded_file.name)
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    st.markdown("---")
    # Display all sessions in sidebar
    sessions = get_chat_sessions()
    active_id = st.session_state.active_session_id
    if sessions:
        # Add custom CSS for active session highlight (center text)
        st.markdown("""
            <style>
            .active-session {
                background-color: #17644d !important;
                color: white !important;
                border-radius: 8px;
                padding: 0.4em 0.8em;
                margin-bottom: 0.2em;
                display: flex;
                align-items: center;
                justify-content: center;
                text-align: center;
                font-weight: 500;
                height: 2.5em;
                min-height: 2.5em;
            }
            </style>
        """, unsafe_allow_html=True)
        for session_id, session_name, created_at, last_updated in sessions:
            col1, col2 = st.columns([0.8, 0.2])
            if session_id == active_id:
                with col1:
                    st.markdown(
                        f'<div class="active-session">{session_name}</div>',
                        unsafe_allow_html=True
                    )
            else:
                with col1:
                    if st.button(f"{session_name}", key=f"select_{session_id}", use_container_width=True):
                        load_session(session_id)
                        st.rerun()
            with col2:
                if st.button("🗑️", key=f"delete_{session_id}"):
                    delete_chat_session(session_id)
                    if session_id == active_id:
                        create_new_session()
                    st.rerun()
    else:
        st.info("No chat history found. Start a new chat!")

#==============================================================================
# Main chat interface
#==============================================================================

if not st.session_state.documents_processed:
    st.warning("Please upload and process documents using the sidebar to enable full functionality.")
else:
    # Automatic session management: create if no active session
    if not st.session_state.active_session_id or st.session_state.active_session_id not in st.session_state.chat_sessions:
        session_id, session_name = create_new_session()
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            display_message(message)
    # Place user input at the bottom
    prompt = st.chat_input("What's up?")
    if prompt:
        user_message = create_message("user", prompt, {
            "timestamp": datetime.now().isoformat(),
            "type": "original_query"
        })
        st.session_state.messages.append(user_message)
        st.session_state.chat_sessions[st.session_state.active_session_id]["messages"] = st.session_state.messages
        display_message(user_message)
        # HYDE RETRIEVAL and answer generation
        with st.spinner("Searching documents..."):
            search_hyde_query = hyde_retrieve(prompt) + " " + prompt
            relevant_docs = hybrid_search(search_hyde_query)   
        response = get_gemini_response(prompt, relevant_docs)
        assistant_message = create_message("assistant", response, {
            "context_docs": relevant_docs,
            "timestamp": datetime.now().isoformat(),
            "original_query": prompt,
            "hybrid_search_query": search_hyde_query
        })
        # Update messages
        st.session_state.messages.append(assistant_message)
        st.session_state.chat_sessions[st.session_state.active_session_id]["messages"] = st.session_state.messages
        # Save to database
        save_chat_history(
            session_id=st.session_state.active_session_id,
            user_message=prompt,
            ai_response=response
        )
