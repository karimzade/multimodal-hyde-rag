# Multimodal HyDE RAG

A Streamlit-based chat application that implements Hypothetical Document Embeddings (HyDE) and multimodal RAG (Retrieval-Augmented Generation) for enhanced document interaction. This application allows users to upload PDF documents, process them with optional OCR capabilities, and engage in context-aware conversations about their content using Google's Gemini API.

## Prerequisites

- Python 3.8 or higher
- PostgreSQL database
- Poppler (required by pdf2image for PDF to image conversion)

## Installation Steps

### 1. PostgreSQL Setup
```bash
# Install PostgreSQL (Ubuntu/Debian)
sudo apt update
sudo apt install postgresql postgresql-contrib

# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Connect to PostgreSQL as postgres user
sudo -u postgres psql

# Create a new database user (replace <your_db_user> and <your_password> with your own values)
CREATE USER <your_db_user> WITH PASSWORD '<your_password>';

# Create a new database and set ownership (replace <your_db_name> and <your_db_user>)
CREATE DATABASE <your_db_name> OWNER <your_db_user>;

# Grant privileges
GRANT ALL PRIVILEGES ON DATABASE <your_db_name> TO <your_db_user>;

# Exit PostgreSQL prompt
\q

# Test connection with your user
psql -U <your_db_user> -d <your_db_name> -h localhost
# Enter your password when prompted
```

### 2. Project Setup

1. Clone the repository:
```bash
git clone https://github.com/karimzade/multimodal-hyde-rag.git 
cd multimodal-hyde-rag
```

2. (Recommended) Create a Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install system dependencies:
```bash
sudo apt-get update
sudo apt-get install -y poppler-utils
```

4. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root with the following variables. **Replace the example values with your own database credentials and API key:**

```env
# Required API Keys
GOOGLE_API_KEY=your_google_api_key

# Database Configuration
DB_NAME=your_database_name      # e.g. the name you used for <your_db_name>
DB_USER=your_database_user      # e.g. the name you used for <your_db_user>
DB_PASSWORD=your_database_password  # e.g. the password you set for <your_db_user>
DB_HOST=localhost
DB_PORT=5432
```

### Environment Variables Description

- `GOOGLE_API_KEY`: API key for Google Gemini services
- `DB_NAME`: PostgreSQL database name
- `DB_USER`: Database user
- `DB_PASSWORD`: Database password
- `DB_HOST`: Database host address
- `DB_PORT`: Database port number (usually 5432)

## Database Setup

1. Create a PostgreSQL database:
```bash
createdb your_database_name
```

2. The application will automatically create the required tables on first run.

## Running the Application

1. Start the Streamlit application:
```bash
streamlit run main.py
```

2. Access the application through your web browser at `http://localhost:8501`

## Usage

1. Use the sidebar to upload PDF documents
2. Choose whether to enable OCR for scanned documents
3. Process the documents using the "Process Documents" button
4. Start chatting with the AI about the document content
5. Create new chat sessions or switch between existing ones using the sidebar

## Features

- PDF document processing with OCR support
- Hybrid search combining BM25 and vector similarity
- Chat session management
- Persistent conversation history
- Hypothetical Document Embeddings (HyDE) for improved retrieval
- Multimodal document understanding
- PostgreSQL database integration for data persistence

## Notes

- Large PDF files may take longer to process, especially with OCR enabled
- The application stores all chat history and document embeddings in the configured PostgreSQL database
