# GovAssist-RAG: Government Schemes Assistant

A Retrieval-Augmented Generation (RAG) system that helps users find and learn about Indian government schemes. This project scrapes government scheme data, embeds it using sentence transformers, stores it in a vector database (Qdrant), and provides an AI-powered chat interface to answer questions about available schemes.

## Features

- **Web Scraping**: Automatically scrapes government scheme data from myscheme.gov.in across multiple categories (Agriculture, Banking, Education, Health, Women & Child)
- **Vector Search**: Uses sentence-transformers (BGE model) for semantic search of schemes
- **AI-Powered Responses**: Integrates with Groq's Llama models for natural language answers
- **Chat Sessions**: Maintains conversation history for contextual follow-up questions
- **Dual Interface**: 
  - FastAPI backend for API access
  - Streamlit web UI for user-friendly interaction
- **Fallback Search**: Keyword-based search when vector search fails
- **Tag-Based Filtering**: Auto-detects query categories (student, farmer, women, loan, etc.) for better results
- **Local Vector Database**: Uses Qdrant for efficient vector storage and retrieval

## Architecture

The system consists of several key components:

- **Scraper (`scrape.py`)**: Uses Playwright to collect scheme data from government websites
- **Embedding Service (`embed.py`)**: Converts text to vector embeddings using BGE-small-en-v1.5
- **Vector Database (`qdrant_db.py`)**: Manages Qdrant collections for vector storage and search
- **LLM Client (`groq_client.py`)**: Handles interactions with Groq API for answer generation
- **RAG Pipeline (`rag_pipeline.py`)**: Orchestrates the entire retrieval and generation process
- **Session Management (`checkpointer.py`)**: Stores chat history in JSON files
- **API Server (`main.py`)**: FastAPI backend providing REST endpoints
- **Web UI (`streamlit_app.py`)**: Streamlit frontend for user interaction

## Tech Stack

- **Backend**: Python, FastAPI, Uvicorn
- **Frontend**: Streamlit
- **AI/ML**: Sentence Transformers, Groq API (Llama 3.1)
- **Vector Database**: Qdrant (local or cloud)
- **Web Scraping**: Playwright
- **Data Storage**: JSON files for schemes and chat sessions

## Installation

### Prerequisites

- Python 3.8+
- Node.js (for Playwright, if scraping)
- Groq API key

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd GovAssist-RAG
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   LOG_LEVEL=INFO
   SCHEMES_FILE=scheme.json
   AUTO_INGEST=true
   FORCE_RECREATE_COLLECTION=false
   QDRANT_MODE=local
   QDRANT_LOCAL_PATH=./qdrant_data
   API_BASE_URL=http://127.0.0.1:8000
   ```

   Get your Groq API key from [groq.com](https://groq.com).

5. **Scrape scheme data** (optional, if you want fresh data):
   ```bash
   python scrape.py
   ```
   
   This will create/update `scheme.json` with the latest scheme information.

## Usage

### Starting the API Server

```bash
python main.py
```

The FastAPI server will start on `http://127.0.0.1:8000`.

### Starting the Web UI

```bash
streamlit run streamlit_app.py
```

The Streamlit app will be available at `http://localhost:8501`.

### API Endpoints

- `GET /health` - Health check
- `POST /chat` - Send a query and get AI response
  - Request body:
    ```json
    {
      "query": "What schemes are available for students?",
      "top_k": 5,
      "session_id": "optional-session-id"
    }
    ```
  - Response:
    ```json
    {
      "session_id": "generated-or-provided-session-id",
      "query": "What schemes are available for students?",
      "answer": "AI-generated answer...",
      "matches": [...]
    }
    ```
- `GET /sessions/{session_id}` - Get chat history for a session

### Example Queries

- "What education schemes are available for students?"
- "Schemes for farmers in agriculture"
- "Women empowerment programs"
- "Loan schemes for small businesses"
- "Health insurance for families"

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | - | Required. Your Groq API key |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | LLM model to use |
| `LOG_LEVEL` | `INFO` | Logging level |
| `SCHEMES_FILE` | `scheme.json` | Path to schemes data file |
| `AUTO_INGEST` | `true` | Auto-ingest schemes on startup |
| `FORCE_RECREATE_COLLECTION` | `false` | Force recreate Qdrant collection |
| `QDRANT_MODE` | `local` | `local` or `cloud` |
| `QDRANT_LOCAL_PATH` | `./qdrant_data` | Local Qdrant storage path |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL (for cloud mode) |
| `QDRANT_API_KEY` | - | Qdrant API key (for cloud mode) |
| `API_BASE_URL` | `http://127.0.0.1:8000` | Base URL for API calls |

### Data Files

- `scheme.json`: Contains scraped government scheme data
- `checkpoints/chat_sessions.json`: Stores chat session history
- `qdrant_data/`: Local Qdrant vector database storage

## Development

### Running Tests

```bash
# Add tests as needed
```

### Code Structure

```
GovAssist-RAG/
├── main.py                 # FastAPI server
├── streamlit_app.py        # Streamlit UI
├── rag_pipeline.py         # Main RAG orchestration
├── embed.py                # Embedding service
├── qdrant_db.py           # Vector database manager
├── groq_client.py         # LLM client
├── checkpointer.py        # Session management
├── scrape.py              # Web scraper
├── scheme.json            # Scheme data
├── requirements.txt       # Python dependencies
└── .env                   # Environment variables
```

### Adding New Features

1. For new embedding models: Modify `EmbeddingService` in `embed.py`
2. For different LLMs: Update `GroqLLMClient` in `groq_client.py`
3. For additional categories: Add URLs to `CATEGORY_URLS` in `scrape.py`
4. For new API endpoints: Add routes in `main.py`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source. Please check the license file for details.

## Disclaimer

This project scrapes data from government websites for informational purposes. Please verify scheme details and eligibility from official sources before applying. The developers are not responsible for any inaccuracies in the scraped data.</content>
<parameter name="filePath">c:\Users\LAKSHMAN\GovAssist-RAG\README.md