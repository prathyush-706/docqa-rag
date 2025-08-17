# Document Q&A RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system that enables intelligent question answering from documents using semantic search, vector embeddings, and Google's Gemini AI.

## 🚀 Features

- **Document Ingestion**: Supports PDF and text files with intelligent text chunking
- **Semantic Search**: Uses sentence transformers and FAISS for efficient vector similarity search
- **AI-Powered Answers**: Leverages Google Gemini 2.5 Flash for context-aware responses
- **Conversation Memory**: Maintains context across multiple user interactions
- **Optional Reranking**: Cross-encoder reranking for improved retrieval quality
- **Flexible Architecture**: Modular design with configurable embedding models and parameters

## 🏗️ Architecture

The system follows a three-stage pipeline:

1. **Document Processing**: Text extraction, chunking, and embedding generation
2. **Retrieval**: Vector similarity search with optional reranking
3. **Generation**: Context-aware answer generation using conversation history

## 📋 Requirements

- Python 3.9+
- Google AI API key
- Virtual environment (recommended)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd docqa_poc
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv docqa
   source docqa/bin/activate  # On Windows: docqa\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create .env file
   echo "GOOGLE_API_KEY=your_api_key_here" > .env
   ```

## 📚 Usage

### Basic Usage

```python
from doc_qa import DocumentQA

# Initialize the system
qa_system = DocumentQA()

# Ingest a document
qa_system.ingest("path/to/your/document.pdf")

# Ask questions
answer = qa_system.query("What is the main topic of this document?")
print(answer)
```

### Interactive Mode

Run the main script for an interactive Q&A session:

```bash
python doc_qa.py
```

Commands:
- Type your question to get an answer
- Type `clear` to clear conversation history
- Type `quit` to exit

### Advanced Configuration

```python
# Custom embedding model and reranker
qa_system = DocumentQA(
    embedder_name="all-mpnet-base-v2",  # Different embedding model
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"  # Enable reranking
)

# Custom retrieval parameters
answer = qa_system.query(
    question="Your question here",
    k=15,  # Number of chunks to retrieve
    use_rerank=True,  # Enable reranking
    max_history=10  # Conversation memory length
)
```

## 🔧 Configuration

### Embedding Models

The system supports various sentence transformer models:
- `all-MiniLM-L6-v2` (default, fast, 384 dimensions)
- `all-mpnet-base-v2` (higher quality, 768 dimensions)
- Any other sentence transformer model from Hugging Face

### Reranking Models

Optional cross-encoder models for improved retrieval:
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (default)
- `cross-encoder/ms-marco-MiniLM-L-12-v2` (higher quality)

### Chunking Parameters

Customize text chunking in `doc_ingest.py`:
- `max_tokens`: Maximum tokens per chunk (default: 500)
- `overlap`: Overlap between chunks (default: 100)

## 📁 Project Structure

```
docqa_poc/
├── doc_qa.py              # Main DocumentQA class
├── doc_ingest.py          # Document loading and chunking
├── embedding.py            # Vector indexing and search
├── reranker.py            # Optional reranking functionality
├── generate_answer.py      # AI answer generation
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔍 How It Works

1. **Document Ingestion**: Documents are loaded and split into overlapping chunks
2. **Embedding Generation**: Each chunk is converted to a vector using sentence transformers
3. **Indexing**: Vectors are stored in a FAISS index for fast similarity search
4. **Query Processing**: User questions are embedded and used to find relevant chunks
5. **Reranking** (optional): Retrieved chunks are reranked using cross-encoders
6. **Answer Generation**: Relevant chunks and conversation history are sent to Gemini AI
7. **Response**: AI generates context-aware answers based on retrieved information

## 🎯 Use Cases

- **Research Papers**: Extract insights and answer specific questions
- **Legal Documents**: Find relevant clauses and legal information
- **Technical Manuals**: Get quick answers to implementation questions
- **Academic Materials**: Study assistance and concept clarification
- **Business Reports**: Extract key metrics and insights