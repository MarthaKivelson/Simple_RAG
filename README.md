# Simple RAG Chatbot

A simple Retrieval-Augmented Generation (RAG) chatbot that answers questions based on your local documents using OpenAI embeddings, FAISS vector search, and an OpenAI chat model.

## Features

- Supports PDF, DOCX, and TXT files.
- Splits documents into overlapping text chunks.
- Generates embeddings using `text-embedding-3-small`.
- Stores and searches embeddings using FAISS.
- Retrieves the top 3 relevant chunks for a question.
- Generates answers using `gpt-4.1-mini`.
- Saves the vector index and chunk metadata for reuse.

## Project Structure

```text
Simple_RAG/
├── data/                  # Documents to query
├── vector_store/           # Generated FAISS index and chunks
│   ├── faiss.index
│   └── chunks.json
├── rag.py                  # Main RAG application
├── template.py             # Supporting script
├── requirements.txt         # Python dependencies
├── .env                    # OpenAI API key (not committed)
├── .gitignore
└── README.md
```

## Requirements

- Python 3.10 or later
- An OpenAI API key
- Git (optional)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/MarthaKivelson/Simple_RAG.git
cd Simple_RAG
```

### 2. Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure your API key

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
```

Replace the placeholder with your actual API key. Never commit your `.env` file to GitHub.

## Add Your Documents

Place your supported documents inside the `data/` directory.

Supported formats:

- `.pdf`
- `.docx`
- `.txt`

The application extracts their text and divides it into chunks before creating embeddings.

## Run the Chatbot

Run the following command from the project root:

```bash
python rag.py
```

On the first run, the application loads documents, creates text chunks, generates embeddings, builds the FAISS index, and saves the index and metadata in `vector_store/`.

On subsequent runs, it loads the saved vector store when both files are available and their chunk counts match.

## Ask Questions

Once the chatbot starts, enter a question about your documents:

```text
Simple RAG chatbot is ready
Ask a question about your document or type exit/quit to quit

You: What is Uttam's background?

Assistant:
[Answer generated from the retrieved document context]

Source: Uttam_Pattar_Resume.pdf
```

To exit, type:

```text
exit
```

or

```text
quit
```

## Configuration

The main settings are defined in `rag.py`:

| Setting | Value | Description |
|---|---|---|
| Embedding model | `text-embedding-3-small` | Converts text into vectors |
| Chat model | `gpt-4.1-mini` | Generates answers |
| Chunk size | `1000` characters | Maximum chunk length |
| Chunk overlap | `150` characters | Preserves context between chunks |
| Top K | `3` | Number of chunks retrieved |
| Embedding batch size | `100` | Texts processed per embedding request |
| Temperature | `0.2` | Controls response randomness |
| Max tokens | `700` | Maximum response tokens |

## How It Works

1. **Document loading:** Extracts text from PDF, DOCX, and TXT files.
2. **Chunking:** Splits extracted text into overlapping sections.
3. **Embedding generation:** Converts each chunk into a numerical vector.
4. **Vector storage:** Stores normalized vectors in a FAISS inner-product index.
5. **Retrieval:** Converts the question into an embedding and retrieves the most similar chunks.
6. **Answer generation:** Sends the question and retrieved context to the OpenAI chat model.
7. **Source reporting:** Includes the filenames of the documents used for the answer.

## Important Notes

- OpenAI API usage may incur charges.
- Scanned PDFs may require OCR if they do not contain extractable text.
- The chatbot can only answer reliably when the relevant information is present in the retrieved context.
- The `vector_store/` directory contains generated files and can be rebuilt from the documents.
- Keep `.env`, API keys, and private documents out of public repositories.

## License

Add a license if you intend to distribute this project publicly.