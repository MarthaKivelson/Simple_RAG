import json
import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from docx import Document
from pypdf import PdfReader
from openai import OpenAI


#settings
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
INDEX_PATH = VECTOR_STORE_DIR / "faiss.index"
CHUNK_PATH = VECTOR_STORE_DIR / "chunks.json"

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 3
EMBEDDING_BATCH_SIZE = 100

TEMPERATURE = 0.2
TOP_P = 0.7
MAX_TOKENS = 700

#client setup
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("OPENAI_API_KEY was not found")

client = OpenAI(api_key=api_key)

# Document loader

def load_documents():
    if not DATA_DIR:
        raise FileNotFoundError(f"Data folder could not be found here: {DATA_DIR}")

    documents = []
    
    for  file_path in sorted(DATA_DIR.rglob("*")):
        if not file_path.is_file():
            continue 

        suffix = file_path.suffix.lower()
        text = ""
        try:
            if suffix == ".txt":
                text = file_path.read_text(
                    encoding="utf-8",
                    errors="ignore"
                )
            elif suffix == ".docx":
                doc = Document(str(file_path))
                text = "\n".join(
                    para.text for para in doc.paragraphs
                    if para.text.strip()
                )
            elif suffix == ".pdf":
                reader = PdfReader(str(file_path))
                text = "\n".join(
                    page.extract_text() or ""
                    for page in reader.pages
                )
            else:
                continue

        except Exception as e:
            print(f"Warning: Could not read the file {file_path.name}: {e}")
            continue  

        text = text.strip()

        if text:
            documents.append({
                "source":str(file_path.relative_to(DATA_DIR)),
                "text": text
            })
            print(f"Loaded: {file_path.relative_to(DATA_DIR)}")
        else:
            print(f"Skipped because coundnt find anything in {file_path.relative_to(DATA_DIR)}")

    if not documents:
        raise ValueError("No readable text found in the folder")

    return documents

def split_into_chunks(documents):
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    for document in documents:
        text = document['text']

        for start in range(0, len(text), step):
            chunk_text = text[start:start+CHUNK_SIZE].strip()

            if chunk_text:
                chunks.append({
                    "source": document['source'],
                    "text": chunk_text
                })
    if not chunks:
        raise ValueError("chunking produced no text chunks")

    print(f"Created {len(chunks)} chunks")
    return chunks

#embeddings

def create_embeddings(texts):
    all_embeddings = []

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start:start+EMBEDDING_BATCH_SIZE]

        response = client.embeddings.create(
            model= EMBEDDING_MODEL,
            input=batch
        )

        batch_embeddings = [item.embedding for item in sorted(response.data, key= lambda item: item.index)]
        all_embeddings.extend(batch_embeddings)

    vectors = np.asarray(all_embeddings, dtype=np. float32)

    if vectors.ndim != 2 or len(vectors) != len(texts):
        raise RuntimeError("Embeddings did not match the input text count")

    #normalize so tht inner-product search behaves like cosine similarity
    faiss.normalize_L2(vectors)

    return vectors

#build & save the faiss
def build_store_vectors():
    document = load_documents()
    chunks = split_into_chunks(document)
    print("Creating embedding, may take a while")
    vectors = create_embeddings([chunk["text"] for chunk in chunks])

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index,str(INDEX_PATH))

    with CHUNK_PATH.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    print(f"Saved index to: {INDEX_PATH}")
    print(f"Saved chunks matadata to: {CHUNK_PATH} ")

    return index, chunks

def load_or_build_vector_store():
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    if INDEX_PATH.exists() and CHUNK_PATH.exists():
        print("Found saved vector store, loading it without rebuilding...")
        index = faiss.read_index(str(INDEX_PATH))

        with CHUNK_PATH.open("r", encoding="utf-8") as file:
            chunks = json.load(file)

        if index.ntotal != len(chunks):
            print("Saved index and chunks and matadata doesnot match, Rebuilding chunks...")
            return build_store_vectors()
        
        if index.ntotal == 0:
            print("Saved vector store is empty therefore, Rebulding...")
            return build_store_vectors()
        
        print(f"Loaded {len(chunks)} chunks from saved vectorDB")
        return index, chunks
    
    print("No complete saved vectorDB found, Building it now...")
    return build_store_vectors()

#retrieve revelent chunks
def retrieve_revelent_chunks(question, index, chunks):
    query_vector = create_embeddings([question])
    num_to_fetch = TOP_K
    
    scores, position = index.search(query_vector, num_to_fetch)
    results = []

    for score, pos in zip(scores[0], position[0]):
        if pos < 0:
            continue
        chunk = chunks[int(pos)]
        results.append({
            "source": chunk["source"],
            "text": chunk["text"],
            "score": float(score)
        })
    return results 

def answer_question(question, index, chunks):
    retrieved_chunks = retrieve_revelent_chunks(question, index, chunks)

    if not retrieved_chunks:
        return "Couldn't find relevent information"

    context_parts = []
    for number, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Source {number}: {chunk['source']} \n {chunk['text']}"
        )
    context = '\n\n---\n\n'.join(context_parts)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    """You are a helpful assesstant answering question about the
                    provided documents. Use the supplied content as your primary source.
                    If the content does not contain answer, say You could not find the answer in the provided documents.
                    Do not invent facts. Mention sources filenames when useful."""
                )
            },
            {
                "role": "user",
                "content": 
                (
                    f"Question: \n {question}\n\n"
                    f"Retrieved document context:\n {context}"
                )
            },
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS
    )

    answer = response.choices[0].message.content
    answer = answer.strip() if answer else "The model returned an empty answer" 

    source = []
    for chunk in retrieved_chunks:
        if chunk['source'] not in source:
            source.append(chunk['source'])

    return answer + "\n\n Source: " + ",".join(source) 


def main():
    try:
        index, chunks = load_or_build_vector_store()
    except Exception as e:
        print(f"Could not prepare the vector store {e}")
        raise
    print("Simple RAG chatbot is ready")
    print("Ask a question about your document or type exit/quit to quit")

    while True:
        try:
            question = input("You: ")
        except KeyboardInterrupt:
            print("Goodbyee")
            break

        if question.lower() in {"exit","quit"}:
            print("Goodbyee")
            break

        if not question:
            continue

        try:
            print("\n Assisstant: ")
            print(answer_question(question, index, chunks))
        except Exception as e:
            print(f"Sorry could not process your question due to {e}")

if __name__ == "__main__":
    main()




    