import os
import json
from transformers import AutoTokenizer, AutoModel
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct, Distance
from PyPDF2 import PdfReader
from langchain.text_splitter import TokenTextSplitter
from dotenv import load_dotenv


load_dotenv(dotenv_path="development.env")

# Constants
CHUNK_SIZE = 500  # Number of tokens
CHUNK_OVERLAP = 50  # Overlap between chunks (in tokens)
VECTOR_SIZE = 1024  # Embedding vector size
OUTPUT_FILE_METADATA = "chunks_with_metadata.json"
OUTPUT_FILE_EMBEDDINGS = "chunks_with_embeddings.json"

API_URL = os.getenv("QDRANT_API_URL")
API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/e5-large")


# Step 1: Extract Text from All PDFs in Current Directory
def extract_text_from_pdfs(directory="."):
    pdf_files = [file for file in os.listdir(directory) if file.endswith(".pdf")]
    extracted_texts = []

    for pdf_file in pdf_files:
        print(f"Extracting text from {pdf_file}...")
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        extracted_texts.append({"text": text, "source": pdf_file})

    return extracted_texts


# Step 2: Chunk the Text Based on Tokens with Metadata
def chunk_text_with_metadata(extracted_texts):
    token_splitter = TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks_with_metadata = []

    for item in extracted_texts:
        text = item["text"]
        source = item["source"]
        chunks = token_splitter.split_text(text)
        for chunk in chunks:
            chunks_with_metadata.append({"chunk": chunk, "source": source})
    
    return chunks_with_metadata


# Step 3: Save Chunks to JSON File
def save_chunks_to_json(chunks, output_file):
    with open(output_file, "w") as file:
        json.dump(chunks, file, indent=4)
    print(f"Chunks with metadata saved to {output_file}")


# Step 4: Load Chunks from JSON File
def load_chunks(json_file):
    with open(json_file, "r") as file:
        chunks = json.load(file)
    return chunks


# Step 5: Generate Embeddings for Chunks
def generate_embeddings(chunks):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    for i, chunk in enumerate(chunks):
        text = chunk["chunk"]

        # Tokenize the text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
        #generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        chunk["embedding"] = embeddings
        chunk["id"] = i
    return chunks


# Step 6: Initialize Qdrant Client and Create Collection
def initialize_qdrant(api_url, api_key, collection_name, vector_size):
    # Initialize Qdrant Client with API
    client = QdrantClient(url=api_url, api_key=api_key)

    # Check if the collection exists; if not, create it
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"Collection '{collection_name}' created.")
    else:
        print(f"Collection '{collection_name}' already exists.")
    return client


# Step 7: Index Chunks with Embeddings into Qdrant
def index_chunks_with_embeddings(client, collection_name, chunks):
    points = []
    for i, chunk in enumerate(chunks):
        # Prepare the point structure for Qdrant
        points.append(PointStruct(
            id=i,  # Use an integer ID
            vector=chunk["embedding"],  # Use the precomputed embeddings
            payload={"source": chunk["source"], "text": chunk["chunk"]}  # Optional metadata
        ))

    # Upsert all points into the collection
    client.upsert(collection_name=collection_name, points=points)
    print(f"Indexed {len(points)} chunks into the Qdrant collection '{collection_name}'.")


# Main Function
def main():
    # Extract text from PDFs
    print("Extracting text from PDF files...")
    extracted_texts = extract_text_from_pdfs()

    # Chunk the text with metadata
    print("Chunking the extracted text based on tokens...")
    chunks_with_metadata = chunk_text_with_metadata(extracted_texts)
    print(f"Total chunks created: {len(chunks_with_metadata)}")

    # Save the chunks to a JSON file
    save_chunks_to_json(chunks_with_metadata, OUTPUT_FILE_METADATA)

    # Load chunks from JSON
    print("Loading chunks from JSON file...")
    chunks = load_chunks(OUTPUT_FILE_METADATA)

    # Generate embeddings for each chunk
    print("Generating embeddings for chunks...")
    chunks_with_embeddings = generate_embeddings(chunks)

    # Save updated chunks back to JSON
    print("Saving updated chunks with embeddings...")
    save_chunks_to_json(chunks_with_embeddings, OUTPUT_FILE_EMBEDDINGS)

    # Initialize Qdrant client and connect to collection
    print("Initializing Qdrant client and connecting to collection...")
    client = initialize_qdrant(API_URL, API_KEY, COLLECTION_NAME, VECTOR_SIZE)

    # Index chunks with embeddings into Qdrant
    print("Indexing chunks with embeddings into Qdrant...")
    index_chunks_with_embeddings(client, COLLECTION_NAME, chunks_with_embeddings)


if __name__ == "__main__":
    main()
