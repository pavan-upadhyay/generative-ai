#!/usr/bin/env python3
import uuid
import time

import PyPDF2
import ads
from opensearchpy import OpenSearch

from chat_engine import create_embedding_model
from config import OPENSEARCH_END_POINT, INDEX_NAME, OCI_OPENSEARCH_USERNAME, \
    OCI_OPENSEARCH_PASSWORD, OCI_OPENSEARCH_VERIFY_CERTS
from oci_utils import load_oci_config

# Create the client with SSL/TLS and hostname verification disabled.
client = OpenSearch(
    OPENSEARCH_END_POINT,  # your OCI OpenSearch private endpoint
    http_auth=(OCI_OPENSEARCH_USERNAME, OCI_OPENSEARCH_PASSWORD),
    verify_certs=OCI_OPENSEARCH_VERIFY_CERTS,
)

# Load OCI configuration
oci_config = load_oci_config()
api_keys_config = ads.auth.api_keys(oci_config)


def save_uploaded_file(uploaded_file, upload_dir):
    file_path = upload_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


# Create index with appropriate settings and mappings
def create_index():
    settings = {
        "settings": {
            "index": {
                "knn": True,
            }
        },
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "page_number": {"type": "integer"},
                "body": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 1024,
                },
            }
        },
    }
    if not client.indices.exists(index=INDEX_NAME):
        response = client.indices.create(index=INDEX_NAME, body=settings)
        print("Index creation response:", response)
    else:
        print(f"Index {INDEX_NAME} already exists.")


# Extract text from PDF page by page
def extract_text_from_pdf_page_by_page(file_path):
    with open(file_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_number, page in enumerate(pdf_reader.pages):
            text = page.extract_text() or ''
            yield page_number + 1, text  # Yield the page number and text


# Get embeddings from text
def get_embeddings(text, embed_model):
    embedding_response = embed_model.embed_documents([text])
    embedding = embedding_response[0]  # Get the first embedding
    print("Embedding size:", len(embedding))  # Debug: Check embedding size
    return embedding


# Index document into OpenSearch
def index_document_to_opensearch(index_name, doc_id, page_number, body, embedding):
    document = {
        "id": doc_id,
        "page_number": page_number,
        "body": body,
        "embedding": embedding,
    }
    response = client.index(index=index_name, id=f"{doc_id}_{page_number}", body=document)
    return response


# Function to generate a unique document ID
def generate_unique_doc_id():
    return f"doc_{int(time.time())}_{uuid.uuid4()}"


# Main function to process and index PDF page by page
def process_and_index_pdf_page_by_page(pdf_path, index_name):
    create_index()
    doc_id = generate_unique_doc_id()
    embed_model = create_embedding_model(auth=api_keys_config)
    try:
        for page_number, text in extract_text_from_pdf_page_by_page(pdf_path):
            content_vector = get_embeddings(text, embed_model)
            response = index_document_to_opensearch(index_name, doc_id, page_number, body=text,
                                                    embedding=content_vector)
            print(f"Page {page_number} indexed:", response)
    except Exception as e:
        print(f"Error in uploading: {e}")
    print("Finished uploading pages!")
