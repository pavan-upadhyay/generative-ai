import os
import logging
from opensearchpy import OpenSearch
import time  # For simulating delay
from llama_index.core.callbacks.global_handlers import set_global_handler
import ads
from ads.llm import GenerativeAIEmbeddings, GenerativeAI
from config import (
    VERBOSE,
    EMBED_MODEL_TYPE,
    EMBED_MODEL,
    GEN_MODEL,
    RERANKER_MODEL,
    ADD_PHX_TRACING,
    PHX_PORT,
    PHX_HOST,
    COMPARTMENT_OCID,
    ENDPOINT,
    COHERE_API_KEY,
    OPENSEARCH_END_POINT, OCI_OPENSEARCH_USERNAME, OCI_OPENSEARCH_PASSWORD,
    OCI_OPENSEARCH_VERIFY_CERTS
)
from oci_utils import load_oci_config, print_configuration
import streamlit as st

# Initialize OpenSearch client
client = OpenSearch(
    OPENSEARCH_END_POINT,
    http_auth=(OCI_OPENSEARCH_USERNAME, OCI_OPENSEARCH_PASSWORD),
    verify_certs=OCI_OPENSEARCH_VERIFY_CERTS,
)

# Load OCI configuration
oci_config = load_oci_config()
api_keys_config = ads.auth.api_keys(oci_config)

if ADD_PHX_TRACING:
    import phoenix as px

# Configure logger
logger = logging.getLogger("ConsoleLogger")

# Initialize Phoenix tracing if enabled
if ADD_PHX_TRACING:
    os.environ["PHOENIX_PORT"] = PHX_PORT
    os.environ["PHOENIX_HOST"] = PHX_HOST
    px.launch_app()
    set_global_handler("arize_phoenix")


# Function to create a large language model (LLM)
def create_llm(auth=None):
    model_list = ["OCI", "LLAMA"]

    # Validate model choice
    if GEN_MODEL not in model_list:
        raise ValueError(f"The value {GEN_MODEL} is not supported. Choose a value in {model_list} for the GenAI model.")

    llm = None
    if GEN_MODEL in ["OCI", "LLAMA"]:
        assert auth is not None
        common_oci_params = {
            "auth": auth,
            "compartment_id": COMPARTMENT_OCID,
            "max_tokens": st.session_state['max_tokens'],
            "temperature": st.session_state['temperature'],
            "truncate": "END",
            "client_kwargs": {"service_endpoint": ENDPOINT},
        }
        model_name = "cohere.command-r-plus" if GEN_MODEL == "OCI" else "meta.llama-3-70b-instruct"
        llm = GenerativeAI(name=model_name, **common_oci_params)

    assert llm is not None
    return llm


# Function to create an embedding model
def create_embedding_model(auth=None):
    model_list = ["OCI"]

    # Validate embedding model choice
    if EMBED_MODEL_TYPE not in model_list:
        raise ValueError(
            f"The value {EMBED_MODEL_TYPE} is not supported. Choose a value in {model_list} for the model.")

    embed_model = None
    if EMBED_MODEL_TYPE == "OCI":
        embed_model = GenerativeAIEmbeddings(
            auth=auth,
            compartment_id=COMPARTMENT_OCID,
            model=EMBED_MODEL,
            truncate="END",
            client_kwargs={"service_endpoint": ENDPOINT},
        )
    return embed_model


# Get embeddings for a search query
def get_embeddings(text, embed_model):
    embedding_response = embed_model.embed_documents([text])
    embedding = embedding_response[0]
    return embedding


# Function to search query in opensearch
def search_opensearch(query, index_name, top_k):
    logger.info("Calling search_opensearch()...")
    print_configuration()

    # Initialize Phoenix tracing if enabled
    if ADD_PHX_TRACING:
        set_global_handler("arize_phoenix")

    embed_model = create_embedding_model(auth=api_keys_config)
    query_vector = get_embeddings(query, embed_model)

    search_body = {
        "size": top_k,
        "_source": ["id", "body"],
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": top_k
                }
            }
        }
    }

    response = client.search(index=index_name, body=search_body)
    # Format search results for LLM
    documents = response['hits']['hits']
    context = "\n\n".join([hit["_source"]["body"] for hit in documents])

    # Create LLM instance
    llm = create_llm(auth=api_keys_config)

    # Generate final response using LLM
    prompt = f"Based on the following documents, provide a detailed response to the query: '{query}'\n\nDocuments:\n{context}"
    llm_response = llm.generate(prompts=[prompt])

    # Print the type and attributes of the LLM response
    print("LLM Response Type:", type(llm_response))
    print("LLM Response Attributes:", dir(llm_response))

    # Extract text from the LLM response
    try:
        # Assuming llm_response has a method or attribute to get generations
        if hasattr(llm_response, 'generations'):
            generations = llm_response.generations
            if generations:
                generated_text = generations[0][0].text  # Access the text field
            else:
                generated_text = "No response generated."
        else:
            generated_text = "No 'generations' attribute in response."
    except Exception as e:
        generated_text = f"Error extracting response: {e}"

    return generated_text


# LLM chat function
def llm_chat(question):
    logger.info("Calling llm_chat()...")
    # Create LLM
    llm = create_llm(auth=api_keys_config)
    # Initialize the QA chain
    response = llm.generate(prompts=[question])
    logger.info("Response generated without RAG.")
    # Extract text from the LLM response
    try:
        # Assuming llm_response has a method or attribute to get generations
        if hasattr(response, 'generations'):
            generations = response.generations
            if generations:
                generated_text = generations[0][0].text  # Access the text field
            else:
                generated_text = "No response generated."
        else:
            generated_text = "No 'generations' attribute in response."
    except Exception as e:
        generated_text = f"Error extracting response: {e}"
    return generated_text
