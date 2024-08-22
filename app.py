import time
import logging
from pathlib import Path
import streamlit as st
import chat_engine
from ingest_data_opens import process_and_index_pdf_page_by_page, save_uploaded_file
from config import (
    STREAM_CHAT,
    INDEX_NAME
)


# Initialize session state
def initialize_session_state():
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 1024
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.0
    if "top_k" not in st.session_state:
        st.session_state.top_k = 3
    if "top_n" not in st.session_state:
        st.session_state.top_n = 3
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "question_count" not in st.session_state:
        st.session_state.question_count = 0
    if "enable_rag" not in st.session_state:
        st.session_state.enable_rag = True
    if "similarity" not in st.session_state:
        st.session_state.similarity = 0.5


initialize_session_state()

st.logo("Oracle-Logo.png")

# Set the configuration for the Streamlit app
st.set_page_config(page_title="HR Chatbot Assistant", layout="wide", page_icon=":robot_face:")

upload_dir = Path("data/unprocessed")
upload_dir.mkdir(parents=True, exist_ok=True)

# Title for the sidebar
st.markdown("<h1 style='text-align: center;'>HR Chatbot Assistant</h1>", unsafe_allow_html=True)


# Modify the conversation reset function to conditionally create the chat engine
def reset_conversation():
    st.session_state.messages = []
    st.session_state.question_count = 0


# Function to handle form submission
def handle_form_submission():
    st.session_state.update({
        "max_tokens": st.session_state.max_tokens,
        "temperature": st.session_state.temperature,
        "top_k": st.session_state.top_k,
        "top_n": st.session_state.top_n,
        "enable_rag": st.session_state.enable_rag,
        "similarity": st.session_state.similarity
    })
    reset_conversation()


# Streamlit sidebar form for adjusting model parameters
def render_sidebar_forms():
    with st.sidebar.form(key="input-form"):
        st.session_state.enable_rag = st.checkbox('Enable RAG', value=True, label_visibility="visible")
        st.session_state.max_tokens = st.number_input('Maximum Tokens', min_value=512, max_value=1024, step=25,
                                                      value=st.session_state.max_tokens)
        st.session_state.temperature = st.number_input('Temperature', min_value=0.0, max_value=1.0, step=0.1,
                                                       value=st.session_state.temperature)
        st.session_state.similarity = st.number_input('Similarity Score', min_value=0.0, max_value=1.0, step=0.05,
                                                      value=st.session_state.similarity)
        st.session_state.top_k = st.slider("TOP_K", 1, 10, step=1, value=st.session_state.top_k)
        st.session_state.top_n = st.slider("TOP_N", 1, 10, step=1, value=st.session_state.top_n)
        submitted_sidebar = st.form_submit_button("Submit", type="primary", on_click=handle_form_submission,
                                                  use_container_width=True)
    return submitted_sidebar


# Render the sidebar forms
render_sidebar_forms()


# Display chat messages in the Streamlit app
def display_chat_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


# Function to handle non-streaming output
def no_stream_output(response):
    if st.session_state.enable_rag:
        output = response
        st.markdown(output)
    else:
        output = response
    return output


# Function to handle streaming output
def stream_output(response):
    text_placeholder = st.empty()
    output = ""
    for text in response:
        output += text
        text_placeholder.markdown(output, unsafe_allow_html=True)
    return output


# Main function to run the Streamlit app
def main():
    _, c1 = st.columns([5, 2])
    c1.button("Clear Chat History", type="primary", on_click=reset_conversation)

    with st.sidebar.form(key="file-uploader-form", clear_on_submit=True):
        uploaded_files = st.file_uploader("Upload PDF Document", accept_multiple_files=True, type=['pdf'], label_visibility="collapsed")
        upload_button = st.form_submit_button("Upload", type="primary", use_container_width=True, on_click=reset_conversation)

    if upload_button and uploaded_files:
        if not isinstance(uploaded_files, list):
            uploaded_files = [uploaded_files]
        logging.info("Uploading file")
        for uploaded_file in uploaded_files:
            # Save the uploaded file to a directory
            file_path = save_uploaded_file(uploaded_file, upload_dir)
            # Call the function to process and index the PDF
            process_and_index_pdf_page_by_page(file_path, INDEX_NAME)
            st.success(f"Uploaded and indexed {uploaded_file.name} successfully!")

    # Configure logging
    logger = logging.getLogger("ConsoleLogger")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False

    # Initialize session state if not already done
    if "messages" not in st.session_state:
        reset_conversation()

    display_chat_messages()

    # Input for the user question
    question = st.chat_input("Hello, how can I help you?")
    if question:
        st.chat_message("user").markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        try:
            logger.info("Calling RAG chain..")
            logger.info(
                f"top_k= {st.session_state.top_k},max_tokens= {st.session_state.max_tokens}, temperature= {st.session_state.temperature},top_n= {st.session_state.top_n},enable_rag= {st.session_state.enable_rag},similarity = {st.session_state.similarity}")

            with st.spinner("Waiting..."):
                time_start = time.time()
                st.session_state.question_count += 1
                logger.info("")
                logger.info(f"Question no. {st.session_state.question_count} is {question}")

                # Generate response using the chat engine
                if st.session_state.enable_rag:
                    if STREAM_CHAT:
                        logger.info("response from STREAM_CHAT.")
                        response = chat_engine.search_opensearch(question, INDEX_NAME, st.session_state.top_k)
                    else:
                        logger.info("response from without STREAM_CHAT.")
                        response = chat_engine.search_opensearch(question, INDEX_NAME, st.session_state.top_k)
                else:
                    logger.info("response without RAG..")
                    response = chat_engine.llm_chat(question)

                # # Debug the response
                print(f"Response type: {type(response)}")
                if isinstance(response, dict):
                    print(f"Response keys: {response.keys()}")

                time_elapsed = time.time() - time_start
                logger.info(f"Elapsed time: {round(time_elapsed, 1)} sec.")

                # Display response from the assistant
                with st.chat_message("assistant"):
                    if st.session_state.enable_rag and STREAM_CHAT:
                        output = stream_output(response)
                    else:
                        output = no_stream_output(response)

                st.session_state.messages.append({"role": "assistant", "content": output})

        except Exception as e:
            logger.error("An error occurred: " + str(e))
            st.error("An error occurred: " + str(e))

        # Force Streamlit to immediately update the UI
        if not st.session_state.enable_rag:
            st.rerun()


# Entry point for the script
if __name__ == "__main__":
    main()
