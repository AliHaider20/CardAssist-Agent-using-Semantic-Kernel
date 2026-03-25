# app.py
import asyncio
import os
import streamlit as st
from dotenv import load_dotenv
import faiss
import numpy as np
import tempfile

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import torch

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.functions import kernel_function

from typing import Annotated, List

# Load environment variables
load_dotenv()

search_api = os.getenv("AI_SEARCH_API")
search_endpoint = "https://cardassist.search.windows.net"
ai_foundry_api = os.getenv("AI_FOUNDRY_MODEL_API")
llm_endpoint = os.getenv("LLM_ENDPOINT")

device = "mps" if torch.backends.mps.is_available() else "cpu"

embedding_service = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device)

st.set_page_config(page_title="💳 Credit Card Assistant", layout="wide")
st.title("💳 Credit Card Assistant")

# Session state initialization
for key, default in {
    "kernel": None,
    "chat_history": [],
    "faiss_index": None,
    "docs": [],
    "loading": False,
    "error": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

system_prompt = "You're a helpful assistant that can answer questions about credit card management, including activating and deactivating cards, and providing information about card features and account management. Make sure you provide accurate and well structured information based on the provided context."

class CreditCardPlugin:
    def __init__(self, faiss_index, docs):
        self.index = faiss_index
        self.docs = docs

    @kernel_function(description="Deactivate a credit card")
    async def deactivate_card(self, card_number: Annotated[str, "Card number to deactivate"]):
        return f"Credit card {card_number} has been deactivated."

    @kernel_function(description="Activate a credit card")
    async def activate_card(self, card_number: Annotated[str, "Card number to activate"]):
        return f"Credit card {card_number} has been activated."

    @kernel_function(description="Retrieve relevant card info using RAG")
    async def rag_query(self, query: Annotated[str, "Query for RAG"]):
        if self.index is None or not self.docs:
            return "No PDF data available for RAG search."
        query_embedding = embedding_service.encode([query]).astype("float32")
        distances, indices = self.index.search(query_embedding, k=5)
        relevant_chunks = [self.docs[i] for i in indices[0]]
        context = "\n".join(relevant_chunks)
        return f"{context}\n\nUser Query: {query}"

def setup_kernel(faiss_index=None, docs=None):
    kernel = Kernel()
    kernel.add_service(AzureChatCompletion(deployment_name="gpt-4o-mini", endpoint=llm_endpoint, api_key=ai_foundry_api))
    kernel.add_plugin(CreditCardPlugin(faiss_index, docs), plugin_name="CreditCard")
    return kernel

async def load_and_process_pdf_async(pdf_path: str) -> List[str]:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = MarkdownTextSplitter(chunk_size=300, chunk_overlap=30)
    return [doc.page_content for doc in splitter.split_documents(documents)]

def generate_embeddings(docs: List[str]) -> np.ndarray:
    return np.array(embedding_service.encode(docs)).astype("float32")

def create_faiss_index(embeddings: np.ndarray):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

async def process_message(message: str) -> str:
    kernel: Kernel = st.session_state.kernel
    if not kernel:
        # If kernel is not ready, initialize minimal kernel (only for activation/deactivation)
        kernel = setup_kernel()
        st.session_state.kernel = kernel

    try:
        args = KernelArguments(
            settings=AzureChatPromptExecutionSettings(
                function_choice_behavior=FunctionChoiceBehavior.Auto(),
                top_p=0.9, temperature=0,
            )
        )
        final_prompt = f"{system_prompt}\n\n{message}"
        response = await kernel.invoke_prompt(final_prompt, arguments=args)
        return response.value[0].content
    except Exception as e:
        return f"Error: {str(e)}"

async def handle_pdf_upload(file):
    st.session_state.loading = True
    try:
        if file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file.read())
                docs = await load_and_process_pdf_async(temp_file.name)
                embeddings = generate_embeddings(docs)
                index = create_faiss_index(embeddings)
                st.session_state.update({
                    "docs": docs,
                    "faiss_index": index,
                    "kernel": setup_kernel(index, docs)
                })
                st.success("PDF loaded, now you can ask quesions about card activation, deactivation, or information about cards!")
        else:
            st.warning("Please upload a PDF.")
    except Exception as e:
        st.error(str(e))
    finally:
        st.session_state.loading = False

# Sidebar PDF Upload
with st.sidebar:
    st.header("📄 Upload PDF Guide")
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
    if st.button("🔄 Load PDF"):
        if uploaded_pdf:
            asyncio.run(handle_pdf_upload(uploaded_pdf))
        else:
            st.warning("Upload a file first.")

# Chat UI
st.divider()
user_input = st.chat_input("Ask a question about card activation, deactivation, or info...")
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.spinner("Thinking..."):
        response = asyncio.run(process_message(user_input))
        st.session_state.chat_history.append(("assistant", response))

# Show chat history
for role, msg in st.session_state.chat_history:
    st.chat_message(role).markdown(msg)

st.divider()
st.subheader("🔧 Quick Actions")

card_number = st.text_input("Enter Card Number", key="card_number_input")
col1, col2 = st.columns(2)

with col1:
    if st.button("✅ Activate Card"):
        if card_number:
            with st.spinner("Activating card..."):
                response = asyncio.run(process_message(f"Activate card {card_number}"))
                st.success(f"Card {card_number} activated successfully! ✅")
        else:
            st.warning("Please enter a card number.")

with col2:
    if st.button("🛑 Deactivate Card"):
        if card_number:
            with st.spinner("Deactivating card..."):
                response = asyncio.run(process_message(f"Deactivate card {card_number}"))
                st.success(f"Card {card_number} deactivated successfully! 🛑")
        else:
            st.warning("Please enter a card number.")