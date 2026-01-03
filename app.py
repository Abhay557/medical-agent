import os
import torch
import pymongo
import streamlit as st
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pypdf import PdfReader

# --- 1. Configuration & Setup ---
st.set_page_config(page_title="Dr. Qwen (Medical RAG)", page_icon="🧬", layout="wide")

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "rag"
COLLECTION_NAME = "knowledgebase"

if not MONGO_URI:
    st.error("Error: MONGO_URI not found.")
    st.stop()

# --- 2. Initialize Connections (Cached) ---
@st.cache_resource
def load_system():
    print("Loading System...")
    try:
        # A. MongoDB
        client = pymongo.MongoClient(MONGO_URI)
        collection = client[DB_NAME][COLLECTION_NAME]
        
        # B. Embedding Model
        embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # C. Load Your Custom Qwen Brain
        base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        adapter_model_id = "./Qwen-Medical-0.5B" # Your local trained folder

        print(f"Loading Base Model: {base_model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        print(f"Loading Adapter: {adapter_model_id}...")
        model = PeftModel.from_pretrained(base_model, adapter_model_id)
        model.eval()
        
        return collection, embed_model, model, tokenizer, None

    except Exception as e:
        return None, None, None, None, str(e)

collection, embed_model, model, tokenizer, error = load_system()

if error:
    st.error(f"System Failed to Load: {error}")
    st.stop()

# --- 3. PDF Logic ---
def process_pdf(uploaded_file):
    """Reads PDF and creates in-memory embeddings."""
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    chunk_size = 500
    overlap = 50
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 50:
            chunks.append(chunk)
            
    if chunks:
        embeddings = embed_model.encode(chunks)
        return chunks, embeddings
    return [], None

# --- 4. RAG Logic (Hybrid) ---
def retrieve_context(query, pdf_chunks=None, pdf_embeddings=None):
    query_vector = embed_model.encode(query)
    found_facts = []

    # A. Search PDF
    if pdf_chunks and pdf_embeddings is not None:
        scores = util.cos_sim(query_vector, pdf_embeddings)[0]
        top_indices = np.argsort(scores)[-3:]
        for idx in reversed(top_indices):
            if scores[idx] > 0.3:
                found_facts.append(f"[PDF]: {pdf_chunks[idx]}")

    # B. Search MongoDB
    if len(found_facts) < 3:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index", 
                    "path": "embedding",
                    "queryVector": query_vector.tolist(),
                    "numCandidates": 100,
                    "limit": 3 - len(found_facts)
                }
            },
            {"$project": {"_id": 0, "text": 1}}
        ]
        results = list(collection.aggregate(pipeline))
        found_facts.extend([f"[DB]: {doc['text']}" for doc in results])
        
    return found_facts

# --- 5. UI Layout ---
st.title("🧬 Dr. Qwen (Medical RAG + PDF)")
st.caption("Powered by **Your Custom Qwen-0.5B** + MongoDB")

with st.sidebar:
    st.header("📄 Upload Report")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    pdf_chunks, pdf_embeddings = None, None
    if uploaded_file:
        with st.spinner("Reading PDF..."):
            pdf_chunks, pdf_embeddings = process_pdf(uploaded_file)
            st.success("PDF Loaded!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a medical question..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        # 1. Retrieve
        facts = retrieve_context(prompt, pdf_chunks, pdf_embeddings)
        context_str = "\n\n".join(facts) if facts else "No specific records found."
        
        with st.expander("🔍 Context Used"):
            st.code(context_str)

        # 2. Prompt Engineering (Qwen Chat Template)
        messages = [
            {"role": "system", "content": "You are a helpful medical assistant. Use the context provided to answer the question."},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {prompt}"}
        ]
        text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

        # 3. Generate (Local Inference)
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True
            )
        
        # 4. Decode
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})