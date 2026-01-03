import os
import torch
import pymongo
import streamlit as st
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer

# --- 1. Configuration & Setup ---
st.set_page_config(page_title="Micro Medical AI", page_icon="🧬", layout="centered")

# Load Secrets
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "rag"
COLLECTION_NAME = "knowledgebase"

if not MONGO_URI:
    st.error("Error: MONGO_URI not found. Please check your .env file.")
    st.stop()

# --- 2. Initialize System (Cached for Speed) ---
# We use st.cache_resource so we don't reload the 2GB model every time you click a button
@st.cache_resource
def load_system():
    print("Loading System...")
    
    # A. Connect to Memory (MongoDB)
    try:
        client = pymongo.MongoClient(MONGO_URI)
        collection = client[DB_NAME][COLLECTION_NAME]
        embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    except Exception as e:
        return None, None, None, None, f"MongoDB Error: {e}"

    # B. Load Brain (Qwen + SFT Adapter)
    base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    adapter_model_id = "./Qwen-Medical-0.5B" # Pointing to your local SFT folder

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        model = PeftModel.from_pretrained(base_model, adapter_model_id)
        model.eval()
    except Exception as e:
        return None, None, None, None, f"Model Error: {e}"

    return collection, embed_model, model, tokenizer, None

# Load everything once
collection, embed_model, model, tokenizer, error = load_system()

if error:
    st.error(error)
    st.stop()

# --- 3. RAG Retrieval Logic ---
def retrieve_context(query):
    vector = embed_model.encode(query).tolist()
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index", 
                "path": "embedding",
                "queryVector": vector,
                "numCandidates": 100,
                "limit": 3
            }
        },
        {"$project": {"_id": 0, "text": 1}}
    ]
    results = list(collection.aggregate(pipeline))
    return [doc['text'] for doc in results]

# --- 4. Chat Interface ---
st.title("🧬 Dr. Qwen (0.5B)")
st.caption("A Micro-Medical AI running locally on your RTX 3050")

# Session State for History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a medical question..."):
    # 1. Show User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Retrieve Context (RAG)
    with st.spinner("Accessing medical database..."):
        facts = retrieve_context(prompt)
        context_str = "\n\n".join(facts) if facts else "No specific records found."
        
        # Optional: Show context in an expander for debugging
        with st.expander("View Retrieved Context"):
            st.write(context_str)

    # 3. Generate Answer
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Construct Prompt using Qwen's Chat Template
        messages = [
            {"role": "system", "content": "You are a helpful medical assistant. Use the context provided to answer the question."},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {prompt}"}
        ]
        text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True
            )
        
        # Decode response (removing the input prompt)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        message_placeholder.markdown(response)
    
    # 4. Save History
    st.session_state.messages.append({"role": "assistant", "content": response})