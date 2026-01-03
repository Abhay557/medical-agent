import os
import torch
import pymongo
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer

# --- 1. Configuration ---
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "rag"
COLLECTION_NAME = "knowledgebase"

# Paths
base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_model_id = "./Qwen-Medical-0.5B" # The folder you downloaded

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device.upper()}")

# --- 2. Load Memory (MongoDB) ---
print("Connecting to Memory...")
try:
    client = pymongo.MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]
    embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("Memory Connected.")
except Exception as e:
    print(f"MongoDB Error: {e}")
    exit()

# --- 3. Load Brain (Qwen + Adapter) ---
print("Loading Model... (This is fast!)")
try:
    # Load Base Qwen
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    # Load YOUR Medical Adapter
    model = PeftModel.from_pretrained(base_model, adapter_model_id)
    model.eval()
    print("Medical AI Brain Loaded Successfully.")
except Exception as e:
    print(f"Model Loading Error: {e}")
    print("Check if the 'Qwen-Medical-0.5B' folder is in the right place.")
    exit()

# --- 4. RAG Logic ---
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

# --- 5. Chat Loop ---
print("\n" + "="*50)
print("MICRO MEDICAL AI READY")
print("Powered by Qwen-0.5B + Your Custom Adapter")
print("Type 'exit' to quit.")
print("="*50)

while True:
    user_query = input("\nMedical Question > ")
    if user_query.lower() in ['exit', 'quit']:
        break
    
    # 1. Retrieve
    facts = retrieve_context(user_query)
    context_str = "\n\n".join(facts) if facts else "No specific records found."
    
    if facts:
        print(f"   > Found {len(facts)} relevant medical notes.")

    # 2. Construct Prompt using Chat Template
    # This instructs the model to use the context
    messages = [
        {"role": "system", "content": "You are a helpful medical assistant. Use the context provided to answer the question."},
        {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {user_query}"}
    ]
    
    # Apply Qwen's specific formatting
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 3. Generate
    model_inputs = tokenizer([text_input], return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True
        )
    
    # 4. Decode
    # We strip the input tokens to show only the new answer
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"\nAI Diagnosis: {response}")