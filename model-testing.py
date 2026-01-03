import os
import torch
import pymongo
import time
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
adapter_model_id = "./Qwen-Medical-0.5B"

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. Test Questions ---
# Add questions that you know exist in your dataset
test_questions = [
    "What are the clinical symptoms of cardiac arrhythmia?",
    "How should I treat a patient with Type 2 Diabetes?",
    "What are the side effects of Lisinopril?",
    "A patient has sudden chest pain and shortness of breath. Diagnosis?",
    "Explain the mechanism of action for Metformin."
]

# --- 3. Load Resources ---
print("Loading Evaluation Resources...")

# Database
try:
    client = pymongo.MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]
    embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
except Exception as e:
    print(f"DB Error: {e}")
    exit()

# Model
print(f"Loading Model on {device}...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, torch_dtype=torch.float16, device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_model_id)
model.eval()

# --- 4. Logic Functions ---
def retrieve(query):
    vector = embed_model.encode(query).tolist()
    pipeline = [
        {"$vectorSearch": {
            "index": "vector_index", "path": "embedding", "queryVector": vector,
            "numCandidates": 100, "limit": 2
        }},
        {"$project": {"_id": 0, "text": 1, "score": {"$meta": "vectorSearchScore"}}}
    ]
    return list(collection.aggregate(pipeline))

def generate(prompt, context):
    messages = [
        {"role": "system", "content": "You are a helpful medical assistant. Use the context to answer."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"}
    ]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text_input], return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# --- 5. Run Evaluation ---
print("\n" + "="*50)
print(f"STARTING EVALUATION ON {len(test_questions)} QUESTIONS")
print("="*50 + "\n")

results_log = []

for i, question in enumerate(test_questions):
    print(f"[{i+1}/{len(test_questions)}] Testing: {question}")
    
    # 1. Measure Retrieval
    start_time = time.time()
    docs = retrieve(question)
    retrieval_time = time.time() - start_time
    
    context_str = "\n".join([d['text'] for d in docs]) if docs else "No context found."
    top_score = docs[0].get('score', 0) if docs else 0.0
    
    # 2. Measure Generation
    answer = generate(question, context_str)
    
    # 3. Log
    log_entry = f"""
    Q: {question}
    --------------------------------------------------
    RETRIEVAL ({retrieval_time:.2f}s | Top Score: {top_score:.4f}):
    {context_str[:200]}... [truncated]
    
    AI ANSWER:
    {answer}
    ==================================================
    """
    print(log_entry)
    results_log.append(log_entry)

# Save to file
with open("evaluation_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(results_log))

print(f"\nEvaluation Complete! Results saved to 'evaluation_results.txt'")