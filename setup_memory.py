import os
import pymongo
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

# 1. Load Secrets
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI: raise ValueError("MONGO_URI not found in .env file. Please create one!")

# 2. Settings
DB_NAME = "rag"
COLLECTION_NAME = "knowledgebase"
DATASET_ID = "FreedomIntelligence/medical-o1-reasoning-SFT"

# 3. Connect to DB
print("Connecting to MongoDB...")
client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Check if data exists
#if collection.count_documents({}) > 0:
  #  print("Database already contains data! Skipping upload.")
  #  exit()

# 4. Load Embedding Model
print("Loading embedding model...")
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 5. Load Data & Upload
print(f"Downloading dataset {DATASET_ID}...")
dataset = load_dataset(DATASET_ID, 'en', split="train")
# Select first 1000 for speed
batch_data = dataset.select(range(19704))

documents = []
print("Processing and uploading...")
for i, entry in enumerate(batch_data):
    q = entry.get('Question', entry.get('question', ''))
    a = entry.get('Response', entry.get('answer', ''))
    text_content = f"Question: {q}\nAnswer: {a}"
    
    doc = {
        "text": text_content,
        "embedding": embed_model.encode(text_content).tolist(),
        "source": "medical_sft"
    }
    documents.append(doc)
    if i % 100 == 0: print(f"Processed {i}...", end="\r")

collection.insert_many(documents)
print(f"\nSuccessfully uploaded {len(documents)} medical records to MongoDB!")
print("REMINDER: Ensure your Vector Search Index 'vector_index' is created in MongoDB Atlas.")