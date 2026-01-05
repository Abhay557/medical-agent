# MedicalAgent: Full-Stack RAG Microservice with Quantized LLM
This repository contains the source code and documentation for MedicalAgent, a production-grade AI microservice designed to serve a specialized medical Large Language Model (LLM) in resource-constrained environments.

The project demonstrates a complete end-to-end pipeline: from fine-tuning a base model on medical data to deploying it as a scalable microservice with Retrieval-Augmented Generation (RAG).

[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/Abhay557/medical-agent) | [Model on Kaggle](https://www.kaggle.com/models/abhay557/qwen-0-5b-medical-adapter)

### Key Engineering Achievements
- Microservice Backend: Engineered a high-performance FastAPI service to serve a 4-bit quantized Qwen-0.5B model. Utilized custom lifespan state management to initialize heavy GPU artifacts and database connections strictly at startup, ensuring zero-latency overhead for runtime requests.
- Hybrid RAG Pipeline: Architected a dual-retrieval mechanism that dynamically routes queries between a persistent MongoDB Atlas vector index (for verified medical literature) and an in-memory session store (for real-time user document analysis), effectively minimizing hallucinations.
- Custom Frontend: Developed a bespoke, lightweight HTML/Tailwind/JS client supporting Server-Sent Events (SSE) for real-time token streaming and granular parameter control (temperature, max tokens).
- Deployment: Dockerized the entire application stack to ensure environment parity across local development (RTX 3050) and cloud orchestration platforms.

### Tech Stack
- Language: Python 3.10
- Core Frameworks: PyTorch, FastAPI, Hugging Face Transformers
- LLM Engineering: QLoRA, PEFT, TRL, BitsAndBytes (4-bit Quantization)
- Data & RAG: MongoDB Atlas (Vector Search), Sentence-Transformers
- Frontend: HTML5, Tailwind CSS, Vanilla JavaScript
- DevOps: Docker, Uvicorn


## The Journey: Training the Model
Before building the application, I engineered the model itself. The goal was to transform a general-purpose language model into a medical specialist.

### 1. Base Model Selection

I chose **Qwen2.5-0.5B-Instruct** for its high performance-to-size ratio. Its small footprint allows for fast inference on consumer hardware while retaining strong reasoning capabilities.

### 2. Fine-Tuning (SFT)

I fine-tuned the model on the `FreedomIntelligence/medical-o1-reasoning-SFT` dataset.

**Technique**: QLoRA (Quantized Low-Rank Adaptation)
**Config**: Rank = 8, Alpha = 16, 4-bit Quantization

**Outcome**: The model learned to adopt a professional medical persona and follow complex diagnostic instructions.

### 3. Alignment (DPO)

I aligned the model using **Direct Preference Optimization (DPO)** on the `Intel/orca_dpo_pairs` dataset to encourage helpful and safe responses, reducing the likelihood of toxic or incorrect medical advice.

---

## Installation & Usage

### Prerequisites

* Python 3.10+
* NVIDIA GPU (Recommended) or CPU
* MongoDB Atlas Account

### 1. Setup Environment

```bash
git clone https://github.com/your-username/MedicalAgent.git
cd MedicalAgent
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Secrets

Create a `.env` file in the root directory:

```env
MONGO_URI="your_mongodb_connection_string"
```

### 3. Populate Knowledge Base

Run the setup script to download the medical dataset and upload embeddings to MongoDB:

```bash
python scripts/setup_memory.py
```

### 4. Run the Application

Start the FastAPI server:

```bash
uvicorn api:app --reload
```

Open your browser and navigate to:

```
http://localhost:8000
```

to use the chat interface.

---

##  Disclaimer

This AI is for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health pr
