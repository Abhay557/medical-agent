import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig

# --- Configuration ---
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
new_model_name = "Qwen-Medical-0.5B"
dataset_id = "FreedomIntelligence/medical-o1-reasoning-SFT"

# --- 1. Load Tokenizer & Model ---
print(f"Loading {model_id}...")

# Quantization Config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, # Base model does math in FP16
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Load Model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="sdpa",
    torch_dtype=torch.float16 
)

# Force Config cleanup
model.config.torch_dtype = torch.float16
model.config.use_cache = False 

# Prepare for training
model = prepare_model_for_kbit_training(model)

# --- 2. Load Dataset ---
print(f"Loading dataset {dataset_id}...")
dataset = load_dataset(dataset_id, 'en', split="train")

def format_chat_template(row):
    messages = [
        {"role": "user", "content": row["Question"]},
        {"role": "assistant", "content": row["Response"]}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}

dataset = dataset.map(format_chat_template)
# Optional: Use a subset for a quick test run
# dataset = dataset.select(range(200))

# --- 3. LoRA Configuration ---
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)

print("Applying LoRA adapters...")
model = get_peft_model(model, peft_config)

# --- CRITICAL FIX FOR WINDOWS ---
# We cast the TRAINABLE parameters (adapters) to Float32.
# This ensures stability since we are disabling the Trainer's AMP.
print("Casting trainable adapters to Float32 for stability...")
for name, param in model.named_parameters():
    if param.requires_grad:
        param.data = param.data.to(torch.float32)

model.print_trainable_parameters()

# --- 4. Training Configuration ---
sft_config = SFTConfig(
    output_dir="./results_medical",
    num_train_epochs=1,
    per_device_train_batch_size=1, # Keep strict Batch Size 1 for 3050 VRAM
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=5,
    save_steps=50,
    
    # !!! THE FIX !!!
    # We DISABLE the Trainer's native Mixed Precision.
    # Because we manually quantized the base model and cast adapters to FP32,
    # we don't need the Trainer to interfere. This bypasses the error.
    fp16=False, 
    bf16=False,
    
    max_length=512,
    packing=False,
    dataset_text_field="text",
    optim="paged_adamw_8bit"
)

# --- 5. Start Training ---
print("Starting Training...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
    processing_class=tokenizer,
)

trainer.train()

# --- 6. Save ---
print("Saving model...")
trainer.model.save_pretrained(new_model_name)
tokenizer.save_pretrained(new_model_name)
print(f"Done! Medical Adapters saved to: {new_model_name}")