
# Optional if u want to make model for intelgience as per preference dataset

import torch
import gc
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from trl import DPOTrainer, DPOConfig

# --- 1. Memory Cleanup ---
# Crucial for running back-to-back training on local GPUs
gc.collect()
torch.cuda.empty_cache()

# --- 2. Configuration ---
model_id = "google/gemma-2b"
# We start with the SFT adapter we just trained in the previous step
adapter_model_id = "gemma-2b-sft-alpaca" 
new_model_name = "gemma-2b-dpo-aligned"

# --- 3. Load Base Model (4-bit) ---
print(f"Loading Base Model: {model_id}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
model.config.use_cache = False

# --- 4. Load SFT Adapter ---
# We load the skills learned in the SFT step
print(f"Loading SFT Adapter: {adapter_model_id}...")
try:
    model = PeftModel.from_pretrained(model, adapter_model_id, is_trainable=True)
except Exception as e:
    print(f"Error loading SFT adapter: {e}")
    print("Did you run train_sft.py successfully?")
    exit()

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# --- 5. Load Preference Dataset ---
print("Loading Preference Dataset (Intel/orca_dpo_pairs)...")
dataset = load_dataset("Intel/orca_dpo_pairs", split="train")

def format_dpo_data(example):
    return {
        "prompt": f"### Instruction:\n{example['question']}\n\n### Response:\n",
        "chosen": example['chosen'],
        "rejected": example['rejected'],
    }

dataset = dataset.map(format_dpo_data, remove_columns=dataset.column_names)

# --- 6. LoRA Config for DPO ---
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# --- 7. DPO Configuration (Speed Run) ---
dpo_config = DPOConfig(
    output_dir="./results_dpo",
    max_steps=60,            # Speed run: 60 steps (approx 5-10 mins)
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=4,
    learning_rate=5e-5,      # Lower LR for DPO
    logging_steps=10,
    save_steps=20,
    fp16=True,               # RTX 3050 supports FP16
    max_grad_norm=0.3,
    warmup_ratio=0.1,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    max_length=512,
    max_prompt_length=256,
)

# --- 8. Start Training ---
print("Starting DPO Training...")
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None, # Trainer creates a reference model automatically
    args=dpo_config,
    train_dataset=dataset,
    processing_class=tokenizer, 
    peft_config=peft_config,
)

dpo_trainer.train()

# --- 9. Save the Final Adapter ---
print("DPO Training Finished!")
print(f"Saving aligned adapter to {new_model_name}...")
dpo_trainer.model.save_pretrained(new_model_name)
print("Done! You now have a fully aligned AI.")