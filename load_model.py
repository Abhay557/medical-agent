import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Configuration ---
# We are using the "Instruct" version, which means it's already trained to chat.
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {model_id} on {device.upper()}...")

# --- 1. Load Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_id)

# --- 2. Load Model ---
# Theory: We load the pre-trained weights (the "brain structure").
# We use torch.float16 (half precision) to make it faster and use less memory.
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Model loaded successfully!")

# --- 3. The Test ---
# Let's verify it understands English before we teach it medicine.
prompt = "Explain simply: Why is the sky blue?"

# Format the prompt so the model knows it's a chat
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Convert text to numbers (tensors) and send to GPU
inputs = tokenizer([text_input], return_tensors="pt").to(device)

print(f"\nThinking about: '{prompt}'...\n")

# Generate response
generated_ids = model.generate(
    **inputs,
    max_new_tokens=100
)

# Convert numbers back to text
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Print only the AI's part (removing the prompt)
# Simple string splitting to clean up the output
print("--- AI RESPONSE ---")
print(response.split("assistant")[-1].strip())
print("-------------------")