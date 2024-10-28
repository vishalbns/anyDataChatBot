'''
# Install necessary packages
%pip install --upgrade pip
%pip install --disable-pip-version-check \
    torch==1.13.1 \
    torchdata==0.5.1 --quiet

%pip install \
    transformers==4.43.0 \
    datasets==2.11.0 \
    evaluate==0.4.0 \
    rouge_score==0.1.2 \
    loralib==0.1.1 \
    peft==0.3.0 --quiet
'''
from datasets import load_dataset
import time
import evaluate
import pandas as pd
import numpy as np

# Load the dataset
dataset = load_dataset("databricks/databricks-dolly-15k")
print("dataset loaded")

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

# Load model and tokenizer
model_id = "meta-llama/Llama-3.2-3B-Instruct"
# Load the model configuration
config = AutoConfig.from_pretrained(model_id)

# Ensure that the rope_scaling type is defined
if not hasattr(config, 'rope_scaling') or 'type' not in config.rope_scaling:
    # Set a default if rope_scaling is not present or type is missing
    config.rope_scaling = {"type": "llama3"}  # Adjust as necessary

config.rope_scaling = {
        "factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
        "type": "llama3"
    }

# Load the model
original_model = AutoModelForCausalLM.from_pretrained(model_id, config=config, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Function to print trainable parameters
def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

print(print_number_of_trainable_model_parameters(original_model))

# Define the tokenize function for conversational fine-tuning
def tokenize_function(example):
    # Construct the input prompt with instruction, context, and category
    prompt = f"Instruction: {example['instruction']}\nContext: {example['context']}\nCategory: {example['category']}\nResponse: "
    
    # Tokenize the prompt for input_ids
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, max_length=512, return_tensors="pt").input_ids[0]

    # Tokenize the response for labels
    example['labels'] = tokenizer(example['response'], padding="max_length", truncation=True, max_length=512, return_tensors="pt").input_ids[0]

    return example

# Map the tokenize function to the dataset splits
tokenized_datasets = dataset.map(tokenize_function)
print("dataset tokenized")

# Configure qLoRA
lora_config = LoraConfig(
    r=32, # Rank
    lora_alpha=32,
    #target_modules=["q", "v"],  # You can adjust this based on your model architecture
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM  # Task type for LLaMA
)

# Get the PEFT model with qLoRA
peft_model = get_peft_model(original_model, lora_config)
print(print_number_of_trainable_model_parameters(peft_model))

# Prepare output directory for saving model
output_dir = f'./peft-llama3.2B-dolly-training-{str(int(time.time()))}'

# Define training arguments
peft_training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    learning_rate=1e-3,  # Higher learning rate than full fine-tuning.
    num_train_epochs=1,
    logging_steps=1,
    max_steps=5,
    fp16=False,  # Enable mixed precision training
)

# Create the Trainer
peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=tokenized_datasets["train"],
)

# Start training
peft_trainer.train()
print("peft model trained")

# Save the fine-tuned model
peft_model_path = "./peft-qLoRA-llama3.2B-dolly15k"
peft_trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)
print("peft model saved")

# Load the fine-tuned model for inference
from peft import PeftModel

peft_model_base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id)

peft_model = PeftModel.from_pretrained(
    peft_model_base, 
    peft_model_path,  # Use the correct path for your trained model
    torch_dtype=torch.bfloat16,
    is_trainable=False
)

# Inference code with attention mask and pad token id
def generate_response(instruction):
    # Construct the prompt using only the instruction
    prompt = f"Instruction: {instruction}\nResponse: "
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(peft_model.device)
    attention_mask = inputs.attention_mask.to(peft_model.device)
    
    # Generate the response using the base model within peft_model
    output = peft_model.base_model.generate(
        input_ids,
        attention_mask=attention_mask,  # Add attention mask for reliable results
        max_new_tokens=256,  # Adjust the number of tokens to generate as needed
        do_sample=True,      # Enable sampling for diverse responses
        top_k=50,            # Top-k sampling
        top_p=0.95,          # Nucleus sampling
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id  # Set pad token ID
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example usage
instruction = "What are the benefits of using renewable energy?"
response = generate_response(instruction)
print(response)

