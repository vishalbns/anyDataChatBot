from datasets import load_dataset
import time
import evaluate
import pandas as pd
import numpy as np

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from peft import get_peft_model, LoraConfig, TaskType

from tqdm import tqdm

'''

DATA PREP

'''
ds = load_dataset("ccdv/patent-classification", "patent")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)
# Apply tokenization to the entire dataset
encoded_dataset = ds.map(tokenize_function, batched=True)

# Check the structure of the dataset
print(encoded_dataset)

# Split the dataset into train and eval sets
train_dataset = encoded_dataset["validation"]
eval_dataset = encoded_dataset["test"]


'''

MODEL PREP

'''

# Check if MPS (Metal Performance Shaders) is available for M1 Pro GPU
#device = "cpu" if torch.backends.mps.is_available() else "cpu"
device = "cpu"

model_id = "meta-llama/Llama-3.2-1B"
config = AutoConfig.from_pretrained(model_id)

# Change the architecture to LlamaForSequenceClassification
config.architectures = ["LlamaForSequenceClassification"]

# Add a fallback for rope_scaling attribute, if it exists
if hasattr(config, "rope_scaling") and "type" not in config.rope_scaling:
    config.rope_scaling["type"] = "default"

num_labels = len(ds["train"].features["label"].names)

# Set num_labels in the config
config.num_labels = num_labels

# Load the sequence classification model
original_model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    config=config,  # Pass the config with num_labels and updated architecture
    torch_dtype=torch.float32,  # Use float16 precision for memory efficiency
).to(device)

# Ensure model is on the correct device
original_model = original_model.to(device)

# Set the padding token ID if not already set
original_model.config.pad_token_id = tokenizer.pad_token_id

'''

PRINT NUMBER OF PARAMETERS TO TRAIN

'''

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


'''

LoRA

'''

from peft import get_peft_model, LoraConfig, TaskType
# Set up LoRA configuration
lora_config = LoraConfig(
    r=8,            # Rank of the low-rank matrices
    lora_alpha=16,  # Scaling factor for LoRA layers
    target_modules=["q_proj", "v_proj"],  # LoRA applies to these layers
    lora_dropout=0.1,  # Dropout in LoRA layers
    bias="none",  # Bias term handling
)

# Apply LoRA to the original model and move it to the appropriate device
peft_model = get_peft_model(original_model, lora_config)
peft_model.to(device)  # Ensure it's moved to the correct device
peft_model.config.pad_token_id = tokenizer.pad_token_id
print(print_number_of_trainable_model_parameters(peft_model))

'''

SET MODEL TRAIN PARAMETERS

'''

from transformers import Trainer, TrainingArguments

# Include evaluation during training
training_args = TrainingArguments(
    output_dir="./results",          # Output directory
    num_train_epochs=1,              # Number of training epochs
    per_device_train_batch_size=1,   # Batch size for training
    per_device_eval_batch_size=1,    # Batch size for evaluation
    warmup_steps=500,                # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # Strength of weight decay
    logging_dir="./logs",            # Directory for storing logs
    logging_steps=10,                # Frequency of logging
    report_to="tensorboard",         # Log to TensorBoard
    eval_strategy="steps",     # Evaluate at the end of each epoch
    do_train=True,
    do_eval=True,
    save_strategy="steps",  # Save at every step
    save_steps=100,
    fp16=True,
    learning_rate=1e-3,
)

from sklearn.metrics import accuracy_score

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    return accuracy_score(labels, preds)


trainer = Trainer(
    model=peft_model,                    # The pre-trained model
    args=training_args,                  # Training arguments
    train_dataset=train_dataset,         # Training dataset
    eval_dataset=eval_dataset,           # Evaluation dataset
    tokenizer=tokenizer,                 # Tokenizer
    compute_metrics=compute_metrics,     # Custom metrics
)

# Force model to use CPU
device = torch.device('cpu')

# Load model and move to the correct device
original_model = original_model.to(device)
peft_model = peft_model.to(device)

# Check which device it's on
print(f"Model is on {peft_model.device}")


'''

TRAIN MODEL

'''

tqdm(trainer.train())

'''

SAVE MODEL

'''

# Save the final model
trainer.save_model("./final_model")