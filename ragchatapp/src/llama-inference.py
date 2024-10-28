# Load the fine-tuned model for inference
from peft import PeftModel
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

model_id = "meta-llama/Llama-3.2-3B-Instruct"
peft_model_path = "./peft-qLoRA-llama3.2B-dolly15k"

peft_model_base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

peft_model = PeftModel.from_pretrained(
    peft_model_base, 
    peft_model_path,  # Use the correct path for your trained model
    torch_dtype=torch.bfloat16,
    is_trainable=False
)

def generate_response(instruction):
    # Set model to evaluation mode
    peft_model.eval()
    
    # Construct the prompt using only the instruction
    prompt = f"Instruction: {instruction}\nResponse: "
    
    # Tokenize with specified padding
    inputs = tokenizer(prompt, return_tensors="pt", padding="longest")
    input_ids = inputs.input_ids.to(peft_model.device)
    attention_mask = inputs.attention_mask.to(peft_model.device)
    
    # Generate the response
    output = peft_model.base_model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=256,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id  # Ensure pad token ID is set correctly
    )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Test the function in a loop until the user types "exit"
while True:
    instruction = input("User instruction (type 'exit' to quit): ")
    
    if instruction.lower() == 'exit':
        print("Exiting the program.")
        break  # Exit the loop if user types 'exit'
    
    response = generate_response(instruction)
    print("Response:", response)
