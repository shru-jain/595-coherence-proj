import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import random
import torch
import argparse
import os
import json
import random
import re
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import login
load_dotenv()
login(token=os.getenv("HF_TOKEN"))

def load_model(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    use_cache=False,
    load_in_4bit=True,
    quant_type="nf4",
    compute_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # Configuration for quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # For sequence classification, we use AutoModelForSequenceClassification
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=low_cpu_mem_usage
    )
    
    # Set caching behavior
    model.config.use_cache = use_cache

    return model, tokenizer

def get_response_template():
    return f"\n### Answer:\n"


# A prompting formatting function
def create_prompt_instruction(num_sentences, permuted_sentences, label=None):
    if label is None:
        label = ""
    return (f"Below is a list of sentences labeled from 1 to {num_sentences} that have been permuted from their original order in a dialogue. "
            f"Using the numerical label for each sentence, write only the original order of the sentences as a list of numbers inside square brackets, separated by commas.\n\n"
            f"### Sentences:\n{permuted_sentences.rstrip()}\n\n"
            f"### Answer:\n{label}")


# Preprocessing function for formatting prompts
def formatting_prompts_func(example):
    inputs = []
    for i in range(len(example['items'])):
        num_sentences = example['items'][i]
        permuted_sentences = example['permuted_sentences'][i]
        label = example['correct_order'][i] if 'correct_order' in example else None
        prompt = create_prompt_instruction(num_sentences, permuted_sentences, label)
        inputs.append(prompt)
    return inputs

def load_data_with_collator(
    dataset_name,
    sample_size=None,
    tokenizer=None,
    response_template="### Answer:\n",
    mlm=False
):
    # Load the dataset split and select a sample of the data
    with open(dataset_name, 'r') as f:
        dataset = json.load(f)
        dataset = datasets.Dataset.from_list(dataset)
    if sample_size is not None:
        dataset = dataset.select(range(sample_size))

    # Initialize the data collator for completion-only language modeling
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=mlm
    )

    return dataset, collator

def setup_and_train_model(
    model,
    dataset,
    tokenizer,
    collator,
    formatting_prompts_func,
    run_name="qlora-sentence",
    output_dir="./results",
    num_train_epochs=2,
    max_seq_length=512,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=200,
    logging_steps=5,
    learning_rate=5e-5,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="linear",
    # report_to="wandb",
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    seed=595
):
    # Set up LoRA configuration
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout
    )

    # Define training arguments
    training_arguments = SFTConfig(
        run_name=run_name,
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        do_eval=False,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        fp16=True,
        max_grad_norm=max_grad_norm,
        warmup_ratio=warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=lr_scheduler_type,
        seed=seed,
        #report_to=report_to,
    )

    # Initialize the Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    # Ensure 'norm' layers are in float32 precision
    # This stablize the training
    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    # Train the model
    train_result = trainer.train()
    trainer.save_model(output_dir=output_dir)
    tokenizer.save_pretrained(output_dir)
    trainer.save_state()
    return trainer, model

def test(model, tokenizer, dataset, create_prompt_func, prediction_save='mistral_predictions.torch'):
    predictions, labels = [], []

    for example in dataset:
        # Create the prompt for the current example
        num_sentences = example['items']
        permuted_sentences = example['permuted_sentences']
        prompt = create_prompt_func(num_sentences, permuted_sentences, None)

        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate the answer
        outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)

        # Decode the generated output to get the predicted answer
        predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        pattern = r"\[\d+(?:, \d+)*\]"
        predicted_answer = re.findall(pattern=pattern, string=predicted_answer)

        # Convert the predicted answer to str
        try:
            predicted_label = predicted_answer[-1]
        except:
            predicted_label = None
        # Ground truth label is either 1 or 2 (assumed to be 0-indexed)
        predictions.append(predicted_label)
        labels.append(str(example['correct_order']))
    # Calculate accuracy
    predictions = np.array(predictions)
    labels = np.array(labels)
    accuracy = np.sum(predictions == labels) / len(predictions)

    print(f'Test Accuracy: {accuracy}')

    # Save predictions to a file
    torch.save(predictions, prediction_save)

def main(params):
    # Load the model and tokenizer
    model, tokenizer = load_model(params.model)
    # Get the response template for the task
    response_template = get_response_template()
    response_template = tokenizer.encode(response_template, add_special_tokens=False)[2:]
    # Load training data and the associated collator
    dataset_train, collator = load_data_with_collator(params.dataset_train, tokenizer=tokenizer, sample_size=None, response_template=response_template)
    
    # Set up and train the model with LoRA
    trainer, model = setup_and_train_model(
        model=model,
        dataset=dataset_train,
        tokenizer=tokenizer,
        collator=collator,
        formatting_prompts_func=formatting_prompts_func,
        output_dir=params.output_dir,
        num_train_epochs=params.num_train_epochs,
        per_device_train_batch_size=params.per_device_train_batch_size,
        gradient_accumulation_steps=params.gradient_accumulation_steps,
        optim=params.optim,
        save_steps=params.save_steps,
        logging_steps=params.logging_steps,
        learning_rate=params.learning_rate,
        max_grad_norm=params.max_grad_norm,
        warmup_ratio=params.warmup_ratio,
        lr_scheduler_type=params.lr_scheduler_type,
        #report_to=params.report_to,
        r=params.r,
        lora_alpha=params.lora_alpha,
        target_modules=params.target_modules,
        lora_dropout=params.lora_dropout,
        seed=params.seed,
    )


    dataset_val, _ = load_data_with_collator(params.dataset_test, tokenizer=tokenizer, sample_size=None, response_template=response_template)
    
    test(model, tokenizer, dataset_val, create_prompt_instruction)
    
    

if __name__ == "__main__":
    num_train_epochs = 3
    max_seq_length = 1024
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 4
    save_steps = 2000
    logging_steps = 500
    learning_rate = 5e-5
    max_grad_norm = 0.3
    warmup_ratio = 0.03
    r = 64
    lora_alpha = 16
    target_modules = [
        "q_proj", "v_proj", "k_proj", "o_proj"
    ]
    lora_dropout = 0.1

    optim = "paged_adamw_32bit"
    lr_scheduler_type = "linear"
    # report_to = "wandb"
    seed=595
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


    parser = argparse.ArgumentParser(description="Finetune LLaMA-based model for Sentence Reconstruction Task")
    parser.add_argument("--dataset_train", type=str, default="datasets/permuted-train-data-newlines.json", help="Train Dataset name")
    parser.add_argument("--dataset_test", type=str, default="datasets/permuted-test-data-newlines.json", help="Test Dataset name")
    parser.add_argument("--output_dir", type=str, default="./results", help="Checkpoint Output")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Pretrained model name")
    parser.add_argument("--num_train_epochs", type=int, default=num_train_epochs, help="Number of training epochs")
    parser.add_argument("--max_seq_length", type=int, default=max_seq_length, help="Maximum sequence length for inputs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=per_device_train_batch_size, help="Batch size per device for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=gradient_accumulation_steps, help="Number of gradient accumulation steps")
    parser.add_argument("--optim", type=str, default=optim, help="Optimizer type")
    parser.add_argument("--save_steps", type=int, default=save_steps, help="Number of steps between saves")
    parser.add_argument("--logging_steps", type=int, default=logging_steps, help="Number of steps between logging updates")
    parser.add_argument("--learning_rate", type=float, default=learning_rate, help="Learning rate for AdamW optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=max_grad_norm, help="Max gradient norm for clipping")
    parser.add_argument("--warmup_ratio", type=float, default=warmup_ratio, help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--lr_scheduler_type", type=str, default=lr_scheduler_type, help="Learning rate scheduler type")
    # parser.add_argument("--report_to", type=str, default=report_to, help="Reporting platform (e.g., wandb)")
    parser.add_argument("--r", type=int, default=r, help="LoRA rank parameter")
    parser.add_argument("--lora_alpha", type=int, default=lora_alpha, help="Alpha parameter for LoRA")
    parser.add_argument("--target_modules", type=list, default=target_modules, help="Target modules for LoRA")
    parser.add_argument("--lora_dropout", type=float, default=lora_dropout, help="Dropout rate for LoRA")
    parser.add_argument("--seed", type=int, default=seed, help="Random seed for reproducibility")

    # Parse arguments and run the main function
    params, unknown = parser.parse_known_args()
    model = main(params)
    
#LOAD MODEL
#tokenizer = AutoTokenizer.from_pretrained(output_dir)
#model = AutoModelForCausalLM.from_pretrained(output_dir, load_in_4bit=True, device_map="auto")
