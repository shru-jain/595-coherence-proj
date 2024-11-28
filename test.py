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

def create_prompt_instruction(num_sentences, permuted_sentences, label=None):
    if label is None:
        label = ""
    return (f"Below is a list of sentences labeled from 1 to {num_sentences} that have been permuted from their original order in a dialogue. "
            f"Using the numerical label for each sentence, write only the original order of the sentences as a list of numbers inside square brackets, separated by commas.\n\n"
            f"### Sentences:\n{permuted_sentences.rstrip()}\n\n"
            f"### Answer:\n{label}")

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
    tokenizer = AutoTokenizer.from_pretrained(params.model_dir)
    model = AutoModelForCausalLM.from_pretrained(params.model_dir, load_in_4bit=True, device_map="auto")
    dataset_test, _ = load_data_with_collator(params.dataset_test, tokenizer=tokenizer, sample_size=None, response_template=response_template)
    test(model, tokenizer, dataset_test, create_prompt_instruction)

if __name__ == "__main__":
    seed=595
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


    parser = argparse.ArgumentParser(description="Finetune LLaMA-based model for Sentence Reconstruction Task")
    parser.add_argument("--dataset_test", type=str, default="datasets/permuted-test-data-newlines.json", help="Test Dataset name")
    parser.add_argument("--model_dir", type=str, default="./results", help="Checkpoint Output")