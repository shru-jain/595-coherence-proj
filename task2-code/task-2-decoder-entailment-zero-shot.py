import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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
import sys
load_dotenv()
login(token=os.getenv("HF_TOKEN"))

def load_model(
    model_name="mistralai/Mistral-7B-v0.3",
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

def create_prompt_instruction(sentences, hypothesis, label=None):
    if label is None:
        label = ""
    dialogue = ""
    for sentence in sentences:
        dialogue += f"{sentence}\n"
    return (f"Below is a dialogue paired with a hypothesis. Your task is to decide whether the hypothesis is directly supported or implied by the dialogue. "
            f"If yes output only 1. If no output 0.\n\n"
            f"### Dialogue:\n{dialogue.rstrip()}\n\n"
            f'### Hypothesis:\n{hypothesis}\n\n'
            f"### Answer:\n{label}")

def load_data(
    dataset_name,
    sample_size=None,
):
    # Load the dataset split and select a sample of the data
    with open(dataset_name, 'r') as f:
        dataset = json.load(f)
        dataset = datasets.Dataset.from_list(dataset)
    if sample_size is not None:
        dataset = dataset.select(range(sample_size))

    return dataset

def test(model, tokenizer, dataset, create_prompt_func, mode='reconstructed', prediction_save='mistral_predictions.torch'):
    predictions_entailments, labels_entailments = [], []

    for example in dataset:
        # Create the prompt for the current example
        hypothesis = example['hypothesis']
        entailment = 1 if example['entailment'] else 0
        sentences = ""
        if mode == "reconstructed":
            sentences = example['reconstructed_order']
        elif mode == "permuted":
            sentences = example['permuted_sentences']
        elif mode == "original":
            sentences = example['original']
            for idx, _ in enumerate(sentences):
                sentences[idx] = f"{idx + 1}) {sentences[idx]}"
        else:
            sys.exit(f"Invalid mode {mode} selected. Only takes in 'reconstruted', 'permuted', or 'original'")
            
        prompt = create_prompt_func(sentences, hypothesis, None)

        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate the answer
        outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)

        # Decode the generated output to get the predicted answer
        predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        print(predicted_answer)

        # Convert the predicted answer to str
        predicted_entailment = None
        try:
            pattern = r"### Answer:\s*([\d])"
            predicted_entailment = re.findall(pattern=pattern, string=predicted_answer)
            predicted_entailment = int(predicted_entailment[-1])
        except:
            predicted_entailment = None
        # Ground truth label is either 1 or 2 (assumed to be 0-indexed)
        if predicted_entailment != 0 and predicted_entailment != 1:
            print(predicted_answer)
            predicted_entailment = None
        predictions_entailments.append(predicted_entailment)
        labels_entailments.append(entailment)
    # Calculate accuracy
    predictions_entailments = np.array(predictions_entailments)
    labels_entailments = np.array(labels_entailments)
    entailment_accuracy = np.sum(predictions_entailments == labels_entailments) / len(predictions_entailments)
    print(f'Entailment Accuracy: {entailment_accuracy}')

    # Save predictions to a file
    torch.save(predictions_entailments, prediction_save)

def main(params):
    # Load the model and tokenizer
    model, tokenizer = load_model(params.model)
    model.eval()
    dataset_test = load_data(params.dataset_test, sample_size=None)
    test(model, tokenizer, dataset_test, create_prompt_instruction, params.mode)

if __name__ == "__main__":
    seed=595
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


    parser = argparse.ArgumentParser(description="Test pre-trained models")
    parser.add_argument("--dataset_test", type=str, default="datasets/output.json", help="Test Dataset name")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.3", help="Pretrained model name")
    parser.add_argument("--mode", type=str, default="reconstructed", help="Either use reconstructed order, permuted order, or original order")
    # Possible flags "reconstructed", "permuted", "original"
    params, unknown = parser.parse_known_args()
    main(params)