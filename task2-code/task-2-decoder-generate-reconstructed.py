import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, AutoPeftModelForCausalLM, PeftConfig, PeftModel
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

def compute_accuracy(pair_list):
    # binary_score, abs_dist_score
    length = len(pair_list)
    score = 0
    for a, b in pair_list:
        if a == b:
            score += 1
    return(float(score)/length)

def compute_partial_accuracy(pair_list):
    length = 0
    score = 0
    for a, b in pair_list:
        for i in range(len(a)):
            length += 1
            if a[i] == b[i]:
                score += 1
    return(float(score)/length)

def load_model(
    model_dir="./results",
    use_cache=False,
    load_in_4bit=True,
    quant_type="nf4",
    compute_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
):
    
    config = PeftConfig.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
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
        config.base_model_name_or_path,
        quantization_config=bnb_config,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=low_cpu_mem_usage
    )
    
    # Set caching behavior
    model.config.use_cache = use_cache
    
    model = PeftModel.from_pretrained(model, model_dir, config=config, low_cpu_mem_usage=low_cpu_mem_usage, use_safetensors=True)

    return model, tokenizer

def get_response_template():
    return f"\n### Answer:\n"

def create_prompt_instruction(num_sentences, permuted_sentences, label=None):
    if label is None:
        label = ""
    permuted_dialogue = ""
    for sentence in permuted_sentences:
        permuted_dialogue += f"{sentence}\n"
    return (f"Below is a list of sentences labeled from 1 to {num_sentences} that have been permuted from their original order in a dialogue. "
            f"Using the numerical label for each sentence, write only the original order of the sentences as a list of numbers inside square brackets, separated by commas.\n\n"
            f"### Sentences:\n{permuted_dialogue.rstrip()}\n\n"
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

def test(model, tokenizer, dataset, create_prompt_func, prediction_save='mistral_predictions.torch'):
    predictions, labels = [], []
    output_guesses = []
    for example in dataset:
        # Create the prompt for the current example
        num_sentences = example['items']
        permuted_sentences = example['permuted_sentences']
        prompt = create_prompt_func(num_sentences, permuted_sentences, None)
        
        dict_entry = {}
        dict_entry['items'] = num_sentences
        dict_entry['hypothesis'] = example['hypothesis']
        dict_entry['original'] = example['original']
        dict_entry['entailment'] = example['entailment']
        dict_entry['permuted_sentences'] = permuted_sentences
        dict_entry['correct_order'] = example['correct_order']

        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate the answer
        outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)

        # Decode the generated output to get the predicted answer
        predicted_ans = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        pattern = r"\[\d+(?:, \d+)*"
        predicted_answer = re.findall(pattern=pattern, string=predicted_ans)

        # Convert the predicted answer to str
        try:
            predicted_label = json.loads(predicted_answer[-1] + "]")
            predicted_label = predicted_label[:len(example['correct_order'])]
        except:
            predicted_label = [0] * len(example['correct_order'])
        dict_entry['reconstructed_order'] = []
        idx = 0
        for sentence in predicted_label:
            if sentence > len(dict_entry['permuted_sentences']) or sentence <= 0:
                continue
            sentence = dict_entry['permuted_sentences'][sentence - 1].lstrip("0123456789) ")
            dict_entry['reconstructed_order'].append(f"{idx + 1}) {sentence}")
            idx += 1
        output_guesses.append(dict_entry)
        predictions.append(predicted_label)
        labels.append(example['correct_order'])
        print(predicted_label)
    # Calculate accuracy
    pair_list = list(zip(predictions, labels))
    partial_accuracy = compute_partial_accuracy(pair_list)
    accuracy = compute_accuracy(pair_list)

    print(f'Order Accuracy: {accuracy}')
    print(f'Partial Accuracy: {partial_accuracy}')

    # Save predictions to a file
    torch.save(predictions, prediction_save)
    return output_guesses

def main(params):
    # Load the model and tokenizer
    model, tokenizer = load_model(params.model_dir)
    model.eval()
    dataset_test = load_data(params.dataset_test, sample_size=None)
    guesses = test(model, tokenizer, dataset_test, create_prompt_instruction)
    with open(f"datasets/output.json", 'w') as f:
        f.write(json.dumps(guesses, indent=4))

if __name__ == "__main__":
    seed=595
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


    parser = argparse.ArgumentParser(description="Test pre-trained models")
    parser.add_argument("--dataset_test", type=str, default="datasets/permuted-sled-test.json", help="Test Dataset name")
    parser.add_argument("--model_dir", type=str, default="./results", help="Model location")
    params, unknown = parser.parse_known_args()
    main(params)