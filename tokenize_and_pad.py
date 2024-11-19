# For Programming Problem 2
#
# Please implement the functions below, and feel free to use the any library.

from tqdm import tqdm
import argparse
import random
from typing import Optional, Union
import json

import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification,


def placeholder():
    raise NotImplementedError("This function has not been implemented yet.")

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

SEED = 595
set_seed(SEED)
device = torch.device("cpu") # TODO: can add gpu here


# TODO: remove this comment https://chatgpt.com/share/673c1c92-56c8-8007-b250-49073c7b409c

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
AutoModelForSequenceClassification.from_pretrained(model_name)

def process_and_pad(entry, max_length=512, max_label_length=15):
    # Tokenize the permuted sentences
    inputs = tokenizer(
        entry["permuted_sentences"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    # Prepare labels: pad correct_order with -100 up to max_length
    correct_order = entry["correct_order"]
    padded_labels = correct_order + [-100] * (max_label_length - len(correct_order))
    labels = torch.tensor(padded_labels).unsqueeze(0)  # Add batch dimension

    # Include labels in the inputs dictionary
    inputs["labels"] = labels
    return inputs


def process_json_data(json_data, max_length=512):
    
    processed_data = []
    for entry in json_data:
        processed_entry = process_and_pad(entry, max_length=max_length)
        processed_data.append(processed_entry)
    return processed_data

def open_files(filename):
    with open(filename, 'r') as file:
        json_data = json.load(file)
    process_json_data(json_data=json_data)



def main():
    open_files('dataset_name.json')


if __name__ == "__main__":
    main()
