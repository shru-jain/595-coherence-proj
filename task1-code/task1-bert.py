import os
import shutil
import sys
import argparse
import random
import json

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import get_scheduler, BertTokenizer, BertForTokenClassification
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

SEED = 595
set_seed(SEED)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu") # TODO: check that this is good


tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = BertForTokenClassification.from_pretrained(
    "bert-base-cased", num_labels=21  # Set max number of dialogues
)
model.to(device)

# get the data from files
def preprocess_data(filename):
    dialogues = []
    labels = []
    hyp = []
    ent = []

    with open(f"datasets/{filename}, "r") as f:
        data = json.load(f)
        for item in data:
            dialogues.append(item["permuted_sentences"])
            labels.append(item["correct_order"])
            if "hypothesis" in item:
              hyp.append(item["hypothesis"])
            else:
              hyp.append("")
            if "entailment" in item:
              ent.append(item["entailment"])
            else:
              ent.append("")
    return dialogues, labels, hyp, ent


# create dataset
class DialogueReconstructionDataset(Dataset):
    def __init__(self, dialogues, labels, tokenizer, hyp, ent, max_length=512):
        self.dialogues = dialogues
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.hyp = hyp
        self.ent = ent

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        dialogue = self.dialogues[idx]
        label = self.labels[idx]
        hyp = self.hyp[idx]
        ent = self.ent[idx]

        # Tokenize dialogue
        encoding = self.tokenizer(
            dialogue,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Create labels for token positions with "mask" using -100

        input_ids = encoding["input_ids"].squeeze()
        cls_token_id = self.tokenizer.convert_tokens_to_ids("[CLS]")
        cls_positions = (input_ids == cls_token_id).nonzero(as_tuple=True)[0]
        labels = [-100] * self.max_length


        for i, cls_pos in enumerate(cls_positions):
            if i < len(label):
                labels[cls_pos] = label[i]


        return {
            "dialogue": dialogue,
            "hypothesis": hyp,
            "entailment": ent,
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(labels)
        }

# compute accuracies
def compute_binary_accuracy(pair_list):
    length = len(pair_list)
    score = 0
    for a, b in pair_list:
        if a == b:
            score += 1
    return(float(score)/length)


def compute_elem_accuracy(pair_list):
    length = 0
    score = 0
    for a, b in pair_list:
      for i in range(len(a)):
          length += 1
          if a[i] == b[i]:
              score += 1
    return(float(score)/length)

# Finetune and validate
def finetune(train_dataloader, eval_dataloader, params):
    print("Finetuning started...")
    # ignore -100 in loss calc
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=params.lr)

    # Calculate the total number of training steps (epochs * batches per epoch).
    num_training_steps = params.num_epochs * len(train_dataloader)
    num_warmup_steps = int(0.1 * num_training_steps)

    lr_scheduler = get_scheduler(
        name="linear",  # Linear learning rate decay.
        optimizer=optimizer,  # Optimizer whose learning rate will be scheduled.
        num_warmup_steps=num_warmup_steps,  # Number of warmup steps before decay begins.
        num_training_steps=num_training_steps  # Total number of training steps.
    )

    model.train()

    for epoch in range(params.num_epochs):  # Number of epochs
        j = 0
        for batch in train_dataloader:
            j += 1
            print(f"Finetune batch {j}")

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits  # Shape: [batch_size, seq_len, num_labels]

            # Calculate loss
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            # Backward pass
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            predictions = torch.argmax(logits, dim=-1)

        print(f"Epoch {epoch + 1} Loss: {loss.item()}")

        model.eval()  # Set the model to evaluation mode (disables dropout and gradient computation).
        to_comp = []
        j = 0
        for batch in eval_dataloader:
            j += 1
            print(f"Eval batch {j}")
            with torch.no_grad():  # Disable gradient calculation for efficiency.
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            logits = outputs.logits  # Extract logits (predicted scores for each class).
            predictions = torch.argmax(logits, dim=-1)  # Take the argmax to get predicted classes.
                    # Initialize lists to hold per-item results
            all_filtered_predictions = []
            all_filtered_labels = []

            # Iterate through the batch
            for i in range(labels.size(0)):  # Loop through batch size, for each example
                valid_indices = labels[i] != -100  # Valid indices for the current item
                filtered_predictions = predictions[i][valid_indices]  # Filter predictions for valid indices
                filtered_labels = labels[i][valid_indices]  # Filter labels for valid indices
                # Convert tensors to lists and append to the respective lists
                all_filtered_predictions.append(filtered_predictions.detach().cpu().tolist())
                all_filtered_labels.append(filtered_labels.detach().cpu().tolist())

            combined = list(zip(all_filtered_predictions, all_filtered_labels))
            to_comp.extend(combined)
        binary_score = compute_binary_accuracy(to_comp)
        elem_score = compute_elem_accuracy(to_comp)
        print(f'BINARY Validation Accuracy in Epoch {epoch + 1}: {binary_score * 100}%')
        print(f'ELEMENT-WISE Validation Accuracy in Epoch {epoch + 1}: {elem_score * 100}%')

    model_path = "finetuned_bert_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def print_pred(listy):
   # pair list in format (dialogue, all_filtered_labels)]
   print("creating outfile")
   items = []
   out_file = "output.json"
   for d, o, h, e in listy:
    dictToAdd = {"Dialogue": d,
                 "Order": o,
                 "Hypothesis": h,
                 "Entailment": e}
    items.append(dictToAdd)
   with open(out_file, 'w') as f:
    json.dump(obj=items, fp=f, indent=4)

def test(test_dataloader):
    print("Beginning testing")
    to_comp = []
    to_print = []
    model.eval()
    j = 0
    for batch in test_dataloader:
        with torch.no_grad():  # Disable gradient calculation for efficiency.
            j += 1
            print(f"Test batch {j}")
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            dialogue = batch["dialogue"]
            hyp = batch["hypothesis"]
            ent = batch["entailment"]
            if (type(ent) != list):
              ent = ent.tolist()

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
          
        logits = outputs.logits  # Extract logits (predicted scores for each class).
        predictions = torch.argmax(logits, dim=-1)  # Take the argmax to get predicted classes.
        # Initialize lists to hold per-item results
        all_filtered_predictions = []
        all_filtered_labels = []

        # Iterate through the batch
        for i in range(labels.size(0)):  # Loop through batch size, for each example
            valid_indices = labels[i] != -100  # Valid indices for the current item
            filtered_predictions = predictions[i][valid_indices]  # Filter predictions for valid indices
            filtered_labels = labels[i][valid_indices]  # Filter labels for valid indices
            # Convert tensors to lists and append to the respective lists
            all_filtered_predictions.append(filtered_predictions.detach().cpu().tolist())
            all_filtered_labels.append(filtered_labels.detach().cpu().tolist())


        to_print.extend(list(zip(dialogue, all_filtered_labels, hyp, ent)))
        combined = list(zip(all_filtered_predictions, all_filtered_labels))
        to_comp.extend(combined)
    binary_score = compute_binary_accuracy(to_comp)
    elem_score = compute_elem_accuracy(to_comp)
    print_pred(to_print)
    print(f'BINARY Test Accuracy: {binary_score * 100}%')  
    print(f'ELEMENT-WISE Test Accuracy: {elem_score * 100}%')

def main(params):

    dialogues, labels, hyp, ent = preprocess_data("task1-train.json") 
    train_dataset = DialogueReconstructionDataset(dialogues, labels, tokenizer, hyp, ent)
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
    dialogues, labels, hyp, ent = preprocess_data("task1-val.json")
    eval_dataset = DialogueReconstructionDataset(dialogues, labels, tokenizer, hyp, ent)
    eval_dataloader = DataLoader(eval_dataset, batch_size=params.batch_size, shuffle=True)
    dialogues, labels, hyp, ent = preprocess_data("task1-sled.json") 
    test_dataset = DialogueReconstructionDataset(dialogues, labels, tokenizer, hyp, ent)
    test_dataloader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=True)
    print("Preprocessing done...")
    if params.model != "":
        model.load_state_dict(torch.load(params.model, map_location=device))
        model.to(device)
    else:
      finetune(train_dataloader=train_dataloader, eval_dataloader=eval_dataloader, params=params) 
      model.to(device)
    test(test_dataloader=test_dataloader)


if __name__ == "__main__":
    batch_size = 16 # change to 16
    num_epochs = 5 # change to 5
    lr = 3e-5

    parser = argparse.ArgumentParser(description="Finetune BERT for PIQA Task")

    parser.add_argument("--batch_size", type=int, default=batch_size, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=num_epochs, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=lr, help="Learning rate for AdamW optimizer")
    parser.add_argument("--model", type=str, default="finetuned_bert_model.pt", help="model path if ONLY testing")
    params, unknown = parser.parse_known_args()
    main(params)
