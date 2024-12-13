import os
import shutil
import sys
from tqdm import tqdm
import argparse
import random
from typing import Optional, Union

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModelForMultipleChoice, get_scheduler
from transformers.tokenization_utils_base import PaddingStrategy

from dataclasses import dataclass
import evaluate
import json
from datasets import Dataset


def placeholder():
    raise NotImplementedError("This function has not been implemented yet.")

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

SEED = 595
set_seed(SEED)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")



@dataclass
class DataCollatorForMultipleChoice:

    tokenizer: AutoTokenizer.from_pretrained("bert-base-cased")
    device: torch.device
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features, label_name="labels"):

        batch = {}

        # Extract labels from the list of feature dictionaries and remove them from the feature dicts
        labels = [feature.pop(label_name) for feature in features]
        # Get the batch size (number of examples in the batch) and the number of options (solutions)
        batch_size = len(features)
        num_solutions = len(features[0]["input_ids"])

        # Flatten the structure of features for padding.
        # Each feature dictionary contains multiple options for the task (e.g., multiple choices),
        # and we want to treat each option as a separate instance for padding purposes.
        # `flattened_features` is now a list of lists, so we flatten it to make it a single list of feature dicts.
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_solutions)] for feature in features]
        flattened_features = sum(flattened_features, [])

        # Apply padding to the flattened features.
        # This ensures that all the sequences (e.g., input_ids, attention_masks) in the batch
        # are padded to the same length. Padding can be customized by the parameters `padding`,
        # `max_length`, and `pad_to_multiple_of`.
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )


        # Reshape the padded batch back to its original structure.
        # After padding, we reshape the tensors to reflect the original batch structure (batch_size, num_solutions, seq_len).
        # For each tensor (input_ids, attention_mask, etc.), we view it as a 3D tensor where the first dimension
        # corresponds to the batch, the second dimension corresponds to the different options (solutions),
        # and the third dimension is the sequence length (after padding).
        batch = {k: v.view(batch_size, num_solutions, -1) for k, v in batch.items()}


        # Convert the labels into a tensor and move them to the specified device (e.g., GPU or CPU).

        batch = {k: v.to(self.device) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels).to(self.device)

        return batch



def load_data(dataset, tokenizer, device, params):

    num_solutions = 2 

    def tokenize_function(examples):

        tokenized_dict = {}


        goals = []
        for goal in examples["goal"]:
          for i in range(num_solutions):
            goals.append(goal)

        # Create a list of solution pairs (sol1 and sol2).
        solutions = []
        for sol1, sol2 in zip(examples["sol1"], examples["sol2"]):
          solutions.extend([sol1, sol2])


        # Flatten the goals and solutions into single lists for tokenization.
        combined = [goal + solution for goal, solution in zip(goals, solutions)]

        # Tokenize the pairs of goal and solution sentences.
        tokenized_dict = tokenizer(combined, truncation=True, max_length=tokenizer.model_max_length, add_special_tokens=False)


        # Group the tokenized results back into pairs, corresponding to the number of solutions.
        for k in tokenized_dict:
            tokenized_dict[k] = [tokenized_dict[k][i:i + num_solutions] for i in range(0, len(tokenized_dict[k]), num_solutions)]
        tokenized_dict["label"] = examples["label"]

        return tokenized_dict

    # Load the dataset from the provided parameters.

    # Tokenize and process the dataset using the `tokenize_function`.
    tokenized_datasets = dataset.map(tokenize_function, batched=True)  # Apply tokenization to the entire dataset.
    tokenized_datasets = tokenized_datasets.remove_columns(["goal", "sol1", "sol2"])  # Remove unnecessary columns.
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")  # Rename 'label' to 'labels' for consistency.
    tokenized_datasets.set_format("torch")  # Set the dataset format to PyTorch tensors.

    # Initialize a data collator to dynamically pad the inputs during batch creation.
    data_collator = DataCollatorForMultipleChoice(tokenizer, device)

    # Create DataLoader objects for the train, validation, and test sets.
    dataloader = DataLoader(
        tokenized_datasets,  # Training dataset.
        shuffle=True,  # Shuffle the training data.
        batch_size=params.batch_size,  # Use the batch size from the params.
        collate_fn=data_collator  # Use the custom data collator to pad and batch the data.
    )

    # Return the DataLoader objects for train, validation, and test sets.
    return dataloader



def finetune(model, train_dataloader, params):

    # Calculate the total number of training steps (epochs * batches per epoch).
    num_training_steps = params.num_epochs * len(train_dataloader)

    # Initialize the optimizer (AdamW) with the model's parameters and learning rate.
    optimizer = AdamW(model.parameters(), lr=params.lr)

    num_warmup_steps = int(0.1*num_training_steps)

    # Set up the learning rate scheduler for controlling the learning rate dynamically during training.
    # We use a linear schedule with warmup steps.
    lr_scheduler = get_scheduler(
        name="linear",  # Linear learning rate decay.
        optimizer=optimizer,  # Optimizer whose learning rate will be scheduled.
        num_warmup_steps=num_warmup_steps,  # Number of warmup steps before decay begins.
        num_training_steps=num_training_steps  # Total number of training steps.
    )

    # Load the evaluation metric to be used during validation (accuracy in this case).
    metric = evaluate.load("accuracy")

    # Initialize a progress bar to visually track training progress over the total number of steps.
    progress_bar = tqdm(range(num_training_steps))

    # Training loop over the specified number of epochs.
    for epoch in range(params.num_epochs):

        # Set the model to training mode (enables dropout and gradient computation).
        model.train()
        for batch in train_dataloader:

            # Forward pass: run the model on the batch and compute the loss.
            outputs = model(**batch)
            loss = outputs.loss

            # Backpropagate the loss to compute gradients.
            loss.backward()


            # Update model parameters using the optimizer.
            optimizer.step()


            # Update the learning rate according to the schedule.
            lr_scheduler.step()


            # Reset the gradients for the next step.
            optimizer.zero_grad()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch: {epoch + 1}, Batch loss: {loss.item()},Learning Rate: {current_lr}")

            # Update the progress bar for each step.
            progress_bar.update(1)

    model_path = "task_2_bert_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    # Return the fine-tuned model.
    return model



def test(model, test_dataloader):

    # Load the accuracy metric to evaluate the test predictions.
    metric = evaluate.load("accuracy")

    # Set the model to evaluation mode (disables dropout and gradient computation).
    model.eval()

    # Initialize a list to store predictions.
    predictions = []

    # Loop over batches of test data from the test dataloader.
    i = 1
    for batch in test_dataloader:
        print(f"on batch {i}")
        i += 1

        with torch.no_grad():  # Disable gradient computation for efficiency during testing.
            # Forward pass: get the model's outputs for the test batch.
            outputs = model(**batch)

        # Extract logits (model's predicted scores for each class).
        logits = outputs.logits

        # Get the predicted class by taking the argmax of the logits (the class with the highest score).
        preds = torch.argmax(logits, dim=-1)

        metric.add_batch(predictions=preds, references=batch["labels"])

        # Convert the predictions to CPU and extend the list with the new batch predictions.
        predictions.extend(preds.cpu().numpy())

    # Compute the final accuracy on the test set.
    score = metric.compute()

    # Print the test accuracy.
    print(f'Test Accuracy: {score["accuracy"]}')



def main(params):

    # Load the tokenizer for the model specified in the params (e.g., BERT tokenizer).
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


    file_path = "datasets/task2/finetune-task2-orig.json"
    with open(file_path, 'r') as file:
        data = json.load(file)
        dataset = Dataset.from_list(data)

    train_dataloader = load_data(dataset, tokenizer, device, params)


    file_path = "datasets/task2/task2-bert.json"
    with open(file_path, 'r') as file:
        data = json.load(file)
        dataset = Dataset.from_list(data)

    test_dataloader = load_data(dataset, tokenizer, device, params)

    model = AutoModelForMultipleChoice.from_pretrained("bert-base-cased")
    if params.model != "":
        model.load_state_dict(torch.load(params.model, map_location=device))
        model.to(device)
    else:
        model = finetune(model, train_dataloader, params)
        model.to(device)


    # Test the fine-tuned model on the test dataset and print accuracy results.
    test(model, test_dataloader)



if __name__ == "__main__":

    batch_size = 32
    num_epochs = 20
    lr = 1e-4

    parser = argparse.ArgumentParser(description="Finetune BERT for Task 2")

    parser.add_argument("--batch_size", type=int, default=batch_size, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=num_epochs, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=lr, help="Learning rate for AdamW optimizer")
    parser.add_argument("--model", type=str, default="task-2-7.pt", help="Path to the pre-trained model")

    params, unknown = parser.parse_known_args()
    main(params)

