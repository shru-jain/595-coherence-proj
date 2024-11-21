from typing import Union, List
import json
import random

def permute(json_input: Union[str, List[str]], use_separators: bool):
    permuted_output = []
    # Check if json_input is just one file and converts it to list format
    if(isinstance(json_input, str)):
        json_input = [json_input]
    
    # Loop through all json files passed as input
    for json_file in json_input:
        json_data = []
        seen_conversations = set()
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        except FileNotFoundError:
            print(f"{json_file} not found.")
        except json.JSONDecodeError:
            print(f"{json_file} not a valid json file.")
        
        for dialogue in json_data:
            if not dialogue['Conversation'] or not dialogue['Conversation'][0]:
                continue
            if str(dialogue['Conversation']) in seen_conversations or dialogue['Items'] > 20:
                continue
            seen_conversations.add(str(dialogue['Conversation']))
            new_conversation = {}
            # These key-value pairs can be commented out if not needed
            new_conversation['items'] = dialogue['Items']
            #new_conversation['hypothesis'] = dialogue['Hypothesis']
            new_conversation['original'] = dialogue['Conversation']
            #new_conversation['entailment'] = dialogue['Entailment']
            
            # Permutation logic
            permuted_order = list(range(1, dialogue['Items'] + 1))
            random.shuffle(permuted_order)
            permuted_sentences = ""
            if use_separators:
                permuted_sentences = "[CLS]"
            correct_order =  list(range(dialogue['Items']))
            for idx, sentenceIdx in enumerate(permuted_order):
                permuted_sentences += f"{idx + 1}) {dialogue['Conversation'][sentenceIdx - 1]}"
                if use_separators:
                    permuted_sentences += " [SEP] "
                else:
                    permuted_sentences += "\n"
                correct_order[sentenceIdx - 1] = idx + 1
            
            if use_separators:
                permuted_sentences = permuted_sentences[:-1]
            
            new_conversation['permuted_sentences'] = permuted_sentences
            new_conversation['permuted_order'] = permuted_order
            new_conversation['correct_order'] = correct_order
            permuted_output.append(new_conversation)
    random.shuffle(permuted_output)
    return permuted_output


result = permute("datasets/no-hyp-train.json", True)


train_size = int(0.8 * len(result))
validation_size = len(result) - train_size

train_data = result[:train_size]
validation_data = result[train_size:]

with open("datasets/permuted-train-data.json", 'w') as f:
    json.dump(obj=train_data, fp=f, indent=4)
with open("datasets/permuted-validation-data.json", 'w') as f:
    json.dump(obj=validation_data, fp=f, indent=4)
