from datasets import load_dataset
import json
import random

def load_sled_dataset():
    ds = load_dataset("sled-umich/Conversation-Entailment")
    seen_conversations = set()
    transformed_data = []
    for item in ds['validation']:
        if len(item["dialog_text_list"]) > 2:
            convo = []
            convo.extend([f"{speaker}: {text}" for speaker, text in zip(item["dialog_speaker_list"], item["dialog_text_list"])])
            if item["h"] not in seen_conversations:
                seen_conversations.add(item["h"])
                transformed_data.append(    
                    {
                        "Items": len(item["dialog_text_list"]),
                        "Hypothesis": item["h"],
                        "Conversation": convo,
                        "Entailment": item["entailment"]
                    }
                )

    for item in ds['test']:
        if len(item["dialog_text_list"]) > 2:
            convo = []
            convo.extend([f"{speaker}: {text}" for speaker, text in zip(item["dialog_speaker_list"], item["dialog_text_list"])])

            if item["h"] not in seen_conversations:
                seen_conversations.add(item["h"])
                transformed_data.append(    
                    {
                        "Items": len(item["dialog_text_list"]),
                        "Hypothesis": item["h"],
                        "Conversation": convo,
                        "Entailment": item["entailment"]
                    }
                )
    with open("sled-dataset.json", "w") as f:
        json.dump(transformed_data, f, indent=4)


def create_split(filename="sled-dataset.json"):
    trueItems = []
    falseItems = []

    with open(f"datasets/{filename}", 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        for item in json_data:
            if item["Entailment"] == True:
                trueItems.append(item)
            else:
                falseItems.append(item)
    
    random.shuffle(trueItems)
    random.shuffle(falseItems)

    trueFinetune = trueItems[0:60]
    trueTest = trueItems[60:]

    falseFinetune = falseItems[0:40]
    falseTest = falseItems[40:]

    finetune = trueFinetune + falseFinetune
    test = trueTest + falseTest

    random.shuffle(finetune)
    random.shuffle(test)
    

    with open("datasets/sled-finetune.json", "w") as f:
        json.dump(finetune, f, indent=4)

    with open("datasets/sled-test.json", "w") as f:
        json.dump(test, f, indent=4)


# call/edit vars as needed
load_sled_dataset()
create_split()

