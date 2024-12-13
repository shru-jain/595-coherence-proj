from typing import Union, List
import json
import random
import re

def create_inp_file(json_file, out_file):
    out = []
    with open(f"datasets/{json_file}", 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        for item in json_data:
            # hypothesis
            if "hypothesis" in item:
                hyp = item["hypothesis"]
            else:
                hyp = item["Hypothesis"]
            # entailment
            if "Entailment" in item:
                ent = 1 if item["Entailment"] else 0
            else: 
                ent = 1 if item["entailment"] else 0
            # dialogue
            dialogue = "Conversation:"
            if "Dialogue" in item:
                initial = item["Dialogue"]
                initial = initial.replace('[CLS] ','')
                initial = re.sub(r'\d+\)\s', '', initial)
                sentences = initial.split(' [SEP] ')[:-1]
                for num, i in enumerate(item["Order"]):
                    dialogue += f" [CLS] {num + 1}) {sentences[i-1]} [SEP]"
            elif "permuted_sentences" in item: # 
                initial = item["permuted_sentences"]
                for i in initial:
                    dialogue += f" [CLS] {i} [SEP]"
            else: 
                initial = item["Conversation"]
                for num, i in enumerate(initial):
                    dialogue += f" [CLS] {num + 1}) {i} [SEP]"
            dialogue = f"{dialogue} \nHypothesis: {hyp}. This hypothesis is? "
            toAdd = {
                "goal": dialogue,
                "sol1": "False",
                "sol2": "True",
                "label": ent
            }
            out.append(toAdd)
    with open(f"datasets/{out_file}", 'w') as f:
            json.dump(obj=out, fp=f, indent=4)

 
create_inp_file("permuted-sled-test.json", "task2/task2-perm.json")
create_inp_file("sled-test.json", "task2/task2-orig.json")
create_inp_file("finetune-output-task1.json", "task2/task2-bert.json")
