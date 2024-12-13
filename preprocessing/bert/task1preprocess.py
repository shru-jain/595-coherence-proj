import json

def prep(json_file, out_file, addHyp=False):
    items = []
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        for item in json_data:
            permuted_sentences = ""
            for dialog in item["permuted_sentences"]:
                permuted_sentences += f"[CLS] {dialog} [SEP] "
            toAppend = {
                "permuted_sentences": permuted_sentences,
                "correct_order": item["correct_order"]
            }
            if addHyp:
                toAppend["hypothesis"] = item["hypothesis"]
                toAppend["entailment"] = item["entailment"]
            items.append(toAppend)
    with open(out_file, 'w') as f:
        json.dump(obj=items, fp=f, indent=4)

prep('datasets/dialogsum/dialogsum.test.processed-output-permuted.json', 'datasets/task1/task1-test.json')
prep('datasets/dialogsum/dialogsum.train.processed-output-permuted.json', 'datasets/task1/task1-train.json')
prep('datasets/dialogsum/dialogsum.validation.processed-output-permuted.json', 'datasets/task1/task1-val.json')

prep('datasets/permuted-sled-test.json','datasets/task1/task1-sled.json', True)
