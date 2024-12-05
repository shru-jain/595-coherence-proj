import json
import sys
import re
total_bad = 0
def parse_dialogue(dialogue_text):
    global total_bad
    dialog_speaker_list = []
    dialog_text_list = []
    if("#Person3" in dialogue_text):
        total_bad += 1
        return [], []
    dialogue_text = dialogue_text.replace("#Person1#", "A").replace("#Person2#", "B")
    lines = dialogue_text.strip().split('\n')
    for line in lines:
        if ': ' in line:
            speaker, message = line.split(': ', 1)
            speaker = speaker.strip()
            message = message.strip()
            dialog_text_list.append(f"{speaker}: {message}")
            # dialog_speaker_list.append(speaker)
            # dialog_text_list.append(message)
        else:
            print("weird formatted line")
            continue
    return dialog_text_list

def convert_json_data(input_file_path, output_file_path):
    output_data = {"total_items": 0, "data": []}
    total_items = 0
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            total_items += 1
            dialogue_text = data.get('dialogue', '')
            dialog_text_list = parse_dialogue(dialogue_text)
            if(len(dialog_text_list) == 0):
                continue
            hypothesis = data.get('summary', '') # summary as hypothseis
            conversation = dialog_text_list
            output_data['data'].append({
                'Hypothesis': hypothesis,
                'Conversation': conversation,
                'Items': len(conversation),
                'Entailment': True
            })
    output_data['total_items'] = total_items

    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        json.dump(output_data, f_out, indent=4)


if len(sys.argv) > 1:
    input_file = sys.argv[1]
    input_file_name = input_file.rsplit(".", 1)[0]
    output_file = f"{input_file_name}.processed-output.json"
    convert_json_data(input_file, output_file)
    print(total_bad)
else:
    print("Input file required")