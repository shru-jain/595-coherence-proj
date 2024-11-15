import json
import sys

def parse_dialogue(dialogue_text):
    dialog_speaker_list = []
    dialog_text_list = []
    lines = dialogue_text.strip().split('\n')
    for line in lines:
        if ': ' in line:
            speaker, message = line.split(': ', 1)
            speaker = speaker.strip()
            message = message.strip()
            dialog_speaker_list.append(speaker)
            dialog_text_list.append(message)
        else:
            print("weird formatted line")
            continue
    return dialog_speaker_list, dialog_text_list

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
            hypothesis = data.get('summary', '') # summary as hypothseis
            dialogue_text = data.get('dialogue', '')
            dialog_speaker_list, dialog_text_list = parse_dialogue(dialogue_text)
            conversation = {
                'dialog_speaker': dialog_speaker_list,
                'dialog_text': dialog_text_list
            }
            output_data['data'].append({
                'hypothesis': hypothesis,
                'conversation': conversation,
                'dialogue_items': len(conversation['dialog_speaker'])
            })
    output_data['total_items'] = total_items

    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        json.dump(output_data, f_out, indent=4)


if len(sys.argv) > 1:
    input_file = sys.argv[1]
    input_file_name = input_file.rsplit(".", 1)[0]
    output_file = f"{input_file_name}.processed-output.json"
    convert_json_data(input_file, output_file)
else:
    print("Input file required")