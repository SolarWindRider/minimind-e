#!/usr/bin/env python3
import json
import argparse
import os
import re
from tqdm import tqdm


def extract_instruction(content):
    instruction_match = re.search(r"instruction:\s*['\"](.*?)['\"]", content, re.DOTALL)
    if instruction_match:
        return instruction_match.group(1)
    return ""


def extract_action_sequence(content, pattern=None):
    if pattern is None:
        pattern = r"action sequence:\s*\[(.*?)\]"

    sequence_match = re.search(pattern, content, re.DOTALL)
    if sequence_match:
        actions_str = sequence_match.group(1)
        action_items = re.findall(r"['\"](.*?)['\"]", actions_str)
        return action_items
    return []


def convert_era_to_vla(era_data_path, output_path, images_folder=""):
    print(f"Loading ERA dataset from {era_data_path}...")

    if era_data_path.endswith('.jsonl'):
        with open(era_data_path, 'r') as f:
            era_data = [json.loads(line) for line in f]
    else:
        with open(era_data_path, 'r') as f:
            era_data = json.load(f)

    print(f"Loaded {len(era_data)} samples")

    action_set = set()
    vla_samples = []

    for item in tqdm(era_data, desc="Processing samples"):
        instruction = ""
        image_path = ""
        actions = []

        for conv in item.get('conversations', []):
            content = conv.get('value', '')

            if conv.get('from') == 'human':
                inst = extract_instruction(content)
                if inst:
                    instruction = inst

            if conv.get('from') == 'gpt':
                action_seq = extract_action_sequence(content)
                if action_seq:
                    actions = action_seq

        if not instruction or not actions:
            continue

        if 'image' in item:
            image_path = item['image']
            if images_folder and image_path:
                image_path = os.path.join(images_folder, image_path)

        for action in actions:
            action_set.add(action.lower().strip())

        sample = {
            'instruction': instruction,
            'actions': actions,
            'image': image_path,
        }
        vla_samples.append(sample)

    action_vocab = sorted(list(action_set))
    print(f"\nExtracted {len(action_vocab)} unique actions")
    print(f"Converted {len(vla_samples)} valid VLA samples")

    output_data = {
        'action_vocab': action_vocab,
        'samples': vla_samples,
        'metadata': {
            'source': 'ERA',
            'num_samples': len(vla_samples),
            'num_actions': len(action_vocab),
        }
    }

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("Done!")

    print(f"\nSample actions from vocabulary:")
    for action in action_vocab[:20]:
        print(f"  - {action}")

    return action_vocab, vla_samples


def main():
    parser = argparse.ArgumentParser(description="Convert ERA dataset to MiniMind-VLA format")
    parser.add_argument("--input_path", type=str, required=True, help="Path to ERA dataset JSON/JSONL file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output VLA dataset JSON file")
    parser.add_argument("--images_folder", type=str, default="", help="Base folder for images")
    args = parser.parse_args()

    convert_era_to_vla(args.input_path, args.output_path, args.images_folder)


if __name__ == "__main__":
    main()
