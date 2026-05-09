import io, json, random, re, os, torch
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import Dataset
from PIL import Image


class VLADataset(Dataset):
    def __init__(self, data_path, tokenizer, vision_processor=None,
                 max_length=1024, action_special_token='<|action_pad|>',
                 action_stop_token='<|action_stop|>', image_special_token='<|image_pad|>',
                 image_token_len=64, action_vocab_size=512,
                 images_folder=""):
        super().__init__()
        self.tokenizer = tokenizer
        self.vision_processor = vision_processor
        self.max_length = max_length
        self.action_token = action_special_token
        self.image_token_len = image_token_len
        self.image_token = image_special_token * image_token_len
        self.images_folder = images_folder

        self.action_vocab_size = action_vocab_size
        self.text_vocab_size = len(tokenizer)
        self.image_token_id = tokenizer.encode(image_special_token, add_special_tokens=False)[0] if isinstance(image_special_token, str) and len(image_special_token) > 1 else image_special_token if isinstance(image_special_token, int) else None
        self.grounding_system_message = "You are a household assistant."

        self.action_pad_token_id = tokenizer.encode(action_special_token, add_special_tokens=False)[0] if isinstance(action_special_token, str) else action_special_token
        self.action_stop_token_id = tokenizer.encode(action_stop_token, add_special_tokens=False)[0] if isinstance(action_stop_token, str) else action_stop_token
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

        self.action_vocab = self._build_action_vocabulary()
        self.action_token_to_id = {action: idx + 10 for idx, action in enumerate(self.action_vocab)}
        self.action_id_to_token = {idx + 10: action for idx, action in enumerate(self.action_vocab)}
        self.action_pad_id = 0
        self.action_stop_id = 1
        self.action_start_id = 2

        self._load_data(data_path)

    def _build_action_vocabulary(self):
        action_set = set()
        for item in self.list_data_dict:
            trajectories = item.get('trajectories', [])
            for traj in trajectories:
                actions = traj.get('actions', [])
                for action in actions:
                    normalized = self._normalize_action(action)
                    if normalized:
                        action_set.add(normalized)
        if not action_set:
            action_set = self._extract_actions_from_conversations()
        return sorted(list(action_set))

    def _extract_actions_from_conversations(self):
        action_set = set()
        patterns = [
            r"'(.*?)'",
            r'"(.*?)"',
        ]
        keywords = ['find', 'pick', 'put', 'open', 'close', 'turn', 'slice', 'clean',
                    'heat', 'place', 'move', 'grab', 'wash', 'cool', 'chill', 'fill',
                    'empty', 'wipe', 'toggle', 'goto', 'walk', 'navigate', 'examine',
                    'look at', 'interact', 'use', 'turn on', 'turn off', 'point']

        for item in self.list_data_dict:
            conversations = item.get('conversations', [])
            for conv in conversations:
                content = conv.get('value', '') if isinstance(conv, dict) else str(conv)
                for pattern in patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        for keyword in keywords:
                            if keyword in match.lower():
                                action_set.add(match.lower().strip())
        return sorted(list(action_set))

    def _normalize_action(self, action):
        if not action:
            return None
        action = action.strip().lower()
        action = re.sub(r'\s+', ' ', action)
        return action

    def _load_data(self, data_path):
        if data_path.endswith('.json') or data_path.endswith('.jsonl'):
            if data_path.endswith('.jsonl'):
                self.list_data_dict = []
                with open(data_path) as f:
                    for line in f:
                        self.list_data_dict.append(json.loads(line.strip()))
            else:
                with open(data_path) as f:
                    self.list_data_dict = json.load(f)
        elif data_path.endswith('.parquet'):
            tables = [pa.Table.from_batches(pq.ParquetFile(p.strip()).iter_batches()) for p in data_path.split(',')]
            tables = [t.cast(pa.schema([f.with_type(pa.large_string()) if pa.types.is_string(f.type) else f for f in t.schema])) for t in tables]
            table = pa.concat_tables(tables, promote_options='default')
            self.list_data_dict = []
            for i in range(len(table)):
                item = {col: table.column(col)[i].as_py() for col in table.column_names}
                self.list_data_dict.append(item)
        else:
            raise ValueError(f"Unsupported data format: {data_path}")

    def _parse_trajectory_data(self, item):
        instruction = ""
        image_path = ""
        actions = []

        instruction = item.get('instruction', '')
        if not instruction:
            for conv in item.get('conversations', []):
                content = conv.get('value', '') if isinstance(conv, dict) else str(conv)
                if conv.get('from') == 'human':
                    inst_match = re.search(r"instruction:\s*['\"](.*?)['\"]", content, re.DOTALL)
                    if inst_match:
                        instruction = inst_match.group(1)
                        break

        image_path = item.get('image', '')
        if not image_path and 'image' in item:
            image_path = item['image']

        trajectories = item.get('trajectories', [])
        if trajectories:
            for traj in trajectories:
                actions.extend(traj.get('actions', []))
        else:
            for conv in item.get('conversations', []):
                content = conv.get('value', '') if isinstance(conv, dict) else str(conv)
                if conv.get('from') == 'gpt':
                    action_match = re.search(r"action sequence:\s*\[(.*?)\]", content, re.DOTALL)
                    if action_match:
                        actions_str = action_match.group(1)
                        action_items = re.findall(r"['\"](.*?)['\"]", actions_str)
                        actions.extend(action_items)

        return instruction, image_path, actions

    def __len__(self):
        return len(self.list_data_dict)

    def load_image_inputs(self, image_path):
        if not image_path or self.vision_processor is None:
            return None
        try:
            if isinstance(image_path, bytes):
                image = Image.open(io.BytesIO(image_path)).convert('RGB')
            else:
                full_path = image_path if os.path.isabs(image_path) else os.path.join(self.images_folder, image_path)
                if not os.path.exists(full_path):
                    return None
                image = Image.open(full_path).convert('RGB')
            inputs = self.vision_processor(images=image, return_tensors="pt")
            if hasattr(inputs, 'keys'):
                return {k: v for k, v in inputs.items()}
            return inputs.pixel_values
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def encode_actions(self, actions):
        action_ids = [self.action_start_id]
        for action in actions:
            normalized = self._normalize_action(action)
            if normalized and normalized in self.action_token_to_id:
                action_ids.append(self.action_token_to_id[normalized])
            else:
                action_hash = hash(normalized) % (self.action_vocab_size - 10) + 10
                action_ids.append(action_hash)
        action_ids.append(self.action_stop_id)
        return action_ids

    def decode_actions(self, action_ids):
        actions = []
        for aid in action_ids:
            if aid == self.action_stop_id or aid == self.action_pad_id or aid == self.action_start_id:
                continue
            if aid in self.action_id_to_token:
                actions.append(self.action_id_to_token[aid])
            else:
                actions.append(f"<action_{aid}>")
        return actions

    def __getitem__(self, index: int):
        try:
            item = self.list_data_dict[index]
            instruction, image_path, actions = self._parse_trajectory_data(item)

            if not instruction:
                instruction = "Complete the household task."

            pixel_values = None
            if image_path:
                pixel_values = self.load_image_inputs(image_path)
            if pixel_values is None:
                pixel_values = {'pixel_values': torch.zeros(1, 3, 256, 256)}

            prompt = f"You are a household assistant. Task: {instruction}"
            prompt_with_image = f"{self.image_token}\n{prompt}"

            prompt_ids = self.tokenizer(prompt_with_image, add_special_tokens=False).input_ids[:self.max_length - 20]
            prompt_ids = [self.tokenizer.bos_token_id] + prompt_ids if self.tokenizer.bos_token_id else prompt_ids

            action_ids = self.encode_actions(actions)[:50]

            max_response_len = self.max_length - len(prompt_ids) - 10
            action_ids = action_ids[:max_response_len]

            full_input_ids = prompt_ids + action_ids
            labels = [-100] * len(prompt_ids) + action_ids

            padding_len = self.max_length - len(full_input_ids)
            if padding_len > 0:
                full_input_ids = full_input_ids + [self.tokenizer.pad_token_id] * padding_len
                labels = labels + [-100] * padding_len
            else:
                full_input_ids = full_input_ids[:self.max_length]
                labels = labels[:self.max_length]

            input_ids_tensor = torch.tensor(full_input_ids, dtype=torch.long)
            labels_tensor = torch.tensor(labels, dtype=torch.long)

            return input_ids_tensor, labels_tensor, pixel_values

        except Exception as e:
            print(f"Error processing sample {index}: {e}")
            return self.__getitem__(random.randint(0, len(self.list_data_dict) - 1))


class VLADatasetCollator:
    def __init__(self, tokenizer, ignore_index=-100):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index

    def __call__(self, instances):
        input_ids, labels, pixel_values_list = zip(*instances)

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.ignore_index)

        max_pixel_len = max(pv['pixel_values'].shape[0] for pv in pixel_values_list)
        pixel_values_padded = []
        image_grid_thw_list = []
        for pv in pixel_values_list:
            pv_tensor = pv['pixel_values']
            pad_len = max_pixel_len - pv_tensor.shape[0]
            if pad_len > 0:
                padding = torch.zeros(pad_len, *pv_tensor.shape[1:], dtype=pv_tensor.dtype)
                pv_tensor = torch.cat([pv_tensor, padding], dim=0)
            pixel_values_padded.append(pv_tensor)
            image_grid_thw_list.append(torch.tensor([1, pv_tensor.shape[2]//16, pv_tensor.shape[3]//16]))

        pixel_values = torch.stack(pixel_values_padded)
        image_grid_thw = torch.stack(image_grid_thw_list)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'image_grid_thw': image_grid_thw,
        }


def create_vla_dataset_from_era(era_data_path, output_path, images_folder=""):
    action_set = set()

    print(f"Loading ERA data from {era_data_path}...")
    if era_data_path.endswith('.jsonl'):
        with open(era_data_path, 'r') as f:
            era_data = [json.loads(line) for line in f]
    else:
        with open(era_data_path, 'r') as f:
            era_data = json.load(f)

    print(f"Loaded {len(era_data)} samples, extracting actions...")

    for item in era_data:
        for conv in item.get('conversations', []):
            content = conv.get('value', '')
            action_match = re.search(r"action sequence:\s*\[(.*?)\]", content, re.DOTALL)
            if action_match:
                actions_str = action_match.group(1)
                action_items = re.findall(r"['\"](.*?)['\"]", actions_str)
                for action in action_items:
                    action_set.add(action.lower().strip())

    action_vocab = sorted(list(action_set))
    print(f"Found {len(action_vocab)} unique actions")

    vla_data = []
    for item in era_data:
        instruction = ""
        image = ""
        actions = []

        for conv in item.get('conversations', []):
            content = conv.get('value', '')
            if conv.get('from') == 'human':
                inst_match = re.search(r"instruction:\s*['\"](.*?)['\"]", content, re.DOTALL)
                if inst_match:
                    instruction = inst_match.group(1)
            if conv.get('from') == 'gpt':
                action_match = re.search(r"action sequence:\s*\[(.*?)\]", content, re.DOTALL)
                if action_match:
                    actions_str = action_match.group(1)
                    action_items = re.findall(r"['\"](.*?)['\"]", actions_str)
                    actions = action_items

        if instruction and actions:
            image = item.get('image', '')
            if images_folder and image:
                image = os.path.join(images_folder, image)

            vla_item = {
                'instruction': instruction,
                'actions': actions,
                'image': image,
            }
            vla_data.append(vla_item)

    print(f"Converted {len(vla_data)} VLA samples")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'action_vocab': action_vocab,
            'samples': vla_data,
        }, f, indent=2, ensure_ascii=False)

    print(f"Saved to {output_path}")
    return action_vocab, vla_data
