import io, json, random, re, os, torch
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import Dataset
from PIL import Image


class VLAStepDataset(Dataset):
    """
    时序 VLA 数据集：每个时间步 (observation, action) 对

    数据格式：
    {
        "instruction": "Put the cup on the table",
        "steps": [
            {"image": "image1.jpg", "action": "find a cup"},
            {"image": "image2.jpg", "action": "pick up the cup"},
            ...
        ]
    }

    训练方式：给定前 N 个 (obs, action) 对，预测第 N+1 个动作
    推理方式：循环执行，观察 → 动作 → 观察 → 动作 → ...
    """

    def __init__(self, data_path, tokenizer, vision_processor=None,
                 max_length=512, image_special_token='<|image_pad|>',
                 image_token_len=64, images_folder="",
                 max_history_steps=10):
        super().__init__()
        self.tokenizer = tokenizer
        self.vision_processor = vision_processor
        self.max_length = max_length
        self.image_token_len = image_token_len
        self.image_token = image_special_token * image_token_len
        self.images_folder = images_folder
        self.max_history_steps = max_history_steps

        self.action_pad_id = 0
        self.action_stop_id = 1
        self.action_start_id = 2
        self.text_vocab_size = len(tokenizer)
        self.image_token_id = tokenizer.encode(image_special_token, add_special_tokens=False)[0] if isinstance(image_special_token, str) else image_special_token

        self._load_data(data_path)
        self.action_vocab = self._build_action_vocabulary()
        self.action_token_to_id = {action: idx + 10 for idx, action in enumerate(self.action_vocab)}
        self.action_id_to_token = {idx + 10: action for idx, action in enumerate(self.action_vocab)}
        self.action_id_to_token[0] = "<pad>"
        self.action_id_to_token[1] = "<stop>"
        self.action_id_to_token[2] = "<start>"

    def _load_data(self, data_path):
        if data_path.endswith('.json') or data_path.endswith('.jsonl'):
            if data_path.endswith('.jsonl'):
                self.list_data_dict = []
                with open(data_path) as f:
                    for line in f:
                        self.list_data_dict.append(json.loads(line.strip()))
            else:
                with open(data_path) as f:
                    data = json.load(f)
                    if isinstance(data, dict) and 'samples' in data:
                        self.list_data_dict = data['samples']
                    else:
                        self.list_data_dict = data
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

    def _build_action_vocabulary(self):
        action_set = set()
        for item in self.list_data_dict:
            steps = item.get('steps', [])
            for step in steps:
                action = step.get('action', '')
                if action:
                    action_set.add(action.lower().strip())
        return sorted(list(action_set))

    def __len__(self):
        return len(self.list_data_dict)

    def encode_action(self, action):
        normalized = action.lower().strip() if action else ""
        if normalized in self.action_token_to_id:
            return self.action_token_to_id[normalized]
        action_hash = hash(normalized) % 500 + 10
        return action_hash

    def decode_action(self, action_id):
        if action_id in self.action_id_to_token:
            return self.action_id_to_token[action_id]
        return f"<action_{action_id}>"

    def load_image(self, image_path):
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
            return None

    def __getitem__(self, index):
        try:
            item = self.list_data_dict[index]
            instruction = item.get('instruction', 'Complete the task.')
            steps = item.get('steps', [])

            if not steps or len(steps) < 2:
                return self._get_dummy_sample()

            history_size = random.randint(1, min(len(steps) - 1, self.max_history_steps))
            history_steps = steps[:history_size]
            target_step = steps[history_size]

            current_image = target_step.get('image', '')
            target_action = target_step.get('action', '')

            prompt = f"Task: {instruction}"
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids[:50]

            history_ids = []
            for step in history_steps:
                action = step.get('action', '')
                action_id = self.encode_action(action)
                history_ids.append(action_id)

            target_action_id = self.encode_action(target_action)

            max_action_len = (self.max_length - len(prompt_ids) - 20) // 2
            history_ids = history_ids[-max_action_len:] if len(history_ids) > max_action_len else history_ids

            input_ids = prompt_ids + [self.action_start_id] + history_ids + [self.action_stop_id]
            labels = [self.action_pad_id] * len(prompt_ids) + [self.action_pad_id] + history_ids + [target_action_id]

            padding_len = self.max_length - len(input_ids)
            if padding_len > 0:
                input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_len
                labels = labels + [self.action_pad_id] * padding_len
            else:
                input_ids = input_ids[:self.max_length]
                labels = labels[:self.max_length]

            current_pixel_values = None
            if current_image:
                current_pixel_values = self.load_image(current_image)
            if current_pixel_values is None:
                current_pixel_values = {'pixel_values': torch.zeros(1, 3, 256, 256)}

            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'pixel_values': current_pixel_values,
                'instruction': instruction,
                'history_actions': [self.decode_action(aid) for aid in history_ids],
                'target_action': target_action,
            }

        except Exception as e:
            return self._get_dummy_sample()

    def _get_dummy_sample(self):
        return {
            'input_ids': torch.zeros(self.max_length, dtype=torch.long),
            'labels': torch.tensor([-100] * self.max_length, dtype=torch.long),
            'pixel_values': {'pixel_values': torch.zeros(1, 3, 256, 256)},
            'instruction': '',
            'history_actions': [],
            'target_action': '',
        }


class VLAStepDatasetCollator:
    def __init__(self, tokenizer, ignore_index=-100):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index

    def __call__(self, instances):
        batch = {k: [inst[k] for inst in instances] for k in instances[0].keys()}

        input_ids = torch.nn.utils.rnn.pad_sequence(batch['input_ids'], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(batch['labels'], batch_first=True, padding_value=self.ignore_index)

        pixel_values_list = [pv['pixel_values'] if isinstance(pv, dict) else pv for pv in batch['pixel_values']]
        max_pixel_len = max(pv.shape[0] for pv in pixel_values_list)
        pixel_values_padded = []
        for pv in pixel_values_list:
            pad_len = max_pixel_len - pv.shape[0]
            if pad_len > 0:
                padding = torch.zeros(pad_len, *pv.shape[1:], dtype=pv.dtype)
                pv = torch.cat([pv, padding], dim=0)
            pixel_values_padded.append(pv)
        pixel_values = torch.stack(pixel_values_padded)

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
        }


def convert_trajectory_to_vla_steps(era_data_path, output_path, images_folder=""):
    """将 ERA 轨迹数据转换为时序 VLA 格式"""
    print(f"Loading ERA data from {era_data_path}...")

    if era_data_path.endswith('.jsonl'):
        with open(era_data_path, 'r') as f:
            era_data = [json.loads(line) for line in f]
    else:
        with open(era_data_path, 'r') as f:
            era_data = json.load(f)

    print(f"Loaded {len(era_data)} samples, converting to VLA steps...")

    action_set = set()
    vla_samples = []

    for item in era_data:
        instruction = ""
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
                    actions = re.findall(r"['\"](.*?)['\"]", actions_str)

        if not instruction or not actions:
            continue

        image = item.get('image', '')
        if images_folder and image:
            image = os.path.join(images_folder, image)

        for action in actions:
            action_set.add(action.lower().strip())

        steps = []
        prev_image = image
        for action in actions:
            steps.append({
                "image": prev_image,
                "action": action
            })

        if steps:
            vla_samples.append({
                "instruction": instruction,
                "steps": steps
            })

    action_vocab = sorted(list(action_set))
    print(f"Found {len(action_vocab)} unique actions")
    print(f"Converted {len(vla_samples)} VLA samples")

    output_data = {
        "action_vocab": action_vocab,
        "samples": vla_samples,
    }

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Saved to {output_path}")
    return action_vocab, vla_samples
