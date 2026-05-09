import os, sys, json, argparse, torch, re
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model_vla import MiniMindVLA, VLAConfig
from model.model_minimind import MiniMindConfig
from dataset.vla_dataset import VLADataset
from trainer.trainer_utils import load_tokenizer


def normalize_action(action):
    if not action:
        return ""
    action = action.strip().lower()
    action = re.sub(r'\s+', ' ', action)
    return action


def compute_sequence_accuracy(pred_actions, target_actions):
    if not target_actions:
        return 0.0

    pred_norm = [normalize_action(a) for a in pred_actions]
    target_norm = [normalize_action(a) for a in target_actions]

    if pred_norm == target_norm:
        return 1.0

    if len(target_norm) == 0:
        return 0.0

    correct = sum(1 for p, t in zip(pred_norm, target_norm) if p == t)
    return correct / len(target_norm)


def run_inference(args):
    device = torch.device(f"cuda:{args.local_rank}") if torch.cuda.is_available() and args.local_rank >= 0 else torch.device("cpu")

    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = load_tokenizer(args.tokenizer_path)

    print(f"Loading VLA model from {args.checkpoint_path}...")
    vla_config = VLAConfig(
        vocab_size=len(tokenizer),
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=8,
        num_attention_heads=12,
        num_key_value_heads=12,
        num_action_hidden_layers=4,
        action_hidden_size=768,
        action_vocab_size=args.action_vocab_size,
        max_position_embeddings=2048,
        use_moe=args.use_moe,
        num_experts=8,
        moe_intermediate_size=1408,
    )

    model = MiniMindVLA(vla_config, vision_model_path=args.vision_model_path)
    model.resize_token_embeddings(len(tokenizer))

    checkpoint_file = os.path.join(args.checkpoint_path, "pytorch_model.bin")
    if os.path.exists(checkpoint_file):
        state_dict = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        ckpt_files = [f for f in os.listdir(args.checkpoint_path) if f.endswith('.pt') or f.endswith('.pth')]
        if ckpt_files:
            state_dict = torch.load(os.path.join(args.checkpoint_path, ckpt_files[0]), map_location=device)
            if isinstance(state_dict, dict) and 'model' in state_dict:
                state_dict = state_dict['model']
            model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()

    print(f"Loading evaluation dataset from {args.eval_data_path}...")
    eval_dataset = VLADataset(
        data_path=args.eval_data_path,
        tokenizer=tokenizer,
        vision_processor=None if model.vision_encoder is None else model.vision_processor,
        max_length=args.max_seq_len,
        images_folder=args.images_folder,
    )

    print(f"Running inference on {len(eval_dataset)} samples...")

    all_results = []
    seq_accuracies = []
    exact_matches = 0

    for idx in tqdm(range(min(len(eval_dataset), args.max_samples))):
        item = eval_dataset.list_data_dict[idx]
        instruction = item.get('instruction', '')
        target_actions = item.get('actions', [])

        if not instruction:
            continue

        prompt = f"You are a household assistant. Task: {instruction}"
        prompt_with_image = f"{eval_dataset.image_token}\n{prompt}"

        input_ids = tokenizer(prompt_with_image, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        input_ids = torch.cat([torch.tensor([[tokenizer.bos_token_id]], device=device), input_ids], dim=1) if tokenizer.bos_token_id else input_ids

        pixel_values = None
        if 'image' in item and item['image']:
            pixel_values = eval_dataset.load_image_inputs(item['image'])
        if pixel_values is None:
            pixel_values = {'pixel_values': torch.zeros(1, 3, 256, 256).to(device)}
        if isinstance(pixel_values, dict):
            pixel_values = {k: v.to(device) if torch.is_tensor(v) else v for k, v in pixel_values.items()}
        else:
            pixel_values = pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate_actions(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                use_cache=True,
            )

        pred_action_ids = generated_ids[0].cpu().tolist()
        pred_actions = eval_dataset.decode_actions(pred_action_ids)

        seq_acc = compute_sequence_accuracy(pred_actions, target_actions)
        seq_accuracies.append(seq_acc)

        if pred_actions and target_actions:
            pred_norm = [normalize_action(a) for a in pred_actions]
            target_norm = [normalize_action(a) for a in target_actions]
            if pred_norm == target_norm:
                exact_matches += 1

        result = {
            "idx": idx,
            "instruction": instruction,
            "target_actions": target_actions,
            "predicted_actions": pred_actions,
            "seq_accuracy": seq_acc,
        }
        all_results.append(result)

        if idx < 5:
            print(f"\n--- Sample {idx} ---")
            print(f"Instruction: {instruction[:100]}...")
            print(f"Target: {target_actions[:5]}...")
            print(f"Predicted: {pred_actions[:5]}...")
            print(f"Seq Acc: {seq_acc:.4f}")

    avg_seq_acc = sum(seq_accuracies) / len(seq_accuracies) if seq_accuracies else 0.0
    exact_match_rate = exact_matches / len(seq_accuracies) if seq_accuracies else 0.0

    print(f"\n=== Evaluation Results ===")
    print(f"Total samples: {len(seq_accuracies)}")
    print(f"Sequence Accuracy: {avg_seq_acc:.4f}")
    print(f"Exact Match Rate: {exact_match_rate:.4f}")

    metrics = {
        "sequence_accuracy": avg_seq_acc,
        "exact_match_rate": exact_match_rate,
        "num_samples": len(seq_accuracies),
    }

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with open(args.output_path, 'w') as f:
            json.dump({
                "metrics": metrics,
                "samples": all_results,
            }, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to VLA model checkpoint")
    parser.add_argument("--eval_data_path", type=str, required=True, help="Path to evaluation dataset")
    parser.add_argument("--tokenizer_path", type=str, default="./model")
    parser.add_argument("--vision_model_path", type=str, default="./model/siglip2-base-p32-256-ve")
    parser.add_argument("--images_folder", type=str, default="", help="Base folder for images")
    parser.add_argument("--action_vocab_size", type=int, default=512)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--output_path", type=str, default="./eval_results.json")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--use_moe", type=int, default=0)
    args = parser.parse_args()

    run_inference(args)
