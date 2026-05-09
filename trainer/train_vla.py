import os, sys, json, time, argparse, contextlib
from datetime import datetime
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import deepspeed
from tqdm import tqdm
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model_vla import MiniMindVLA, VLAConfig
from model.model_minimind import MiniMindConfig
from dataset.vla_dataset import VLADataset, VLADatasetCollator
from trainer.trainer_utils import load_tokenizer, get_cosine_schedule_with_warmup


def torch_run(args):
    if args.deepspeed:
        deepspeed.init_distributed()
    else:
        if "WORLD_SIZE" in os.environ:
            args.local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(args.local_rank)
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.rank = int(os.environ["RANK"])
        else:
            args.local_rank = 0
            args.world_size = 1
            args.rank = 0

    if args.rank == 0 and args.use_tb:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"{args.output_dir}/logs_{timestamp}"
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None

    device = torch.device(f"cuda:{args.local_rank}") if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = load_tokenizer(args.tokenizer_path)
    special_tokens = {
        "additional_special_tokens": [
            "<|action_pad|>",
            "<|action_stop|>",
            "<|image_pad|>",
            "<|think_start|>",
            "<|think_end|>",
        ]
    }
    num_new_tokens = tokenizer.add_special_tokens(special_tokens)
    print(f"[Rank {args.rank}] Added {num_new_tokens} special tokens")

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
    model = model.to(device)

    train_dataset = VLADataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        vision_processor=None if model.vision_encoder is None else model.vision_processor,
        max_length=args.max_seq_len,
        images_folder=args.images_folder,
    )

    print(f"[Rank {args.rank}] Training dataset size: {len(train_dataset)}")
    print(f"[Rank {args.rank}] Action vocabulary size: {len(train_dataset.action_vocab)}")

    collator = VLADatasetCollator(tokenizer=tokenizer, ignore_index=-100)

    if args.deepspeed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True)
    else:
        train_sampler = RandomSampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
    )

    no_decay = ["bias", "layer_norm", "layernorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95))

    total_steps = len(train_loader) * args.epochs // max(1, args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    if args.deepspeed:
        ds_config = {
            "train_batch_size": args.batch_size * args.world_size,
            "train_micro_batch_size_per_gpu": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "steps_per_print": 10,
            "zero_optimization": {"stage": 2, "offload_optimizer": {"device": "cpu", "pin_memory": True}},
            "bf16": {"enabled": True},
            "gradient_clipping": args.clip_grad_norm,
            "wall_clock_breakdown": False,
        }
        model_engine, optimizer, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=ds_config)
        print(f"[Rank {args.rank}] Initialized DeepSpeed")
    else:
        scaler = GradScaler() if args.fp16 else None
        model_engine = model

    global_step = 0
    for epoch in range(args.epochs):
        if args.deepspeed:
            train_sampler.set_epoch(epoch)
        model_engine.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}") if args.rank == 0 else train_loader

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device) if "pixel_values" in batch else None

            with autocast(dtype=torch.bfloat16) if not args.deepspeed else contextlib.nullcontext():
                outputs = model_engine(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    use_cache=False,
                    logits_to_keep=1,
                )

                action_logits = outputs.action_logits

                shift_action_logits = action_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                valid_mask = shift_labels != -100
                if valid_mask.any():
                    loss_action = F.cross_entropy(
                        shift_action_logits.view(-1, shift_action_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                        reduction='mean'
                    )
                else:
                    loss_action = torch.tensor(0.0, device=device)

                if hasattr(outputs, 'aux_loss') and outputs.aux_loss is not None:
                    loss = loss_action + outputs.aux_loss * 0.01
                else:
                    loss = loss_action

            if args.deepspeed:
                model_engine.backward(loss)
                model_engine.step()
            else:
                if scaler:
                    scaler.scale(loss).backward()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    loss.backward()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                        optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            if args.rank == 0 and step % 10 == 0 and writer:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/loss_action", loss_action.item(), global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

            if args.rank == 0 and step % 50 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "loss_action": f"{loss_action.item():.4f}"})

        if (epoch + 1) % 1 == 0 and args.rank == 0:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model_engine.save_checkpoint(checkpoint_dir)
            print(f"\n[Rank {args.rank}] Saved checkpoint to {checkpoint_dir}")

    checkpoint_dir = os.path.join(args.output_dir, "final")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_engine.save_checkpoint(checkpoint_dir)
    print(f"[Rank {args.rank}] Training complete. Saved to {checkpoint_dir}")

    if writer:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to VLA dataset JSON")
    parser.add_argument("--output_dir", type=str, default="../out/vla")
    parser.add_argument("--tokenizer_path", type=str, default="../model")
    parser.add_argument("--vision_model_path", type=str, default="../model/siglip2-base-p32-256-ve")
    parser.add_argument("--action_vocab_path", type=str, default="", help="Path to action vocabulary JSON")
    parser.add_argument("--images_folder", type=str, default="", help="Base folder for images")
    parser.add_argument("--action_vocab_size", type=int, default=512, help="Size of action vocabulary")
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use_tb", action="store_true")
    parser.add_argument("--use_moe", type=int, default=0)
    parser.add_argument("--deepspeed", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    torch_run(args)
