import os, math, torch, warnings, numpy as np
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers import SiglipImageProcessor, SiglipVisionModel, logging as hf_logging
from .model_minimind import *


class VLAConfig(MiniMindConfig):
    model_type = "minimind-vla-step"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_action_hidden_layers = kwargs.get("num_action_hidden_layers", 4)
        self.action_hidden_size = kwargs.get("action_hidden_size", 768)
        self.action_ids = kwargs.get("action_ids", [16])
        self.action_special_token = kwargs.get("action_special_token", "<|action_pad|>")
        self.action_hidden_dim = kwargs.get("action_hidden_dim", 512)
        self.action_vocab_size = kwargs.get("action_vocab_size", 512)
        self.action_pad_token = kwargs.get("action_pad_token", 2049)
        self.action_stop_token = kwargs.get("action_stop_token", 2050)
        self.image_ids = kwargs.get("image_ids", [12])
        self.image_special_token = kwargs.get("image_special_token", "<|image_pad|>")
        self.image_hidden_size = kwargs.get("image_hidden_size", 768)
        self.image_token_len = kwargs.get("image_token_len", 64)
        self.bridge_layer = kwargs.get("bridge_layer", self.num_hidden_layers // 2 - 1)


class MMActionProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, x):
        return self.mlp(x)


class MMVisionProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, x):
        return self.mlp(x)


class ActionHead(nn.Module):
    def __init__(self, in_features, out_features, rank=256):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.adapter = nn.Sequential(
            nn.Linear(in_features, rank, bias=False),
            nn.GELU(),
            nn.Linear(rank, out_features, bias=False)
        )
    def forward(self, x):
        return self.base(x) + self.adapter(x)


class ActionEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, rank=256):
        super().__init__()
        self.base = nn.Embedding(num_embeddings, embedding_dim)
        self.adapter = nn.Sequential(
            nn.Embedding(num_embeddings, rank),
            nn.GELU(),
            nn.Linear(rank, embedding_dim, bias=False)
        )
    def forward(self, x):
        return self.base(x) + self.adapter(x)


class ActionModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.action_config = MiniMindConfig(hidden_size=config.action_hidden_size, use_moe=config.use_moe)
        self.layers = nn.ModuleList([MiniMindBlock(l, self.action_config) for l in range(config.num_action_hidden_layers)])
        self.norm = RMSNorm(config.action_hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ActionHead(config.action_hidden_size, config.action_vocab_size)
        self.embed_tokens = ActionEmbedding(config.action_vocab_size, config.action_hidden_size)
        self.embed_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.action_hidden_size),
            RMSNorm(config.action_hidden_size, eps=config.rms_norm_eps)
        )
        self.text_scale = nn.Parameter(torch.tensor(3.0))
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.action_config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)


class MiniMindVLAStep(nn.Module):
    """
    时序 VLA 模型：每个时间步接收 (图像 + 指令 + 历史动作) → 输出下一动作

    真正的时序控制能力：
    - 训练：给定前 N 个 (obs, action) 对，预测第 N+1 个动作
    - 推理：循环执行，观察 → 动作 → 观察 → 动作 → ...
    """

    config_class = VLAConfig

    def __init__(self, config: VLAConfig = None, vision_model_path="./model/siglip2-base-p32-256-ve"):
        config = config or VLAConfig()
        super().__init__()
        self.config = config

        self.thinker = MiniMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.thinker.lm_head = self.lm_head

        self.action = ActionModule(config)
        self.vision_proj = MMVisionProjector(config.image_hidden_size, config.hidden_size)
        self.action_pad_token = config.action_pad_token
        self.action_stop_token = config.action_stop_token

        self.action_id_to_token = {}
        self.action_token_to_id = {}

        vision_encoder, vision_processor = self._load_vision(vision_model_path)
        self.vision_encoder = vision_encoder
        self.vision_processor = vision_processor

    def resize_token_embeddings(self, new_num_tokens):
        """Resize token embeddings when tokenizer size changes"""
        if new_num_tokens is None or new_num_tokens == self.thinker.embed_tokens.num_embeddings:
            return
        # Resize thinker's embeddings
        old_emb = self.thinker.embed_tokens
        new_emb = nn.Embedding(new_num_tokens, old_emb.embedding_dim, device=old_emb.weight.device, dtype=old_emb.weight.dtype)
        num_copy = min(old_emb.num_embeddings, new_num_tokens)
        new_emb.weight.data[:num_copy] = old_emb.weight.data[:num_copy].clone()
        self.thinker.embed_tokens = new_emb
        # Resize lm_head if it shares weights
        if self.lm_head.weight.shape[0] == old_emb.num_embeddings:
            old_head = self.lm_head
            new_head = nn.Linear(old_head.in_features, new_num_tokens, bias=False, device=old_head.weight.device, dtype=old_head.weight.dtype)
            new_head.weight.data[:num_copy] = old_head.weight.data[:num_copy].clone()
            self.lm_head = new_head
            self.thinker.lm_head = self.lm_head

    def _load_vision(self, path):
        if path is None or not os.path.exists(path):
            warnings.warn(f"[MiniMindVLAStep] Vision model path not found: {path}")
            return None, None
        hf_logging.set_verbosity_error()
        try:
            model = SiglipVisionModel.from_pretrained(path)
        except (RuntimeError, ValueError):
            return None, None
        processor = SiglipImageProcessor.from_pretrained(path)
        for p in model.parameters():
            p.requires_grad = False
        return model.eval(), processor

    def set_action_vocabulary(self, action_vocab):
        self.action_id_to_token = {}
        self.action_token_to_id = {}
        for idx, action in enumerate(action_vocab):
            token_id = idx + 10
            self.action_id_to_token[token_id] = action
            self.action_token_to_id[action] = token_id
        self.action_id_to_token[0] = "<pad>"
        self.action_id_to_token[1] = "<stop>"
        self.action_id_to_token[2] = "<start>"
        self.action_token_to_id["<pad>"] = 0
        self.action_token_to_id["<stop>"] = 1
        self.action_token_to_id["<start>"] = 2

    def get_image_embeddings(self, pixel_values):
        if pixel_values is None or self.vision_encoder is None:
            return None
        with torch.no_grad():
            outputs = self.vision_encoder(**pixel_values)
        return outputs.last_hidden_state

    @torch.compiler.disable
    def inject_vision_features(self, input_ids, hidden_states, pixel_values, seqlen):
        if pixel_values is None:
            return hidden_states

        marker = self.config.image_ids[0]
        img_emb = self.get_image_embeddings(pixel_values)
        if img_emb is None:
            return hidden_states

        img_emb = self.vision_proj(img_emb)
        batch_size = hidden_states.size(0)
        out = []
        for b in range(batch_size):
            hb = hidden_states[b]
            seq = input_ids[b].tolist()
            new_hb = []
            i = 0
            while i < len(seq):
                if seq[i] == marker:
                    new_hb.append(img_emb[b])
                    i += 1
                else:
                    new_hb.append(hb[i])
                    i += 1
            if len(new_hb) < seqlen:
                new_hb = new_hb + [hb[len(new_hb):] if len(new_hb) < len(hb) else torch.zeros_like(hb[0])]
            out.append(torch.stack(new_hb[:seqlen]) if isinstance(new_hb[0], torch.Tensor) else torch.tensor(new_hb[:seqlen]))
        return torch.stack(out)

    def forward(self, input_ids, pixel_values=None, attention_mask=None, past_key_values=None, use_cache=False, **args):
        batch_size, seq_length = input_ids.shape

        if hasattr(past_key_values, 'layers'):
            past_key_values = None

        n_thinker = len(self.thinker.layers)
        n_action = len(self.action.layers)
        past_key_values = past_key_values or ([None] * (n_thinker + n_action))
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        if self.thinker.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=self.config.head_dim,
                end=self.config.max_position_embeddings,
                rope_base=self.config.rope_theta,
                rope_scaling=self.config.rope_scaling
            )
            self.thinker.freqs_cos = freqs_cos.to(input_ids.device)
            self.thinker.freqs_sin = freqs_sin.to(input_ids.device)

        if self.action.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=self.action.action_config.head_dim,
                end=self.config.max_position_embeddings,
                rope_base=self.config.rope_theta,
                rope_scaling=self.config.rope_scaling
            )
            self.action.freqs_cos = freqs_cos.to(input_ids.device)
            self.action.freqs_sin = freqs_sin.to(input_ids.device)

        presents = []

        hidden_states = self.thinker.dropout(self.thinker.embed_tokens(input_ids))
        position_embeddings = (self.thinker.freqs_cos[start_pos:start_pos + seq_length],
                               self.thinker.freqs_sin[start_pos:start_pos + seq_length])

        if pixel_values is not None and start_pos == 0:
            hidden_states = self.inject_vision_features(input_ids, hidden_states, pixel_values, seq_length)

        bridge_states = hidden_states
        for i, (layer, past_key_value) in enumerate(zip(self.thinker.layers, past_key_values[:n_thinker])):
            hidden_states, present = layer(hidden_states, position_embeddings, past_key_value=past_key_value, use_cache=use_cache, attention_mask=attention_mask)
            presents.append(present)
            if i == self.config.bridge_layer:
                bridge_states = hidden_states
        h_thinker = self.thinker.norm(hidden_states)

        action_ids_for_emb = input_ids.clone()
        action_ids_for_emb = action_ids_for_emb - 10
        action_ids_for_emb = action_ids_for_emb.clamp(0, self.config.action_vocab_size - 1)
        action_emb = self.action.embed_tokens(action_ids_for_emb)
        action_hidden = self.action.embed_proj(bridge_states) * self.action.text_scale + action_emb
        action_position_embeddings = (self.action.freqs_cos[start_pos:start_pos + seq_length],
                                     self.action.freqs_sin[start_pos:start_pos + seq_length])
        for layer, past_key_value in zip(self.action.layers, past_key_values[n_thinker:]):
            action_hidden, present = layer(action_hidden, action_position_embeddings, past_key_value=past_key_value, use_cache=use_cache, attention_mask=attention_mask)
            presents.append(present)
        h_action = self.action.norm(action_hidden)

        action_logits = self.action.lm_head(h_action)

        aux_loss = sum(l.mlp.aux_loss for l in list(self.thinker.layers) + list(self.action.layers) if isinstance(l.mlp, MOEFeedForward))

        return MoeCausalLMOutputWithPast(
            logits=action_logits,
            past_key_values=presents,
            aux_loss=aux_loss
        )

    def predict_action_logits(self, input_ids, pixel_values=None, use_cache=True):
        """
        获取动作 logits（不采样）
        输入: (图像, 指令, 历史动作) → 输出: action_logits [1, action_vocab_size]
        """
        out = self.forward(
            input_ids,
            pixel_values=pixel_values,
            past_key_values=None,
            use_cache=use_cache,
        )
        return out.logits[:, -1, :]  # [1, action_vocab_size]

    def predict_next_action(self, input_ids, pixel_values=None, max_new_tokens=1, temperature=0.7, top_p=0.9, use_cache=True):
        """
        预测下一个动作（单步）
        输入: (图像, 指令, 历史动作) → 输出: (action_idx, confidence)

        直接从 action_logits 采样，不需要文本解码
        """
        logits = self.predict_action_logits(input_ids, pixel_values, use_cache)

        if temperature < 0.01:
            # greedy
            action_idx = torch.argmax(logits[0]).item()
            confidence = torch.softmax(logits[0], dim=-1)[action_idx].item()
        else:
            logits = logits[0] / (temperature + 1e-9)
            if top_p and top_p < 1.0:
                sorted_l, sorted_i = torch.sort(logits, descending=True)
                mask = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1) > top_p
                mask[1:], mask[0] = mask[:-1].clone(), False
                logits[sorted_i[mask]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            action_idx = torch.multinomial(probs, 1).item()
            confidence = probs[action_idx].item()

        if action_idx >= self.config.action_vocab_size:
            action_idx = 8  # stay

        return action_idx, confidence

    def decode_action(self, token_id):
        if token_id in self.action_id_to_token:
            return self.action_id_to_token[token_id]
        if token_id == 0:
            return "<pad>"
        if token_id == 1:
            return "<stop>"
        if token_id == 2:
            return "<start>"
        return f"<action_{token_id}>"

    def encode_action(self, action):
        if action in self.action_token_to_id:
            return self.action_token_to_id[action]
        return hash(action.lower().strip()) % 500 + 10


class VLAController:
    """
    VLA 控制器：管理时序执行循环

    真正的时序控制流程：
    1. 获取当前观察 (图像)
    2. 构造输入：图像 + 指令 + 历史动作
    3. 模型预测下一动作
    4. 执行动作 → 获取执行结果/新观察
    5. 检查是否完成，循环或结束
    """

    def __init__(self, model: MiniMindVLAStep, tokenizer, max_steps=20, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        self.device = device

    def construct_input(self, instruction, history_actions, image_token="<|image_pad|>"):
        """构造模型输入"""
        prompt = f"Task: {instruction}"
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids

        input_ids = [self.tokenizer.bos_token_id] + prompt_ids + [2]  # 2 = <start>

        for action in history_actions:
            action_id = self.model.encode_action(action)
            input_ids.append(action_id)

        input_ids.append(1)  # 1 = <stop>

        return torch.tensor([input_ids], dtype=torch.long, device=self.device)

    def step(self, instruction, history_actions, pixel_values):
        """
        执行一步：给定当前观察和历史，预测下一动作

        Returns:
            action_str: 预测的动作字符串
            action_id: 预测的动作 ID
            confidence: 置信度
        """
        input_ids = self.construct_input(instruction, history_actions)

        if pixel_values is not None:
            if isinstance(pixel_values, dict):
                pixel_values = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in pixel_values.items()}
            else:
                pixel_values = pixel_values.to(self.device)

        results = list(self.model.predict_next_action(
            input_ids,
            pixel_values=pixel_values,
            max_new_tokens=5,
            temperature=0.7,
        ))

        if not results:
            return None, None, 0.0

        action_id, confidence = results[0]
        action_str = self.model.decode_action(action_id)

        return action_str, action_id, confidence

    def run_episode(self, instruction, get_observation_fn, execute_action_fn, is_done_fn=None):
        """
        运行一个完整的任务 episode

        Args:
            instruction: 任务指令
            get_observation_fn: 获取当前观察的函数 () -> pixel_values
            execute_action_fn: 执行动作的函数 (action_str) -> success
            is_done_fn: 检查是否完成的函数 (action_str, history) -> bool

        Returns:
            executed_actions: 执行的动作列表
            success: 是否成功完成任务
        """
        executed_actions = []
        history = []

        for step in range(self.max_steps):
            pixel_values = get_observation_fn()

            action_str, action_id, confidence = self.step(instruction, history, pixel_values)

            if action_str is None or action_str in ("<pad>", "<stop>", "<start>"):
                break

            executed_actions.append(action_str)
            history.append(action_str)

            success = execute_action_fn(action_str)

            if not success:
                break

            if is_done_fn and is_done_fn(action_str, history):
                break

        return executed_actions, len(executed_actions) > 0 and executed_actions[-1] != "<stop>"
