import os, math, torch, warnings, numpy as np, logging, contextlib, io
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers import SiglipImageProcessor, SiglipVisionModel, logging as hf_logging
from .model_minimind import *


class VLAConfig(MiniMindConfig):
    model_type = "minimind-vla"
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
        self.action_stop_ids = kwargs.get("action_stop_ids", [26, 234, 234])


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
    def __init__(self, in_dim, out_dim, source_tokens=64, target_tokens=64):
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
        self.action_proj = nn.Sequential(
            nn.Linear(config.action_hidden_size, config.action_hidden_size),
            nn.GELU(),
            nn.Linear(config.action_hidden_size, config.action_hidden_size),
            RMSNorm(config.action_hidden_size, eps=config.rms_norm_eps)
        )
        self.embed_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.action_hidden_size),
            RMSNorm(config.action_hidden_size, eps=config.rms_norm_eps)
        )
        self.text_scale, self.action_scale = nn.Parameter(torch.tensor(3.0)), nn.Parameter(torch.tensor(1.0))
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.action_config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)


class MiniMindVLA(MiniMindForCausalLM):
    config_class = VLAConfig
    def __init__(self, config: VLAConfig = None, vision_model_path="./model/siglip2-base-p32-256-ve"):
        config = config or VLAConfig()
        super().__init__(config)
        object.__setattr__(self, 'thinker', self.model)
        object.__setattr__(self.model, 'lm_head', self.lm_head)
        self.action = ActionModule(config)
        self.vision_proj = MMVisionProjector(config.image_hidden_size, config.hidden_size, target_tokens=config.image_token_len)
        self.action_pad_token, self.action_stop_token = config.action_pad_token, config.action_stop_token
        vision_encoder, vision_processor = self.load_vision(vision_model_path)
        object.__setattr__(self, 'vision_encoder', vision_encoder)
        object.__setattr__(self, 'vision_processor', vision_processor)

    @staticmethod
    def load_vision(path):
        if path is None or not os.path.exists(path):
            warnings.warn(f"[MiniMindVLA] Vision model path not found: {path}. vision_encoder will be None!")
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

    @torch.compiler.disable
    def get_image_embeddings(self, image_inputs):
        if hasattr(image_inputs, 'keys'):
            image_inputs = {k: v.squeeze(1) if v.ndim > 2 and v.shape[1] == 1 else v for k, v in image_inputs.items()}
            pixel_attention_mask = image_inputs.get('pixel_attention_mask')
            if pixel_attention_mask is not None and not pixel_attention_mask.any():
                pv = image_inputs['pixel_values']
                return pv.new_zeros(pv.size(0), pv.size(1), self.config.image_hidden_size)
        with torch.no_grad():
            outputs = self.vision_encoder(**image_inputs)
        return outputs.last_hidden_state

    @torch.compiler.disable
    def encode_image_inputs(self, pixel_values):
        if pixel_values is None or self.vision_encoder is None:
            return None
        mask = pixel_values.flatten(1).any(1)
        if not mask.any():
            return pixel_values.new_zeros(pixel_values.size(0), self.config.image_token_len, self.config.hidden_size)
        with torch.no_grad():
            emb = self.vision_encoder(pixel_values=pixel_values[mask]).last_hidden_state
        if emb.dim() == 2:
            emb = emb.unsqueeze(0)
        emb = self.vision_proj(emb)
        if mask.all():
            return emb
        idx = mask.nonzero().view(-1, 1, 1).expand_as(emb)
        return emb.new_zeros(pixel_values.size(0), *emb.shape[1:]).scatter(0, idx, emb)

    @torch.compiler.disable
    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):
        if vision_tensors is None or not self.config.image_ids:
            return h
        marker, vf = self.config.image_ids[0], vision_tensors
        if vf.dim() == 3:
            vf = vf.unsqueeze(1)
        out = []
        for b in range(h.size(0)):
            hb, seq, k, i = h[b], tokens[b].tolist(), 0, 0
            while i < len(seq):
                if seq[i] == marker:
                    start = i
                    while i < len(seq) and seq[i] == marker:
                        i += 1
                    if k < vf.size(1):
                        hb = torch.cat((hb[:start], vf[b][k][:i - start], hb[i:]), dim=0)[:seqlen]
                        k += 1
                else:
                    i += 1
            out.append(hb)
        return torch.stack(out)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, pixel_values=None, **args):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        n_thinker, n_action = len(self.thinker.layers), len(self.action.layers)
        past_key_values = past_key_values or ([None] * (n_thinker + n_action))
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        if self.thinker.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.thinker.freqs_cos, self.thinker.freqs_sin = freqs_cos.to(input_ids.device), freqs_sin.to(input_ids.device)
        if self.action.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.action.action_config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.action.freqs_cos, self.action.freqs_sin = freqs_cos.to(input_ids.device), freqs_sin.to(input_ids.device)
        presents = []

        hidden_states = self.thinker.dropout(self.thinker.embed_tokens(input_ids))
        position_embeddings = (self.thinker.freqs_cos[start_pos:start_pos + seq_length], self.thinker.freqs_sin[start_pos:start_pos + seq_length])

        if pixel_values is not None and start_pos == 0:
            if hasattr(pixel_values, 'keys'):
                img_emb = self.get_image_embeddings(pixel_values).to(hidden_states.dtype)
                vision_tensors = self.vision_proj(img_emb)
            else:
                if len(pixel_values.shape) == 6:
                    pixel_values = pixel_values.squeeze(2)
                if len(pixel_values.shape) == 4:
                    pixel_values = pixel_values.unsqueeze(1)
                bs, num, c, im_h, im_w = pixel_values.shape
                stack_dim = 1 if bs > 1 else 0
                vision_tensors = torch.stack([
                    self.encode_image_inputs(pixel_values[:, i, :, :, :])
                    for i in range(num)
                ], dim=stack_dim)
            hidden_states = self.count_vision_proj(tokens=input_ids, h=hidden_states, vision_tensors=vision_tensors, seqlen=seq_length)

        bridge_states = hidden_states
        for i, (layer, past_key_value) in enumerate(zip(self.thinker.layers, past_key_values[:n_thinker])):
            hidden_states, present = layer(hidden_states, position_embeddings, past_key_value=past_key_value, use_cache=use_cache, attention_mask=attention_mask)
            presents.append(present)
            if i == self.config.bridge_layer:
                bridge_states = hidden_states
        h_thinker = self.thinker.norm(hidden_states)

        action_emb = self.action.embed_tokens(input_ids)
        hidden_states = self.action.embed_proj(bridge_states) * self.action.text_scale + self.action.action_proj(action_emb) * self.action.action_scale
        action_pos_emb = (self.action.freqs_cos[start_pos:start_pos + seq_length], self.action.freqs_sin[start_pos:start_pos + seq_length])
        for layer, past_key_value in zip(self.action.layers, past_key_values[n_thinker:]):
            hidden_states, present = layer(hidden_states, action_pos_emb, past_key_value=past_key_value, use_cache=use_cache, attention_mask=attention_mask)
            presents.append(present)
        h_action = self.action.norm(hidden_states)

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        aux_loss = sum(l.mlp.aux_loss for l in list(self.thinker.layers) + list(self.action.layers) if isinstance(l.mlp, MOEFeedForward))
        aux_loss += sum(p.sum() for p in self.vision_proj.parameters()) * 0
        text_logits = self.thinker.lm_head(h_thinker[:, slice_indices, :])
        action_logits = self.action.lm_head(h_action[:, slice_indices, :])

        out = MoeCausalLMOutputWithPast(aux_loss=aux_loss, logits=text_logits, past_key_values=presents)
        out.action_logits = action_logits
        return out

    @torch.inference_mode()
    def generate_actions(self, input_ids, pixel_values=None, max_new_tokens=128, temperature=0.7, top_p=0.9, use_cache=True, eos_token_id=None):
        if eos_token_id is None:
            eos_token_id = self.config.action_stop_token

        start_pos = input_ids.shape[1]
        past_kvs = None
        finished = [False] * input_ids.shape[0]

        while input_ids.shape[1] < start_pos + max_new_tokens:
            out = self.forward(
                input_ids,
                pixel_values=pixel_values,
                past_key_values=past_kvs,
                use_cache=use_cache,
            )
            past_kvs = out.past_key_values

            logits = out.action_logits[0, -1, :].clone() / (temperature + 1e-9)
            if top_p and top_p < 1.0:
                sorted_l, sorted_i = torch.sort(logits, descending=True)
                mask = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1) > top_p
                mask[1:], mask[0] = mask[:-1].clone(), False
                logits[sorted_i[mask]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            if next_token == eos_token_id:
                finished[0] = True
                break

            input_ids = torch.cat((input_ids, torch.tensor([[next_token]], device=input_ids.device)), dim=1)

            if finished[0]:
                break

        return input_ids[:, start_pos:]

    @torch.inference_mode()
    def generate(self, input_ids, pixel_values=None, max_new_tokens=128, temperature=0.7, top_p=0.9, use_cache=True, return_actions=False, **args):
        if return_actions:
            return self.generate_actions(input_ids, pixel_values, max_new_tokens, temperature, top_p, use_cache)
        return self.generate_actions(input_ids, pixel_values, max_new_tokens, temperature, top_p, use_cache)
