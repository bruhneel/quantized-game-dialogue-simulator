"""Local data preparation, language modeling, and quantization."""
from dataclasses import dataclass
import random
import unicodedata

import torch
from torch import Tensor, nn
from torch.nn import functional as F

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def clean_dialogue(text: str) -> list[str]:
    """Normalize punctuation and whitespace, preserving first occurrences."""
    lines = unicodedata.normalize("NFKC", text).translate(_QUOTES).splitlines()
    normalized = (" ".join(line.split()) for line in lines)
    return list(dict.fromkeys(line for line in normalized if line))


@dataclass(frozen=True)
class Tokenizer:
    """A reversible character vocabulary with a reserved unknown token."""

    alphabet: str

    @classmethod
    def fit(cls, lines: list[str]) -> "Tokenizer":
        return cls("".join(sorted(set("\n".join(lines)) | {"\n"})))

    @property
    def size(self) -> int:
        return len(self.alphabet) + 1

    def encode(self, text: str) -> list[int]:
        lookup = {char: i for i, char in enumerate(self.alphabet, 1)}
        return [lookup.get(char, 0) for char in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.alphabet[i - 1] if i else "�" for i in ids)


def split_dialogue(lines: list[str], seed: int = 42) -> dict[str, list[str]]:
    """Split unique lines into disjoint training, validation and calibration sets."""
    if len(lines) < 10:
        raise ValueError("At least ten clean dialogue lines are required.")
    shuffled = list(lines)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled) // 10
    return {"train": shuffled[2 * n:], "val": shuffled[:n], "calibration": shuffled[n:2 * n]}
@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    context: int = 64
    width: int = 96
    heads: int = 4
    layers: int = 2


class CausalAttention(nn.Module):
    """Multi-head attention with a causal mask shared by all heads."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.width % config.heads:
            raise ValueError("Model width must be divisible by the number of heads.")
        self.heads = config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.output = nn.Linear(config.width, config.width)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).reshape(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.output(attended.transpose(1, 2).reshape(batch, length, width))


class Block(nn.Module):
    """Pre-normalized attention and feed-forward residual sublayers."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.width)
        self.attention = CausalAttention(config)
        self.norm2 = nn.LayerNorm(config.width)
        self.mlp = nn.Sequential(
            nn.Linear(config.width, 4 * config.width), nn.GELU(),
            nn.Linear(4 * config.width, config.width)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    """Decoder-only model returning next-character logits at every position."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.tokens = nn.Embedding(config.vocab_size, config.width)
        self.positions = nn.Embedding(config.context, config.width)
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.layers)))
        self.norm = nn.LayerNorm(config.width)
        self.head = nn.Linear(config.width, config.vocab_size, bias=False)

    def forward(self, ids: Tensor) -> Tensor:
        positions = torch.arange(ids.shape[1], device=ids.device)
        return self.head(self.norm(self.blocks(self.tokens(ids) + self.positions(positions))))


def batch_from(
    tokens: Tensor, context: int, batch_size: int, rng: torch.Generator
) -> tuple[Tensor, Tensor]:
    if len(tokens) <= context:
        raise ValueError("Token stream must be longer than the context.")
    starts = torch.randint(len(tokens) - context, (batch_size, 1), generator=rng)
    offsets = starts + torch.arange(context + 1)
    windows = tokens[offsets]
    return windows[:, :-1], windows[:, 1:]


def lm_loss(model: GPT, x: Tensor, targets: Tensor) -> Tensor:
    logits = model(x)
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


def train_model(
    model: GPT, train: Tensor, val: Tensor, steps: int, batch_size: int, seed: int,
    *, log: bool = True
) -> list[dict[str, float | int]]:
    """Train with AdamW and evaluate on fixed held-out windows."""
    if steps < 1 or batch_size < 1:
        raise ValueError("Steps and batch size must be positive.")
    rng = torch.Generator().manual_seed(seed)
    context = model.config.context
    vx, vy = batch_from(val, context, 32, torch.Generator().manual_seed(seed + 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    history = []
    for step in range(1, steps + 1):
        model.train()
        x, y = batch_from(train, context, batch_size, rng)
        optimizer.zero_grad(set_to_none=True)
        loss = lm_loss(model, x, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step == steps or step % max(1, steps // 5) == 0:
            model.eval()
            with torch.no_grad():
                val_loss = lm_loss(model, vx, vy).item()
            history.append({"step": step, "train_loss": loss.item(), "val_loss": val_loss})
            if log:
                print(f"step {step:4d} train {loss.item():.4f} val {val_loss:.4f}")
    return history


@torch.inference_mode()
def generate_ids(
    model: GPT, prompt: list[int], tokens: int, temperature: float = 0.8,
    top_k: int = 12, seed: int = 42
) -> list[int]:
    """Sample one token at a time from the rolling context window."""
    if not prompt or tokens < 0 or temperature <= 0 or top_k < 1:
        raise ValueError("Provide a prompt, non-negative token count, and positive temperature/top-k.")
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    ids = list(prompt)
    for _ in range(tokens):
        window = torch.tensor([ids[-model.config.context:]])
        scores = model(window)[0, -1] / temperature
        values, candidates = scores.topk(min(top_k, len(scores)))
        sampled = torch.multinomial(values.softmax(dim=0), 1, generator=generator)
        ids.append(int(candidates[sampled].item()))
    return ids[len(prompt):]


def affine_qparams(low: Tensor, high: Tensor) -> tuple[Tensor, Tensor]:
    """Map a range containing zero onto signed int8."""
    low = low.clamp_max(0)
    high = high.clamp_min(0)
    scale = ((high - low) / 255).clamp_min(torch.finfo(low.dtype).eps)
    zero = (-128 - low / scale).round().clamp(-128, 127).to(torch.int32)
    return scale, zero


def quantize_affine(x: Tensor, scale: Tensor, zero: Tensor) -> Tensor:
    """Round and saturate a real-valued tensor to signed int8."""
    return (x / scale + zero).round().clamp(-128, 127).to(torch.int8)


def dequantize_affine(x: Tensor, scale: Tensor, zero: Tensor) -> Tensor:
    """Reconstruct approximate real values from an affine int8 tensor."""
    return (x.to(torch.int32) - zero) * scale


class Int8Linear(nn.Module):
    """CPU reference int8 linear with signed storage and int32 accumulation."""

    def __init__(
        self, linear: nn.Linear, activation_range: tuple[Tensor, Tensor] | None
    ) -> None:
        super().__init__()
        weight = linear.weight.detach()
        scale = (weight.abs().amax(dim=1, keepdim=True) / 127).clamp_min(1e-8)
        self.register_buffer("weight", (weight / scale).round().clamp(-127, 127).to(torch.int8))
        self.register_buffer("weight_scale", scale.flatten())
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().clone())
        if activation_range is None:
            self.register_buffer("activation_scale", None)
            self.register_buffer("activation_zero", None)
        else:
            scale, zero = affine_qparams(*activation_range)
            self.register_buffer("activation_scale", scale)
            self.register_buffer("activation_zero", zero)

    def forward(self, x: Tensor) -> Tensor:
        if self.activation_scale is None:
            scale, zero = affine_qparams(x.amin(), x.amax())
        else:
            scale, zero = self.activation_scale, self.activation_zero
        integer_input = quantize_affine(x, scale, zero).to(torch.int32) - zero
        integer_weight = self.weight.to(torch.int32)
        output = (integer_input @ integer_weight.T).to(x.dtype) * (scale * self.weight_scale)
        return output if self.bias is None else output + self.bias


@torch.no_grad()
def quantized_model(
    model: GPT, variant: str, calibration: list[Tensor] | None = None
) -> GPT:
    """Quantize linear layers. A missing calibration creates a reload skeleton."""
    from copy import deepcopy

    if variant not in ("dynamic", "static"):
        raise ValueError(f"Unknown quantization variant: {variant}")
    result = deepcopy(model).eval()
    layers = {name: layer for name, layer in result.named_modules() if isinstance(layer, nn.Linear)}
    ranges: dict[str, tuple[Tensor, Tensor]] = {}

    def observe(name: str, inputs: tuple[Tensor, ...]) -> None:
        low, high = inputs[0].amin(), inputs[0].amax()
        if name in ranges:
            previous_low, previous_high = ranges[name]
            low = torch.minimum(previous_low, low)
            high = torch.maximum(previous_high, high)
        ranges[name] = low, high

    if variant == "static" and calibration is not None:
        if not calibration:
            raise ValueError("Static quantization requires calibration examples.")
        handles = [
            layer.register_forward_pre_hook(lambda module, inputs, key=name: observe(key, inputs))
            for name, layer in layers.items()
        ]
        try:
            for batch in calibration:
                result(batch)
        finally:
            for handle in handles:
                handle.remove()

    for name, layer in layers.items():
        activation_range = (
            ranges.get(name, (torch.tensor(-1.0), torch.tensor(1.0)))
            if variant == "static" else None
        )
        result.set_submodule(name, Int8Linear(layer, activation_range))
    return result


def model_size(model: GPT) -> dict:
    """Count persisted weight, bias, and quantization metadata bytes by dtype."""
    dtypes: dict[str, int] = {}
    weight_count = 0
    weight_bytes = 0
    metadata_bytes = 0
    for name, tensor in model.state_dict().items():
        count = tensor.numel()
        size = count * tensor.element_size()
        dtype = str(tensor.dtype).removeprefix("torch.")
        dtypes[dtype] = dtypes.get(dtype, 0) + size
        if name.endswith((".weight_scale", ".activation_scale", ".activation_zero")):
            metadata_bytes += size
        else:
            weight_count += count
            weight_bytes += size
    return {
        "parameter_count": weight_count,
        "weight_bytes": weight_bytes,
        "metadata_bytes": metadata_bytes,
        "model_bytes": weight_bytes + metadata_bytes,
        "bytes_by_dtype": dtypes,
    }


# --- Milestone 6: Simulate inference speed by comparing operations on original versus quantized weights.
# Hint: Estimate computational cost based on data types.
def milestone_6():  # TODO(milestone 6)
    raise NotImplementedError("TODO(milestone 6)")
