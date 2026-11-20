"""A tiny key-value store in one JSON file (no server, easy to read). Values can be anything JSON can hold."""
import json
import os
from pathlib import Path


class Store:
    def __init__(self, path: str | None = None):
        self.path = Path(path or os.environ.get("DATA_PATH", "data/app.json"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}")

    def _read(self) -> dict:
        return json.loads(self.path.read_text() or "{}")

    def put(self, key: str, value) -> None:
        data = self._read()
        data[key] = value
        self.path.write_text(json.dumps(data, indent=2))

    def get(self, key: str, default=None):
        return self._read().get(key, default)

    def all(self) -> dict:
        return dict(sorted(self._read().items()))


def write_json(path: Path, data: dict) -> None:
    """Persist JSON to a local output directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def prepare_data() -> dict:
    """Prepare the original corpus and store the vocabulary and line splits."""
    from app.core import Tokenizer, clean_dialogue, split_dialogue

    splits = split_dialogue(clean_dialogue(Path("data/dialogue.txt").read_text(encoding="utf-8")))
    tokenizer = Tokenizer.fit(splits["train"])
    data = {"alphabet": tokenizer.alphabet, "splits": splits}
    write_json(Path("artifacts/dialogue.json"), data)
    return data


def load_data() -> dict:
    """Load the prepared dataset, creating it on the first run."""
    path = Path("artifacts/dialogue.json")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else prepare_data()


def train(steps: int = 700, seed: int = 42) -> None:
    """Train a seeded CPU model and save its state and training losses."""
    from dataclasses import asdict

    import torch

    from app.core import GPT, GPTConfig, Tokenizer, train_model

    torch.manual_seed(seed)
    torch.set_num_threads(2)
    data = load_data()
    tokenizer = Tokenizer(data["alphabet"])
    model = GPT(GPTConfig(tokenizer.size))
    streams = {
        name: torch.tensor(tokenizer.encode("\n".join(data["splits"][name]) + "\n"))
        for name in ("train", "val")
    }
    history = train_model(model, streams["train"], streams["val"], steps, 16, seed)
    torch.save({"config": asdict(model.config), "alphabet": tokenizer.alphabet,
                "state": model.state_dict()}, "artifacts/fp32.pt")
    write_json(Path("results/training.json"), {"seed": seed, "history": history})


def load_model(variant: str = "fp32") -> tuple:
    """Load a locally saved inference model and its tokenizer."""
    import torch

    from app.core import GPT, GPTConfig, Tokenizer, quantized_model

    path = Path("artifacts") / f"{variant}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing. Train and quantize before inference.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**checkpoint["config"]))
    if variant != "fp32":
        model = quantized_model(model, variant)
    model.load_state_dict(checkpoint["state"])
    return model.eval(), Tokenizer(checkpoint["alphabet"])


def generate(
    prompt: str, tokens: int, temperature: float, top_k: int, seed: int,
    variant: str = "fp32"
) -> str:
    """Return the prompt and sampled continuation."""
    import torch

    from app.core import generate_ids

    torch.set_num_threads(2)
    model, tokenizer = load_model(variant)
    continuation = generate_ids(model, tokenizer.encode(prompt), tokens, temperature, top_k, seed)
    return prompt + tokenizer.decode(continuation)


def save_samples() -> None:
    """Save representative, unedited continuations from five NPC prompts."""
    prompts = ("Guard:", "Shopkeeper:", "Quest Giver:", "Companion:", "Villain:")
    samples = [generate(prompt, 110, 0.65, 8, 42 + i) for i, prompt in enumerate(prompts)]
    path = Path("results/samples_original.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(samples) + "\n", encoding="utf-8")


def quantize() -> None:
    """Save independently loadable dynamic and static INT8 checkpoints."""
    from dataclasses import asdict

    import torch

    from app.core import quantized_model

    torch.set_num_threads(2)
    model, tokenizer = load_model()
    lines = load_data()["splits"]["calibration"]
    calibration = [
        torch.tensor([tokenizer.encode(line + "\n")[start:start + model.config.context]])
        for line in lines
        for start in range(0, len(line) + 1, model.config.context)
    ]
    for variant in ("dynamic", "static"):
        converted = quantized_model(model, variant, calibration)
        torch.save(
            {"config": asdict(model.config), "alphabet": tokenizer.alphabet,
             "state": converted.state_dict()},
            Path("artifacts") / f"{variant}.pt",
        )


VARIANTS = ("fp32", "dynamic", "static")


def report_size() -> str:
    """Write model tensor footprint and actual serialized file sizes."""
    from app.core import model_size

    rows = []
    for variant in VARIANTS:
        model, _ = load_model(variant)
        rows.append({"variant": variant, **model_size(model),
                     "file_bytes": (Path("artifacts") / f"{variant}.pt").stat().st_size})
    original = rows[0]
    for row in rows:
        row["tensor_compression"] = original["model_bytes"] / row["model_bytes"]
        row["file_compression"] = original["file_bytes"] / row["file_bytes"]
    path = Path("results/size_report.json")
    write_json(path, {"units": "bytes", "variants": rows})
    table = [
        "# Model size comparison", "",
        "Tensor bytes include persistent scales and zero points. File bytes include checkpoint overhead.",
        "The inventory excludes runtime activations and temporary buffers.", "",
        "| Model | Parameters | Weights (B) | Metadata (B) | Tensor total (B) | File (B) | Tensor ratio | File ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        table.append(
            f'| {row["variant"]} | {row["parameter_count"]:,} | {row["weight_bytes"]:,} | '
            f'{row["metadata_bytes"]:,} | {row["model_bytes"]:,} | {row["file_bytes"]:,} | '
            f'{row["tensor_compression"]:.2f}× | {row["file_compression"]:.2f}× |'
        )
    report = "\n".join(table) + "\n"
    path.with_suffix(".md").write_text(report, encoding="utf-8")
    return report


def report_speed() -> str:
    """Measure unbatched CPU generation and simulate the same full context."""
    import platform
    from statistics import median
    from time import perf_counter

    import torch

    from app.core import DEVICE_PROFILE, generate_ids, simulate_generation

    torch.set_num_threads(2)
    steps, repetitions = 80, 3
    rows = []
    for variant in VARIANTS:
        model, tokenizer = load_model(variant)
        text = "Guard: The north bridge is closed. Show your pass at the ferry dock.\n"
        prompt = tokenizer.encode(text)[-model.config.context:]
        generate_ids(model, prompt, 10)
        durations = []
        for _ in range(repetitions):
            start = perf_counter()
            generate_ids(model, prompt, steps)
            durations.append(perf_counter() - start)
        rows.append({
            "variant": variant, **simulate_generation(model, variant),
            "measured_seconds": durations,
            "measured_tokens_per_second": steps / median(durations),
        })

    host = f"{platform.system()} {platform.release()} {platform.machine()}"
    path = Path("results/speed_report.json")
    write_json(path, {
        "device_profile": DEVICE_PROFILE, "host": host, "torch_version": str(torch.__version__),
        "cpu_threads": 2, "new_tokens_per_repeat": steps, "rows": rows,
    })
    table = [
        "# Generation speed comparison", "",
        f"Measured on {host}, PyTorch {torch.__version__}, CPU with two threads.",
        f"Batch size one, context filled from the first step, 10 warm-up tokens, and median latency across {repetitions} runs of {steps} new tokens.", "",
        "| Model | FP32 MAC/token | INT8 MAC/token | FLOP-equivalent/token | Simulated tok/s | Laptop tok/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        table.append(
            f'| {row["variant"]} | {row["fp32_macs_per_token"]:,} | {row["int8_macs_per_token"]:,} | '
            f'{row["flop_equivalent_per_token"]:,} | {row["simulated_tokens_per_second"]:.2f} | '
            f'{row["measured_tokens_per_second"]:.2f} |'
        )
    table += [
        "", "Device assumptions: 100M FP32 MAC/s, 400M INT8 MAC/s, 200 MB/s of weight bandwidth, "
        "and 80M activation elements/s. Compute, weight reads, and activation conversion times are added.",
        "Dynamic quantization models two extra passes for range lookup. A full tensor footprint is charged per step.",
        "Attention uses a dense MAC estimate, including masked positions. Two arithmetic operations per MAC are counted.",
        "The simulator omits caching, cache reuse, kernel overhead and several elementwise ops.",
        "On the laptop, both INT8 paths widen packed weights and activations to int32 for reference matmul.",
    ]
    report = "\n".join(table) + "\n"
    path.with_suffix(".md").write_text(report, encoding="utf-8")
    return report
