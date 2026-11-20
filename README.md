# Quantized Game Dialogue Simulator

[![Built with askThili](https://msrit.ask.thili.ai/api/v1/public/projects/rcxwvusdev/badge.svg)](https://msrit.ask.thili.ai/b/rcxwvusdev)

Train a 240,000-parameter GPT on original NPC dialogue, generate dialogue from prompts, and compare FP32 with dynamic and calibrated static INT8. The CLI runs entirely on CPU and requires no hosting or API keys.

## Setup

Install Python 3.12 and Git, then run from the repository root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# On Linux, install the CPU build of PyTorch before installing requirements.
# pip install 'torch>=2.5,<3' --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env
python -m app.cli hello
pytest -q
```

The optional Ollama, Groq and Gemini settings affect only the starter chat helper in `app/llm.py`. Keep `.env` local. The local GPT pipeline does not use it.

## Run the milestones

1. **Prepare original dialogue.** Run `python -m app.cli prepare-data` to clean and tokenize 300 original lines covering five NPC roles. The seeded split has 240 training, 30 validation and 30 calibration lines. `python data/build_corpus.py` regenerates the corpus.
2. **Train the GPT.** Run `python -m app.cli train`. The two-layer decoder has width 96, four attention heads and context 64. AdamW trains for 700 steps by default. On the reference run, held-out loss fell from 3.8222 to 0.4808 nats per token. Training history is in `results/training.json`.
3. **Generate dialogue.** Run `python -m app.cli generate --prompt "Guard:" --tokens 200`. Use `--temperature`, `--top-k`, and `--seed` to control sampling. The pipeline writes five unedited continuations to `results/samples_original.txt`.
4. **Quantize.** Run `python -m app.cli quantize`, then try `python -m app.cli generate --variant static --prompt "Shopkeeper:" --tokens 80`. Choose `dynamic` or `static` for INT8 generation.
5. **Report footprint.** Run `python -m app.cli report-size` for tensor and serialized checkpoint sizes. The Markdown and JSON reports are under `results/`.
6. **Compare speed.** Run `python -m app.cli simulate-speed` for a device estimate alongside measured laptop throughput. Both versions of the report are under `results/`.

Run the full pipeline with `python -m app.cli run-all`. Generated checkpoints and the processed dataset live in the ignored `artifacts/` directory. Defaults use fixed seeds, with possible numerical variation across PyTorch versions or CPU backends.

## Results

| Model | Persistent tensors (B) | Checkpoint (B) | Tensor compression | Simulated tok/s | Measured laptop tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp32 | 960,000 | 968,291 | 1.00× | 6.05 | 2319.41 |
| dynamic | 288,592 | 301,041 | 3.33× | 17.60 | 400.12 |
| static | 288,664 | 304,445 | 3.33× | 18.34 | 425.85 |

The footprint counts stored weights, biases, scales, and zero points. See [size details](results/size_report.md) and [benchmark details](results/speed_report.md) for units and throughput assumptions.

## Quantization and simulation

Both modes store each linear output channel's weights as signed INT8 with a symmetric scale. Dynamic quantization derives an affine activation scale and zero point from each forward input tensor. Static quantization fixes those parameters using min/max ranges recorded on the separate calibration split. The portable CPU reference path widens integers to INT32 for matrix multiplication, rescales to FP32, and leaves embeddings, normalization, and attention in FP32. This makes the quantized operations explicit and avoids dependence on a particular packed-kernel backend.

The simulation counts dominant linear and attention MACs for one generated character at a full context window with no KV cache. It assumes a hypothetical device with 100M FP32 MAC/s, 400M INT8 MAC/s and 200 MB/s of weight bandwidth. Conversion overhead is included as an estimate. The laptop measurement includes actual generation and sampling, and its INT32 reference kernels can be slower than FP32. Neither the model footprint nor the speed estimate is a measurement of peak process memory or performance on a specific handheld device.

This small character model learns the corpus style and can repeat templates or produce malformed text. It has no game-state grounding or factual guarantees.
