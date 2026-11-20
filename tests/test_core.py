"""Regression tests for the dialogue pipeline."""
import pytest

def test_milestone_1():
    from app.core import Tokenizer, clean_dialogue, split_dialogue

    assert clean_dialogue("  Guard:  “Halt!”\n\nGuard: \"Halt!\"\n  Wait.") == [
        'Guard: "Halt!"', "Wait."
    ]
    lines = [f"Guard: report {i}" for i in range(30)]
    splits = split_dialogue(lines)
    assert splits == split_dialogue(lines)
    assert [len(splits[key]) for key in ("train", "val", "calibration")] == [24, 3, 3]
    assert sorted(sum(splits.values(), [])) == sorted(lines)
    tokenizer = Tokenizer.fit(lines)
    text = "\n".join(lines)
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_milestone_2():
    import torch

    from app.core import GPT, GPTConfig, train_model

    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = GPT(GPTConfig(5, context=8, width=16, heads=4, layers=1))
    ids = torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4]])
    changed = ids.clone()
    changed[:, 4:] = 0
    assert model(ids).shape == (1, 8, 5)
    torch.testing.assert_close(model(ids)[:, :4], model(changed)[:, :4])
    stream = ids.flatten().repeat(32)
    history = train_model(model, stream, stream, 50, 4, 42, log=False)
    assert history[-1]["val_loss"] < 0.3 * history[0]["val_loss"]


def test_milestone_3():
    import torch

    from app.core import GPT, GPTConfig, generate_ids

    torch.manual_seed(42)
    model = GPT(GPTConfig(6, context=8, width=16, heads=4, layers=1))
    args = (model, [1, 2], 17)
    ids = generate_ids(*args, top_k=3)
    assert len(ids) == 17
    assert all(0 <= token < 6 for token in ids)
    assert ids == generate_ids(*args, top_k=3)


@pytest.mark.parametrize("variant", ["dynamic", "static"])
def test_milestone_4(variant):
    import torch

    from app.core import (
        GPT, GPTConfig, Int8Linear, affine_qparams, dequantize_affine,
        generate_ids, quantize_affine, quantized_model,
    )

    torch.manual_seed(42)
    values = torch.linspace(-1, 2, 200)
    scale, zero = affine_qparams(values.min(), values.max())
    encoded = quantize_affine(values, scale, zero)
    assert encoded.dtype == torch.int8
    assert (dequantize_affine(encoded, scale, zero) - values).abs().max() <= scale * 0.501

    model = GPT(GPTConfig(8, context=8, width=16, heads=4, layers=1)).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    quantized = quantized_model(model, variant, [ids])
    assert isinstance(quantized.head, Int8Linear)
    assert quantized.head.weight.dtype == torch.int8
    assert torch.isfinite(quantized(ids)).all()
    torch.testing.assert_close(quantized(ids), model(ids), atol=0.04, rtol=0.08)
    assert len(generate_ids(quantized, [1, 2], 5)) == 5


def test_milestone_5():
    from app.core import GPT, GPTConfig, model_size, quantized_model

    model = GPT(GPTConfig(8, context=8, width=16, heads=4, layers=1))
    original = model_size(model)
    dynamic = model_size(quantized_model(model, "dynamic"))
    static = model_size(quantized_model(model, "static"))
    assert original["model_bytes"] == 4 * original["parameter_count"]
    assert original["parameter_count"] == dynamic["parameter_count"] == static["parameter_count"]
    assert dynamic["model_bytes"] < static["model_bytes"] < original["model_bytes"]
    assert sum(static["bytes_by_dtype"].values()) == static["model_bytes"]


def test_milestone_6():
    from app.core import GPT, GPTConfig, quantized_model, simulate_generation

    model = GPT(GPTConfig(8, context=8, width=16, heads=4, layers=1))
    fp32 = simulate_generation(model, "fp32")
    dynamic = simulate_generation(quantized_model(model, "dynamic"), "dynamic")
    static = simulate_generation(quantized_model(model, "static"), "static")

    assert fp32["fp32_macs_per_token"] == 8 * (12 * 16**2 + 16 * 8) + 2 * 8**2 * 16
    assert fp32["int8_macs_per_token"] == 0
    assert dynamic["fp32_macs_per_token"] == static["fp32_macs_per_token"] == 2 * 8**2 * 16
    assert fp32["flop_equivalent_per_token"] == dynamic["flop_equivalent_per_token"]
    assert dynamic["activation_elements_per_token"] == 3 * static["activation_elements_per_token"]
    assert static["simulated_tokens_per_second"] > dynamic["simulated_tokens_per_second"] > fp32["simulated_tokens_per_second"]
