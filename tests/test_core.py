"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest

# Milestone 1: Prepare a small game dialogue text corpus.
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


# Milestone 2: Implement a basic GPT architecture and train it on the corpus.
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


# Milestone 3: Generate sample dialogue from the trained GPT.
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


# Milestone 4: Apply dynamic and static post-training quantization to the model.
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


# Milestone 5: Measure and compare the original and quantized model sizes.
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


# Milestone 6: Simulate inference speed by comparing operations on original versus quantized weights.
@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")

