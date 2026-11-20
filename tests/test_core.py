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
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Generate sample dialogue from the trained GPT.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Apply dynamic and static post-training quantization to the model.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Measure and compare the original and quantized model sizes.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")


# Milestone 6: Simulate inference speed by comparing operations on original versus quantized weights.
@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")

