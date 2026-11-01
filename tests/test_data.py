import pytest

from meld_emotion.data import LABELS, build_examples, class_weights, clean_text, read_split, render


def utterances(directory):
    return read_split("train", directory, verify=False)


def by_key(examples):
    return {example.utterance.key: example for example in examples}


def test_clean_text_keeps_case_and_punctuation():
    assert clean_text("What?!  No  way…") == "What?! No way..."
    assert clean_text("I’m “fine” — really") == "I'm \"fine\" - really"


def test_context_holds_only_earlier_lines(transcript):
    examples = by_key(build_examples(utterances(transcript), context_size=2))
    assert [item.index for item in examples["0_3"].context] == [0, 1]
    for example in examples.values():
        assert all(item.index < example.utterance.index for item in example.context)


def test_context_never_crosses_dialogues(transcript):
    for example in build_examples(utterances(transcript), context_size=10):
        assert all(item.dialogue == example.utterance.dialogue for item in example.context)


def test_first_line_has_no_context(transcript):
    examples = by_key(build_examples(utterances(transcript), context_size=3))
    assert examples["0_0"].context == ()


def test_context_size_limits_history(transcript):
    examples = by_key(build_examples(utterances(transcript), context_size=1))
    assert [item.index for item in examples["0_3"].context] == [1]


def test_zero_context_is_empty(transcript):
    assert all(example.context == () for example in build_examples(utterances(transcript), 0))


def test_gap_in_utterance_ids_is_kept_in_order(transcript):
    assert [example.utterance.index for example in build_examples(utterances(transcript), 3)][:3] == [0, 1, 3]


def test_negative_context_is_rejected(transcript):
    with pytest.raises(ValueError):
        build_examples(utterances(transcript), -1)


def test_render_without_speakers_hides_names(transcript):
    example = by_key(build_examples(utterances(transcript), 2))["0_3"]
    context, target = render(example, use_speakers=False)
    assert context == "Hi there. What?! No way..."
    assert target == "I'm so happy!"
    assert "Ross" not in context + target


def test_render_with_speakers_labels_each_line(transcript):
    example = by_key(build_examples(utterances(transcript), 1))["0_3"]
    assert render(example, use_speakers=True) == ("Rachel: What?! No way...", "Ross: I'm so happy!")


def test_render_never_exposes_a_label(transcript):
    for example in build_examples(utterances(transcript), 3):
        context, target = render(example, use_speakers=True)
        assert example.utterance.emotion not in (context + " " + target).lower().split()


def test_unknown_emotion_is_rejected(transcript):
    path = transcript / "train_sent_emo.csv"
    path.write_text(path.read_text(encoding="utf-8").replace("anger", "boredom"), encoding="utf-8")
    with pytest.raises(ValueError):
        utterances(transcript)


def test_official_counts_are_enforced(transcript):
    with pytest.raises(ValueError):
        read_split("train", transcript, verify=True)


def test_class_weights_favor_rare_classes():
    weights = class_weights([0] * 100 + [1] * 4)
    assert len(weights) == len(LABELS)
    assert weights[1] > weights[0]
    assert sum(weights) / len(weights) == pytest.approx(1.0)
