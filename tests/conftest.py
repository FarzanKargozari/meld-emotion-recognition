import csv

import pytest

COLUMNS = ["Sr No.", "Utterance", "Speaker", "Emotion", "Sentiment", "Dialogue_ID", "Utterance_ID",
           "Season", "Episode", "StartTime", "EndTime"]


@pytest.fixture
def transcript(tmp_path):
    """A small transcript in the official CSV layout, including a gap in one dialogue's IDs."""
    rows = [
        (1, "Hi there.", "Ross", "neutral", 0, 0),
        (2, "What?!  No way…", "Rachel", "surprise", 0, 1),
        (3, "I’m so happy!", "Ross", "joy", 0, 3),
        (4, "Other dialogue.", "Joey", "neutral", 1, 0),
        (5, "Get out!", "Monica", "anger", 1, 1),
    ]
    with (tmp_path / "train_sent_emo.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for number, text, speaker, emotion, dialogue, index in rows:
            writer.writerow([number, text, speaker, emotion, "neutral", dialogue, index, 1, 1, "0", "1"])
    return tmp_path
