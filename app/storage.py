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
