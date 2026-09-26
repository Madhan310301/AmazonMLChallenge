from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

from . import __version__


def file_fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=str):
        digest.update(str(path.name).encode("utf-8"))
        if path.exists():
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:
            digest.update(b"<missing>")
    return digest.hexdigest()


def cache_metadata(
    *,
    dataset_fingerprint: str,
    source_name: str,
    row_count: int,
    columns: list[str],
    configuration: dict,
    feature_version: str,
) -> dict:
    config_blob = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    return {
        "dataset_fingerprint": dataset_fingerprint,
        "source_name": source_name,
        "row_count": row_count,
        "column_schema": columns,
        "preprocessing_version": "unicode-name-address-v1",
        "embedding_model_name": None,
        "blocking_configuration_hash": hashlib.sha256(
            config_blob.encode("utf-8")
        ).hexdigest(),
        "feature_version": feature_version,
        "software_version": __version__,
        "python_version": platform.python_version(),
    }


class CacheStore:
    def __init__(self, root: Path, enabled: bool = True):
        self.root = root
        self.enabled = enabled
        self.root.mkdir(parents=True, exist_ok=True)

    def read_json(self, name: str, expected_metadata: dict) -> Any | None:
        if not self.enabled:
            return None
        path = self.root / f"{name}.json"
        if not path.is_file():
            return None
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if blob.get("metadata") != expected_metadata:
            return None
        return blob.get("payload")

    def write_json(self, name: str, metadata: dict, payload: Any) -> None:
        if not self.enabled:
            return
        path = self.root / f"{name}.json"
        temp_path = path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(
                {"metadata": metadata, "payload": payload},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        temp_path.replace(path)

    def clear(self) -> None:
        if not self.root.exists():
            return
        for path in self.root.iterdir():
            if path.is_file() and path.name != ".gitkeep":
                path.unlink()