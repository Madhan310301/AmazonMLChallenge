from __future__ import annotations

import csv
from pathlib import Path
from typing import Generator

from .schema import resolve_column


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required TSV file not found: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if not reader.fieldnames:
                raise ValueError(f"TSV file has no header row: {path}")
            columns = [str(column).strip() for column in reader.fieldnames]
            if len(set(columns)) != len(columns):
                raise ValueError(f"TSV file has duplicate column names: {path}")
            rows = []
            for row_number, row in enumerate(reader, start=2):
                if None in row:
                    raise ValueError(
                        f"Malformed TSV row {row_number} in {path}: more fields "
                        "than the header."
                    )
                if any(value is None for value in row.values()):
                    raise ValueError(
                        f"Malformed TSV row {row_number} in {path}: fewer fields "
                        "than the header."
                    )
                rows.append(
                    {
                        str(key).strip(): (value or "").strip()
                        for key, value in row.items()
                        if key is not None
                    }
                )
    except UnicodeDecodeError as error:
        raise ValueError(f"TSV file is not valid UTF-8: {path}") from error
    return columns, rows


def read_tsv_header(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required TSV file not found: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if not reader.fieldnames:
                raise ValueError(f"TSV file has no header row: {path}")
            columns = [str(column).strip() for column in reader.fieldnames]
            if len(set(columns)) != len(columns):
                raise ValueError(f"TSV file has duplicate column names: {path}")
            return columns
    except UnicodeDecodeError as error:
        raise ValueError(f"TSV file is not valid UTF-8: {path}") from error


def iter_tsv(path: Path) -> Generator[dict[str, str], None, None]:
    """Yield rows from a TSV file one at a time (streaming)."""
    if not path.is_file():
        raise FileNotFoundError(f"Required TSV file not found: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if not reader.fieldnames:
                raise ValueError(f"TSV file has no header row: {path}")
            columns = [str(column).strip() for column in reader.fieldnames]
            if len(set(columns)) != len(columns):
                raise ValueError(f"TSV file has duplicate column names: {path}")
            for row_number, row in enumerate(reader, start=2):
                if None in row:
                    raise ValueError(
                        f"Malformed TSV row {row_number} in {path}: more fields "
                        "than the header."
                    )
                if any(value is None for value in row.values()):
                    raise ValueError(
                        f"Malformed TSV row {row_number} in {path}: fewer fields "
                        "than the header."
                    )
                yield {
                    str(key).strip(): (value or "").strip()
                    for key, value in row.items()
                    if key is not None
                }
    except UnicodeDecodeError as error:
        raise ValueError(f"TSV file is not valid UTF-8: {path}") from error


def load_source(path: Path, source_name: str) -> list[dict]:
    columns = read_tsv_header(path)
    id_column = resolve_column(columns, "id")
    name_column = resolve_column(columns, "business_name")
    address_column = resolve_column(columns, "business_address")
    if id_column is None or name_column is None:
        required = []
        if id_column is None:
            required.append("an ID column")
        if name_column is None:
            required.append("a business-name column")
        raise ValueError(
            f"{path} is missing {' and '.join(required)}. "
            f"Available columns: {', '.join(columns)}"
        )
    optional = {
        key: resolve_column(columns, key)
        for key in ("country", "city", "region", "postal_code")
    }
    seen: set[str] = set()
    records: list[dict] = []
    for row_number, row in enumerate(iter_tsv(path), start=2):
        record_id = row.get(id_column, "").strip()
        if not record_id:
            raise ValueError(f"Empty ID at row {row_number} in {path}.")
        if record_id in seen:
            raise ValueError(f"Duplicate ID {record_id!r} in {path}.")
        seen.add(record_id)
        record = {
            "id": record_id,
            "source": source_name,
            "business_name": row.get(name_column, ""),
            "business_address": row.get(address_column, "") if address_column else "",
            "country": row.get(optional["country"], "") if optional["country"] else "",
            "city": row.get(optional["city"], "") if optional["city"] else "",
            "region": row.get(optional["region"], "") if optional["region"] else "",
            "postal_code": row.get(optional["postal_code"], "")
            if optional["postal_code"]
            else "",
            "raw_fields": row,
        }
        records.append(record)
    return records


def write_tsv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=columns,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    temp_path.replace(path)


def load_ground_truth_rows(path: Path) -> list[dict[str, str]]:
    _, rows = read_tsv(path)
    return rows