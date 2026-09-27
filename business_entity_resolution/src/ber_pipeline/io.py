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


def load_source(
    path: Path,
    source_name: str,
    max_records: int | None = None,
    needed_ids: set[str] | None = None,
) -> list[dict]:
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
    id_idx = columns.index(id_column)
    name_idx = columns.index(name_column)
    addr_idx = columns.index(address_column) if address_column else None
    country_idx = columns.index(optional["country"]) if optional["country"] else None
    city_idx = columns.index(optional["city"]) if optional["city"] else None
    region_idx = columns.index(optional["region"]) if optional["region"] else None
    postal_idx = columns.index(optional["postal_code"]) if optional["postal_code"] else None
    num_cols = len(columns)

    seen: set[str] = set()
    records: list[dict] = []
    needed_count = len(needed_ids) if needed_ids else 0
    found_needed = 0

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        next(stream, None)  # skip header
        for row_number, line in enumerate(stream, start=2):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < num_cols:
                parts += [""] * (num_cols - len(parts))
            record_id = parts[id_idx].strip()
            if not record_id or record_id in seen:
                continue

            is_needed = needed_ids is not None and record_id in needed_ids
            if is_needed:
                found_needed += 1
            elif max_records is not None and (len(records) - found_needed) >= max_records:
                if needed_ids is None or found_needed >= needed_count:
                    break
                continue

            seen.add(record_id)
            records.append({
                "id": record_id,
                "source": source_name,
                "business_name": parts[name_idx].strip(),
                "business_address": parts[addr_idx].strip() if addr_idx is not None else "",
                "country": parts[country_idx].strip() if country_idx is not None else "",
                "city": parts[city_idx].strip() if city_idx is not None else "",
                "region": parts[region_idx].strip() if region_idx is not None else "",
                "postal_code": parts[postal_idx].strip() if postal_idx is not None else "",
                "raw_fields": {},
            })
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


def load_ground_truth_rows(
    path: Path, source1_ids: set[str] | None = None
) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required TSV file not found: {path}")
    columns = read_tsv_header(path)
    if len(columns) < 2:
        _, rows = read_tsv(path)
        return rows
    s1_col, match_col = columns[0], columns[1]
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        next(stream, None)  # skip header
        for line in stream:
            parts = line.rstrip("\r\n").split("\t")
            if not parts:
                continue
            s1_id = parts[0].strip()
            if source1_ids is not None and s1_id not in source1_ids:
                continue
            match_val = parts[1].strip() if len(parts) > 1 else ""
            rows.append({s1_col: s1_id, match_col: match_val})
    return rows