"""Reading uploaded spreadsheets into clean pandas DataFrames.

Handles the usual real-world Excel problems: title rows above the header, merged header cells,
empty trailing rows/columns, Excel error values, numbers stored as text (thousands separators,
currency signs, percents), mixed date formats, Excel serial dates, duplicate column names and
fully empty columns. Pure library code: no web framework imports.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

FileType = Literal["xlsx", "xls", "csv"]

NULL_TOKENS = {
    "", "na", "n/a", "nan", "null", "none", "-", "--", "?", "#n/a", "#value!", "#div/0!", "#ref!", "#null!", "#name?", "#num!",
    "missing", "nil",
}  # fmt: skip
DATE_PATTERN = re.compile(
    r"^\s*(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{4}[-/]\d{1,2}$|"
    r"\d{1,2}[ -](jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*[ -]\d{2,4}|"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*[ -]\d{1,2},?[ -]\d{2,4}|"
    r"\d{4}-\d{2}-\d{2}[t ]\d{1,2}:\d{2})",
    re.I,
)
NUMERIC_JUNK = re.compile(r"[,\s$€£₹¥%]")


class IngestError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class SheetFrame:
    name: str
    df: pd.DataFrame
    header_row: int | None  # 0-based row of the header in the raw sheet, None when the sheet has no header
    notes: list[str] = field(default_factory=list)


def sniff_file_type(filename: str, head: bytes) -> FileType:
    """Decide the file type from its content (magic bytes), cross-checked with the extension."""
    ext = Path(filename or "").suffix.lower()
    if head.startswith(b"PK\x03\x04"):
        if ext not in (".xlsx", ".xlsm", ""):
            raise IngestError("unsupported_type", f"The file content is an Excel workbook but the name ends in '{ext}'.")
        return "xlsx"
    if head.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        if ext not in (".xls", ""):
            raise IngestError("unsupported_type", f"The file content is an Excel 97-2003 workbook but the name ends in '{ext}'.")
        return "xls"
    if ext in (".xlsx", ".xlsm", ".xls"):
        raise IngestError("invalid_excel", "This file has an Excel extension but its content is not an Excel workbook.")
    if ext in (".csv", ".txt") and b"\x00" not in head[:4096]:
        return "csv"
    raise IngestError("unsupported_type", "Only Excel (.xlsx, .xls) and CSV files are supported.")


def read_file(path: Path, file_type: FileType, max_rows: int, max_cols: int) -> list[SheetFrame]:
    """Parse every sheet that contains a table. Raises IngestError with a user-friendly message."""
    raws: list[tuple[str, pd.DataFrame]] = []
    if file_type == "csv":
        raws.append((Path(path).stem, _read_csv_raw(path)))
    else:
        try:
            xl = pd.ExcelFile(path, engine="openpyxl" if file_type == "xlsx" else "xlrd")
        except Exception as e:
            raise IngestError(
                "invalid_excel", f"The workbook could not be opened ({e.__class__.__name__}). Is it password-protected or corrupted?"
            ) from e
        for name in xl.sheet_names:
            try:
                raws.append((str(name), pd.read_excel(xl, sheet_name=name, header=None)))
            except Exception:
                continue
    sheets = []
    for name, raw in raws:
        _check_limits(name, raw, max_rows, max_cols)
        df, header_row, notes = parse_sheet(raw)
        if df is not None and len(df) >= 1:
            sheets.append(SheetFrame(name=name, df=df, header_row=header_row, notes=notes))
    if not sheets:
        raise IngestError("empty_file", "No sheet in this file contains a table with data.")
    return sheets


def reparse_sheet(path: Path, file_type: FileType, sheet: str, header_row: int | None, max_rows: int, max_cols: int) -> SheetFrame:
    """Parse one sheet again with an explicit header row chosen by the user."""
    raw = _read_csv_raw(path) if file_type == "csv" else pd.read_excel(path, sheet_name=sheet, header=None)
    _check_limits(sheet, raw, max_rows, max_cols)
    df, hr, notes = parse_sheet(raw, header_row=header_row)
    if df is None:
        raise IngestError("empty_sheet", f"Sheet '{sheet}' has no data below row {(header_row or 0) + 1}.")
    return SheetFrame(name=sheet, df=df, header_row=hr, notes=notes)


def _read_csv_raw(path: Path) -> pd.DataFrame:
    data = Path(path).read_bytes()
    text = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise IngestError("invalid_csv", "The CSV file has an unknown text encoding.")
    try:
        dialect = csv.Sniffer().sniff(text[:65536], delimiters=",;\t|")
        sep = dialect.delimiter
    except csv.Error:
        sep = ","
    try:
        return pd.read_csv(io.StringIO(text), sep=sep, header=None, dtype=str, keep_default_na=False, skip_blank_lines=True)
    except Exception as e:
        raise IngestError("invalid_csv", f"The CSV file could not be parsed: {e}") from e


def _check_limits(name: str, raw: pd.DataFrame, max_rows: int, max_cols: int) -> None:
    rows, cols = raw.dropna(how="all").shape[0], raw.dropna(axis=1, how="all").shape[1]
    if rows > max_rows + 50:
        raise IngestError(
            "too_many_rows", f"Sheet '{name}' has {rows:,} rows; the limit is {max_rows:,}. Split the file or raise MAX_ROWS."
        )
    if cols > max_cols:
        raise IngestError(
            "too_many_columns",
            f"Sheet '{name}' has {cols:,} columns; the limit is {max_cols:,}. Remove unused columns or raise MAX_COLUMNS.",
        )


def _is_blank(v: Any) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and v.strip() == "")


def _looks_numeric(v: Any) -> bool:
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        return True
    try:
        float(NUMERIC_JUNK.sub("", str(v)))
        return True
    except ValueError:
        return False


def detect_header(raw: pd.DataFrame) -> int | None:
    """0-based index of the header row, or None when the first wide row is already data."""
    head = raw.head(30)
    counts = head.map(lambda v: not _is_blank(v)).sum(axis=1)
    if counts.empty:
        return None
    widest = int(counts.max())
    for i in range(len(head)):
        vals = [v for v in head.iloc[i].tolist() if not _is_blank(v)]
        if len(vals) < max(1, 0.5 * widest):
            continue
        texty = sum(isinstance(v, str) and not _looks_numeric(v) and not DATE_PATTERN.match(v) for v in vals)
        if texty / len(vals) >= 0.6 and i + 1 < len(raw):
            return i
        return None
    return None


def _make_names(values: list[Any], has_data: list[bool]) -> tuple[list[str], int]:
    """Clean header names: merged cells take the name of the cell on their left, blanks become column_N, duplicates get _2."""
    names: list[str] = []
    seen: set[str] = set()
    merged = 0
    prev = ""
    for j, v in enumerate(values):
        if isinstance(v, float) and not np.isnan(v) and v.is_integer():
            n = str(int(v))
        else:
            n = "" if _is_blank(v) else re.sub(r"\s+", " ", str(v)).strip()
        if not n and prev and has_data[j]:
            n = prev  # part of a merged header cell spanning several columns
            merged += 1
        elif n:
            prev = n
        if not n or n.lower().startswith("unnamed:"):
            n = f"column_{j + 1}"
        base, k = n, 2
        while n in seen:
            n = f"{base}_{k}"
            k += 1
        seen.add(n)
        names.append(n)
    return names, merged


def parse_sheet(raw: pd.DataFrame, header_row: int | None = -1) -> tuple[pd.DataFrame | None, int | None, list[str]]:
    """header_row=-1 means detect automatically; None means the sheet has no header."""
    notes: list[str] = []
    raw = raw.copy()
    raw = raw.map(lambda v: np.nan if isinstance(v, str) and v.strip() == "" else v)
    raw = raw.dropna(how="all").dropna(axis=1, how="all")
    if raw.empty:
        return None, None, notes
    raw = raw.reset_index(drop=False).rename(columns={"index": "__row__"})
    raw_rows = raw.pop("__row__")
    h = detect_header(raw) if header_row == -1 else (None if header_row is None else _position_of(raw_rows, header_row))
    if h is None:
        names = [f"column_{j + 1}" for j in range(raw.shape[1])]
        body = raw
        abs_header: int | None = None
        notes.append("No header row found, so the columns were named column_1, column_2, …")
    else:
        body = raw.iloc[h + 1 :]
        has_data = [bool(body[c].notna().any()) for c in raw.columns]
        names, merged = _make_names(list(raw.iloc[h]), has_data)
        abs_header = int(raw_rows.iloc[h])
        if abs_header > 0:
            notes.append(
                f"The header is on row {abs_header + 1}; the {abs_header} row{'s' if abs_header > 1 else ''} above it were skipped"
            )
        if merged:
            notes.append(f"Named {merged} column{'s' if merged > 1 else ''} under merged header cells")
        dups = len(names) - len({re.sub(r"_\d+$", "", n) for n in names})
        if dups > 0:
            notes.append(f"Renamed {dups} duplicate column name{'s' if dups > 1 else ''}")
    df = body.copy()
    df.columns = names
    df = df.dropna(how="all").reset_index(drop=True)
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        notes.append(f"Dropped {len(empty_cols)} empty column{'s' if len(empty_cols) > 1 else ''}")
    if df.empty or df.shape[1] == 0:
        return None, abs_header, notes
    df, type_notes = infer_types(df)
    notes.extend(type_notes)
    return df, abs_header, notes


def _position_of(raw_rows: pd.Series, header_row: int) -> int:
    matches = np.flatnonzero(raw_rows.to_numpy() == header_row)
    if len(matches) == 0:
        raise IngestError("bad_header_row", f"Row {header_row + 1} is empty, so it cannot be the header.")
    return int(matches[0])


def _clean_object(s: pd.Series) -> pd.Series:
    def f(v: Any) -> Any:
        if isinstance(v, str):
            t = v.strip()
            return np.nan if t.lower() in NULL_TOKENS else t
        return v

    return s.map(f)


def infer_types(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    converted_num: list[str] = []
    converted_date: list[str] = []
    errors_found = 0
    for c in list(df.columns):
        s = df[c]
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            before = s.notna().sum()
            s = _clean_object(s.astype(object))
            errors_found += int(before - s.notna().sum())
            nonnull = s.dropna()
            if nonnull.empty:
                df[c] = s
                continue
            is_dt = nonnull.map(
                lambda v: isinstance(v, (pd.Timestamp,)) or (hasattr(v, "year") and hasattr(v, "month") and not isinstance(v, str))
            )
            if is_dt.mean() >= 0.9:
                df[c] = pd.to_datetime(s, errors="coerce")
                converted_date.append(c)
                continue
            as_str = nonnull.astype(str)
            has_leading_zero = as_str.str.match(r"^0\d+$").mean() > 0.05  # zip codes, ids: keep as text
            if not has_leading_zero:
                num = pd.to_numeric(as_str.str.replace(NUMERIC_JUNK, "", regex=True), errors="coerce")
                if num.notna().mean() >= 0.95:
                    full = pd.to_numeric(s.map(lambda v: v if _is_blank(v) else NUMERIC_JUNK.sub("", str(v))), errors="coerce")
                    df[c] = full.astype(float)
                    if (nonnull.map(lambda v: isinstance(v, str))).any():
                        converted_num.append(c)
                    continue
            sample = as_str.head(300)
            if sample.map(lambda x: bool(DATE_PATTERN.match(x))).mean() >= 0.8:
                parsed = pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=_is_dayfirst(as_str))
                if parsed.notna().sum() >= 0.9 * len(nonnull):
                    df[c] = parsed
                    converted_date.append(c)
                    continue
            if nonnull.map(type).nunique() > 1:
                s = s.map(lambda v: v if _is_blank(v) else str(v))
            df[c] = s.astype(object)
    # Excel serial dates stored as plain numbers in a column whose name says it is a date
    for c in list(df.columns):
        s = df[c]
        if (
            pd.api.types.is_numeric_dtype(s)
            and not pd.api.types.is_bool_dtype(s)
            and re.search(r"date|day|time|month|period", str(c), re.I)
        ):
            nn = s.dropna()
            if len(nn) and nn.between(20000, 60000).mean() > 0.95 and not re.search(r"days|count|num", str(c), re.I):
                df[c] = pd.to_datetime(s, unit="D", origin="1899-12-30", errors="coerce")
                converted_date.append(c)
                if c in converted_num:
                    converted_num.remove(c)
    if errors_found:
        notes.append(f"Treated {errors_found:,} placeholder or Excel error cells (#N/A, #DIV/0!, n/a, …) as missing")
    if converted_num:
        notes.append(
            f"Converted {len(converted_num)} text column{'s' if len(converted_num) > 1 else ''} to numbers ({', '.join(converted_num[:4])}{'…' if len(converted_num) > 4 else ''})"
        )
    if converted_date:
        notes.append(
            f"Recognised {len(converted_date)} date column{'s' if len(converted_date) > 1 else ''} ({', '.join(converted_date[:4])})"
        )
    return df, notes


def _is_dayfirst(values: pd.Series) -> bool:
    """dd/mm/yyyy vs mm/dd/yyyy: decided by any value whose first part cannot be a month."""
    parts = values.str.extract(r"^\s*(\d{1,2})[/.-](\d{1,2})[/.-]\d{2,4}").dropna().astype(int)
    if parts.empty:
        return False
    return bool((parts[0] > 12).any())


def dataset_hash(df: pd.DataFrame) -> str:
    """Stable content hash of a parsed table (column names + values), saved with every model."""
    h = hashlib.sha256()
    h.update("|".join(map(str, df.columns)).encode())
    h.update(pd.util.hash_pandas_object(df.astype(str), index=False).to_numpy().tobytes())
    return h.hexdigest()
