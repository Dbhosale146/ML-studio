from pathlib import Path

import pandas as pd
import pytest

from datalyon.ml.ingest import IngestError, dataset_hash, read_file, reparse_sheet, sniff_file_type
from tests.fixtures.builders import messy_workbook, simple_workbook


def test_sniff_by_content_not_extension(tmp_path: Path) -> None:
    p = simple_workbook(tmp_path / "a.xlsx")
    assert sniff_file_type("a.xlsx", p.read_bytes()[:16]) == "xlsx"
    with pytest.raises(IngestError):
        sniff_file_type("fake.xlsx", b"col1,col2\n1,2\n")
    with pytest.raises(IngestError):
        sniff_file_type("report.pdf", b"%PDF-1.7")
    assert sniff_file_type("data.csv", b"a,b\n1,2\n") == "csv"


def test_messy_workbook_is_cleaned(tmp_path: Path) -> None:
    sheets = read_file(messy_workbook(tmp_path / "m.xlsx"), "xlsx", 1000, 50)
    data = next(s for s in sheets if s.name == "Data")
    df = data.df
    assert data.header_row == 2  # title row and blank row skipped
    assert list(df.columns) == ["id", "Amount", "Region", "Region_2", "Region_3", "When", "Serial date"]
    assert len(df) == 5  # trailing empty rows removed, empty column dropped
    assert df["Amount"].tolist()[:2] == [1200.5, 950.0]  # $ and thousands separators
    assert df["Amount"].isna().sum() == 2  # #N/A and n/a are missing values
    assert pd.api.types.is_datetime64_any_dtype(df["When"])
    assert df["When"].iloc[0] == pd.Timestamp("2024-06-05")  # day-first detected from 13/06/2024
    assert pd.api.types.is_datetime64_any_dtype(df["Serial date"])
    assert df["Serial date"].iloc[0] == pd.Timestamp("2024-01-01")
    assert any("merged" in n for n in data.notes)
    assert {s.name for s in sheets} == {"Data", "Lookup"}


def test_explicit_header_row(tmp_path: Path) -> None:
    p = messy_workbook(tmp_path / "m.xlsx")
    sf = reparse_sheet(p, "xlsx", "Data", 3, 1000, 50)  # use the first data row as the header
    assert sf.header_row == 3
    assert "1" in sf.df.columns


def test_csv_with_semicolons(tmp_path: Path) -> None:
    p = tmp_path / "d.csv"
    p.write_text("name;amount;when\nA;1 200;2024-01-03\nB;300;2024-02-03\nC;NA;2024-03-03\n", encoding="utf-8")
    df = read_file(p, "csv", 100, 10)[0].df
    assert df["amount"].tolist()[:2] == [1200.0, 300.0]
    assert pd.api.types.is_datetime64_any_dtype(df["when"])


def test_limits_raise_clear_errors(tmp_path: Path) -> None:
    p = simple_workbook(tmp_path / "s.xlsx", n=200)
    with pytest.raises(IngestError) as e:
        read_file(p, "xlsx", 100, 50)
    assert e.value.code == "too_many_rows"
    with pytest.raises(IngestError) as e2:
        read_file(p, "xlsx", 1000, 3)
    assert e2.value.code == "too_many_columns"


def test_dataset_hash_is_stable_and_sensitive(tmp_path: Path) -> None:
    df = read_file(simple_workbook(tmp_path / "s.xlsx"), "xlsx", 1000, 50)[0].df
    assert dataset_hash(df) == dataset_hash(df.copy())
    changed = df.copy()
    changed.loc[0, "spend"] = -1
    assert dataset_hash(changed) != dataset_hash(df)
