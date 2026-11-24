"""Builders for small, deliberately messy spreadsheets used by the tests."""

from pathlib import Path

from openpyxl import Workbook


def messy_workbook(path: Path) -> Path:
    wb = Workbook()
    ws = wb.active
    ws.title = "Data"
    ws["A1"] = "Quarterly export - do not edit"  # title row above the header
    ws.append([])
    ws.append(["id", "Amount", "Region", None, "Region", "When", "Serial date", "Empty"])  # merged cell gap + duplicate name
    ws.merge_cells("C3:D3")
    rows = [
        [1, "$1,200.50", "North", "N-1", "east", "05/06/2024", 45292, None],
        [2, "950", "South", "S-2", "west", "13/06/2024", 45293, None],
        [3, "#N/A", "north ", "N-3", "east", "14/06/2024", 45294, None],
        [4, "1,050", "NORTH", "N-4", "east", "15/06/2024", 45295, None],
        [5, "n/a", "South", "S-5", "west", "16/06/2024", 45296, None],
    ]
    for r in rows:
        ws.append(r)
    ws.append([])  # trailing empty rows
    ws.append([])
    other = wb.create_sheet("Lookup")
    other.append(["code", "label"])
    other.append(["N", "North"])
    wb.save(path)
    return path


def simple_workbook(path: Path, n: int = 40) -> Path:
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["customer", "age", "plan", "spend", "churned"])
    for i in range(n):
        ws.append([f"C{i:03d}", 20 + i % 40, ["Basic", "Pro"][i % 2], 100 + 7 * i, "Yes" if i % 5 == 0 else "No"])
    wb.save(path)
    return path
