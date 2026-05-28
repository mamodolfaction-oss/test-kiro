import sys
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill

# Read data from stdin or hardcoded
# X1: 1=Teknik, 2=Bisnis, 3=Kesehatan
# X2: 1=2023, 2=2024, 3=2025

dept_map = {1: "Teknik", 2: "Bisnis", 3: "Kesehatan"}
ang_map = {1: 2023, 2: 2024, 3: 2025}

# Target responden per cell (proporsional n=128)
target = {
    (2023, "Teknik"): 7, (2023, "Bisnis"): 17, (2023, "Kesehatan"): 17,
    (2024, "Teknik"): 8, (2024, "Bisnis"): 19, (2024, "Kesehatan"): 20,
    (2025, "Teknik"): 7, (2025, "Bisnis"): 17, (2025, "Kesehatan"): 16,
}

# Read data file
with open("/projects/sandbox/test-kiro/data_clean.tsv", "r") as f:
    lines = f.readlines()

header = lines[0].strip().split('\t')
rows = []
for line in lines[1:]:
    if line.strip():
        rows.append(line.strip().split('\t'))

print(f"Total rows: {len(rows)}")
