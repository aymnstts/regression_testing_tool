import os
import pandas as pd
import json

# === CONFIGURATION ===
FOLDER_PATH = "reference_files"  # ← Replace this with the path to your .xlsx files
OUTPUT_JSON = True                   # Set to True if you want to export to JSON
OUTPUT_FILE = "schemas.json"         # Output file name

# === TYPE GUESSER ===
def guess_type(series):
    if pd.api.types.is_integer_dtype(series):
        return "int"
    elif pd.api.types.is_float_dtype(series):
        return "float"
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    else:
        return "string"

# === MAIN SCRIPT ===
schemas = {}
for file in os.listdir(FOLDER_PATH):
    if file.endswith(".xlsx"):
        file_path = os.path.join(FOLDER_PATH, file)
        try:
            df = pd.read_excel(file_path)
            columns = df.columns.tolist()
            types = {col: guess_type(df[col]) for col in columns}
            schemas[file] = {
                "columns": columns,
                "types": types
            }
        except Exception as e:
            schemas[file] = {"error": str(e)}

# === OUTPUT RESULTS ===
# Print to console
for fname, info in schemas.items():
    print(f"\nFile: {fname}")
    print(json.dumps(info, indent=2))

# Optionally save to JSON file
if OUTPUT_JSON:
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(schemas, f, indent=2, ensure_ascii=False)

print("\n✅ Extraction complete.")
