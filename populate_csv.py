import pandas as pd
import re

# Read the CSV with proper handling for multiline fields
try:
    df = pd.read_csv('C:/pi-engine/english_poetry.csv')
    print(f"Loaded {len(df)} rows")
    print("Columns:", list(df.columns))
    print("\nFirst few rows:")
    print(df.head(3))
except Exception as e:
    print(f"Error reading CSV: {e}")