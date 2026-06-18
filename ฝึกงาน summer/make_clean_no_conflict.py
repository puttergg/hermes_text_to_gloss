import pandas as pd
from pathlib import Path


BASE_DIR = Path(r"D:\SIIT_Y1\ฝึกงาน summer")

CLEANED_FILE = BASE_DIR / "D:\SIIT_Y1\ฝึกงาน summer\cleaned_weather_tsl_quality_report_v2.csv"
CONFLICT_FILE = BASE_DIR / "D:\SIIT_Y1\ฝึกงาน summer\manual_review_conflicts_v2.csv"

OUTPUT_FILE = BASE_DIR / "clean_no_conflictv2.csv"


THAI_COL = "thai_sentence"


def main():
    cleaned_df = pd.read_csv(CLEANED_FILE, encoding="utf-8-sig")
    conflict_df = pd.read_csv(CONFLICT_FILE, encoding="utf-8-sig")

    conflict_sentences = set(conflict_df[THAI_COL].dropna().astype(str))

    clean_no_conflict = cleaned_df[
        ~cleaned_df[THAI_COL].astype(str).isin(conflict_sentences)
    ].copy()

    clean_no_conflict.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("Saved:", OUTPUT_FILE)
    print("Original cleaned rows:", len(cleaned_df))
    print("Conflict groups:", len(conflict_sentences))
    print("Rows after removing conflicts:", len(clean_no_conflict))
    print("Unique Thai sentences after removing conflicts:", clean_no_conflict[THAI_COL].nunique())


if __name__ == "__main__":
    main()