import pandas as pd
from pathlib import Path

# =========================
# Experiment 006 RTT Working Upload CSV Maker
# Thai weather forecast -> Thai Sign Language gloss
# =========================

BASE_DIR = Path(r"D:\SIIT_Y1\ฝึกงาน summer")
OUT_DIR = Path(r"D:\SIIT_Y1\exp010_upload_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_FILE = BASE_DIR / "clean_no_conflict_v3.csv"
PREDICTION_FILE = BASE_DIR / "qwen_hermes_result_45rows_exp006_prompt3.csv"
EXPECTED_ROWS = 45

OUTPUT_REF = OUT_DIR / "answer_exp010_v3_100rows.csv"
OUTPUT_PRED = OUT_DIR / "test_exp010_v3_100rows.csv"
OUTPUT_CHECK = OUT_DIR / "check_exp010_v3_100rows.csv"

SOURCE_COL = "source_row"
REF_COL = "gloss_reference"
PRED_COL = "hermes_output"




def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def main():
    print("Reading files...")

    ref_df = pd.read_csv(REFERENCE_FILE, encoding="utf-8-sig")
    pred_df = pd.read_csv(PREDICTION_FILE, encoding="utf-8-sig")

    # Clean hidden header characters
    ref_df.columns = [str(c).strip().replace("\ufeff", "") for c in ref_df.columns]
    pred_df.columns = [str(c).strip().replace("\ufeff", "") for c in pred_df.columns]

    # Check required columns
    for col in [SOURCE_COL, REF_COL]:
        if col not in ref_df.columns:
            raise ValueError(f"Missing column in reference file: {col}")

    for col in [SOURCE_COL, PRED_COL]:
        if col not in pred_df.columns:
            raise ValueError(f"Missing column in prediction file: {col}")

    # Check prediction row count
    if len(pred_df) != EXPECTED_ROWS:
        raise ValueError(
            f"{PREDICTION_FILE.name} must have exactly {EXPECTED_ROWS} rows, "
            f"but found {len(pred_df)} rows."
        )

    # Keep only needed columns
    ref_small = ref_df[[SOURCE_COL, REF_COL]].copy()
    pred_small = pred_df[[SOURCE_COL, PRED_COL]].copy()

    # Clean source_row for matching
    ref_small[SOURCE_COL] = ref_small[SOURCE_COL].astype(str).str.strip()
    pred_small[SOURCE_COL] = pred_small[SOURCE_COL].astype(str).str.strip()

    # Match prediction with reference by source_row
    merged = pred_small.merge(
        ref_small,
        on=SOURCE_COL,
        how="left",
        validate="many_to_one"
    )

    # Check missing reference
    missing = merged[
        merged[REF_COL].isna()
        | merged[REF_COL].astype(str).str.strip().eq("")
    ]

    if not missing.empty:
        print("ERROR: Missing reference rows:")
        print(missing[[SOURCE_COL, PRED_COL]].to_string(index=False))
        raise ValueError("Cannot create upload files because some references are missing.")

    # Check blank prediction
    blank_pred = merged[
        merged[PRED_COL].isna()
        | merged[PRED_COL].astype(str).str.strip().eq("")
    ]

    if not blank_pred.empty:
        print("ERROR: Blank prediction rows:")
        print(blank_pred[[SOURCE_COL, PRED_COL]].to_string(index=False))
        raise ValueError("Cannot create upload files because some predictions are blank.")

    # Create RTT message ID 0-29
    merged["รหัสข้อความ"] = range(len(merged))

    # Clean text
    merged["answer_text"] = merged[REF_COL].apply(clean_text)
    merged["test_text"] = merged[PRED_COL].apply(clean_text)

    # RTT working format:
    # รหัสข้อความ,ข้อความ
    answer_upload = merged[["รหัสข้อความ", "answer_text"]].rename(
        columns={"answer_text": "ข้อความ"}
    )

    test_upload = merged[["รหัสข้อความ", "test_text"]].rename(
        columns={"test_text": "ข้อความ"}
    )

    # Check file only for inspection
    check_df = merged[
        [
            "รหัสข้อความ",
            SOURCE_COL,
            "answer_text",
            "test_text"
        ]
    ].copy()

    # Save
    answer_upload.to_csv(OUTPUT_REF, index=False, encoding="utf-8")
    test_upload.to_csv(OUTPUT_PRED, index=False, encoding="utf-8")
    check_df.to_csv(OUTPUT_CHECK, index=False, encoding="utf-8-sig")

    print("\nDONE")
    print("Upload these files to RTT:")
    print("Files Answer:", OUTPUT_REF)
    print("Files Test:  ", OUTPUT_PRED)

    print("\nDo NOT upload check file:")
    print(OUTPUT_CHECK)

    print("\nRows:")
    print("Answer rows:", len(answer_upload))
    print("Test rows:  ", len(test_upload))

    print("\nHeaders:")
    print("Answer:", list(answer_upload.columns))
    print("Test:  ", list(test_upload.columns))

    print("\nPreview:")
    print(check_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()