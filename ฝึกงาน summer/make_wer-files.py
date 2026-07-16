import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"D:\SIIT_Y1\ฝึกงาน summer")
OUT_DIR = Path(r"D:\SIIT_Y1\exp007_upload_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_FILE = BASE_DIR / "clean_no_conflict_v3.csv"
PREDICTION_FILE = BASE_DIR / "groq_result_100rowsv3.csv"
EXPECTED_ROWS = 70

OUTPUT_REF = OUT_DIR / "answer_exp007_v3_100rows.csv"
OUTPUT_PRED = OUT_DIR / "test_exp007_v3_100rows.csv"
OUTPUT_CHECK = OUT_DIR / "check_exp007_v3_100rows.csv"

SOURCE_COL = "source_row"
REF_COL = "gloss_reference"
PRED_COL = "hermes_output"   # change this if your Groq file uses another column name


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def main():
    ref_df = pd.read_csv(REFERENCE_FILE, encoding="utf-8-sig")
    pred_df = pd.read_csv(PREDICTION_FILE, encoding="utf-8-sig")

    ref_df.columns = [str(c).strip().replace("\ufeff", "") for c in ref_df.columns]
    pred_df.columns = [str(c).strip().replace("\ufeff", "") for c in pred_df.columns]

    print("Reference columns:", ref_df.columns.tolist())
    print("Prediction columns:", pred_df.columns.tolist())
    print("Prediction rows:", len(pred_df))

    if len(pred_df) != EXPECTED_ROWS:
        raise ValueError(f"Prediction file must have {EXPECTED_ROWS} rows, but found {len(pred_df)}")

    for col in [SOURCE_COL, REF_COL]:
        if col not in ref_df.columns:
            raise ValueError(f"Missing column in reference file: {col}")

    for col in [SOURCE_COL, PRED_COL]:
        if col not in pred_df.columns:
            raise ValueError(f"Missing column in prediction file: {col}")

    ref_df[SOURCE_COL] = ref_df[SOURCE_COL].astype(str).str.strip()
    pred_df[SOURCE_COL] = pred_df[SOURCE_COL].astype(str).str.strip()

    merged = pred_df[[SOURCE_COL, PRED_COL]].merge(
        ref_df[[SOURCE_COL, REF_COL]],
        on=SOURCE_COL,
        how="left",
        validate="many_to_one"
    )

    if merged[REF_COL].isna().any():
        missing = merged[merged[REF_COL].isna()]
        print(missing[[SOURCE_COL, PRED_COL]])
        raise ValueError("Some source_row values do not match reference file.")

    if merged[PRED_COL].isna().any() or merged[PRED_COL].astype(str).str.strip().eq("").any():
        blank = merged[merged[PRED_COL].isna() | merged[PRED_COL].astype(str).str.strip().eq("")]
        print(blank[[SOURCE_COL, PRED_COL]])
        raise ValueError("Some prediction rows are blank.")

    merged["รหัสข้อความ"] = range(len(merged))
    merged["answer_text"] = merged[REF_COL].apply(clean_text)
    merged["test_text"] = merged[PRED_COL].apply(clean_text)

    answer_upload = merged[["รหัสข้อความ", "answer_text"]].rename(
        columns={"answer_text": "ข้อความ"}
    )

    test_upload = merged[["รหัสข้อความ", "test_text"]].rename(
        columns={"test_text": "ข้อความ"}
    )

    check_df = merged[["รหัสข้อความ", SOURCE_COL, "answer_text", "test_text"]].copy()

    answer_upload.to_csv(OUTPUT_REF, index=False, encoding="utf-8-sig")
    test_upload.to_csv(OUTPUT_PRED, index=False, encoding="utf-8-sig")
    check_df.to_csv(OUTPUT_CHECK, index=False, encoding="utf-8-sig")

    print("\nDONE")
    print("Upload these two files to RTT:")
    print("Answer:", OUTPUT_REF)
    print("Test:  ", OUTPUT_PRED)

    print("\nDo NOT upload this:")
    print("Check:", OUTPUT_CHECK)

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