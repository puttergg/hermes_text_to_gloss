import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher


BASE_DIR = Path(r"D:\SIIT_Y1\ฝึกงาน summer")

CLEANED_FILE = BASE_DIR / "cleaned_weather_tsl.csv"
CONFLICT_FILE = BASE_DIR / "manual_review_conflicts.csv"

# This file must be created AFTER you run Hermes
HERMES_FILE = BASE_DIR / "hermes_result.csv"

OUTPUT_CLEAN_NO_CONFLICT = BASE_DIR / "D:\SIIT_Y1\ฝึกงาน summer\clean_no_conflict.csv"
OUTPUT_ERROR_DETAIL = BASE_DIR / "error_analysis_detail.csv"
OUTPUT_ERROR_SUMMARY = BASE_DIR / "error_summary.csv"


THAI_COL = "thai_sentence"
REF_COL = "gloss_reference"

HERMES_OUTPUT_COL = "hermes_output"


def read_file(path):
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def split_gloss(text):
    if pd.isna(text):
        return []
    return [x.strip() for x in str(text).split("|") if x.strip()]


def compare_gloss(reference, prediction):
    ref = split_gloss(reference)
    pred = split_gloss(prediction)

    matcher = SequenceMatcher(None, ref, pred)

    inserted = []
    deleted = []
    replaced = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        ref_part = ref[i1:i2]
        pred_part = pred[j1:j2]

        if tag == "equal":
            continue
        elif tag == "insert":
            inserted.extend(pred_part)
        elif tag == "delete":
            deleted.extend(ref_part)
        elif tag == "replace":
            replaced.append(f"{'|'.join(ref_part)} -> {'|'.join(pred_part)}")

    error_types = []

    if inserted:
        error_types.append("Insert")
    if deleted:
        error_types.append("Delete")
    if replaced:
        error_types.append("Replace")

    if not error_types:
        error_type = "Exact"
    else:
        error_type = "+".join(error_types)

    return {
        "error_type": error_type,
        "inserted_gloss": "|".join(inserted),
        "deleted_gloss": "|".join(deleted),
        "replaced_gloss": " ; ".join(replaced),
        "reference_len": len(ref),
        "prediction_len": len(pred),
    }


def main():
    cleaned_df = read_file(CLEANED_FILE)
    conflict_df = read_file(CONFLICT_FILE)
    hermes_df = read_file(HERMES_FILE)

    print("Cleaned columns:", cleaned_df.columns.tolist())
    print("Conflict columns:", conflict_df.columns.tolist())
    print("Hermes columns:", hermes_df.columns.tolist())

    if HERMES_OUTPUT_COL not in hermes_df.columns:
        raise ValueError(
            f"\nCannot find Hermes output column: {HERMES_OUTPUT_COL}\n"
            f"Available columns are:\n{hermes_df.columns.tolist()}\n\n"
            f"Fix this line:\nHERMES_OUTPUT_COL = 'your_column_name'"
        )

    # remove 10 conflict groups
    conflict_sentences = set(conflict_df[THAI_COL].dropna().astype(str))

    clean_no_conflict = cleaned_df[
        ~cleaned_df[THAI_COL].astype(str).isin(conflict_sentences)
    ].copy()

    clean_no_conflict.to_csv(
        OUTPUT_CLEAN_NO_CONFLICT,
        index=False,
        encoding="utf-8-sig"
    )

    print("Saved clean_no_conflict.csv")
    print("Rows before:", len(cleaned_df))
    print("Rows after removing conflicts:", len(clean_no_conflict))

    hermes_small = hermes_df[["source_row", HERMES_OUTPUT_COL]].copy()
    hermes_small["source_row"] = hermes_small["source_row"].astype(str)

    clean_no_conflict["source_row"] = clean_no_conflict["source_row"].astype(str)

    merged = clean_no_conflict.merge(
        hermes_small,
        on="source_row",
        how="inner"
    )

    merged = merged.dropna(subset=[HERMES_OUTPUT_COL]).copy()
    merged = merged[merged[HERMES_OUTPUT_COL].astype(str).str.strip() != ""].copy()

    print(f"Rows with Hermes output for analysis: {len(merged)}")

    rows = []

    for _, row in merged.iterrows():
        reference = row.get(REF_COL, "")
        prediction = row.get(HERMES_OUTPUT_COL, "")

        result = compare_gloss(reference, prediction)

        rows.append({
            "source_row": row.get("source_row", ""),
            "thai_sentence": row.get(THAI_COL, ""),
            "gloss_reference": reference,
            "hermes_output": prediction,
            "is_exact_match": result["error_type"] == "Exact",
            "error_type": result["error_type"],
            "inserted_gloss": result["inserted_gloss"],
            "deleted_gloss": result["deleted_gloss"],
            "replaced_gloss": result["replaced_gloss"],
            "reference_len": result["reference_len"],
            "prediction_len": result["prediction_len"],
        })

    detail_df = pd.DataFrame(rows)

    detail_df.to_csv(
        OUTPUT_ERROR_DETAIL,
        index=False,
        encoding="utf-8-sig"
    )

    summary_df = (
        detail_df.groupby("error_type")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    summary_df["percent"] = (
        summary_df["count"] / len(detail_df) * 100
    ).round(2)

    summary_df.to_csv(
        OUTPUT_ERROR_SUMMARY,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n========== DONE ==========")
    print(f"Saved: {OUTPUT_CLEAN_NO_CONFLICT}")
    print(f"Saved: {OUTPUT_ERROR_DETAIL}")
    print(f"Saved: {OUTPUT_ERROR_SUMMARY}")
    print("\nError summary:")
    print(summary_df)
    print("==========================")


if __name__ == "__main__":
    main()