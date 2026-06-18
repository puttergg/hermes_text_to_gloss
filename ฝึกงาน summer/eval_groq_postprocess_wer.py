from pathlib import Path
from difflib import SequenceMatcher
import pandas as pd


BASE_DIR = Path(r"D:\SIIT_Y1\ฝึกงาน summer")

INPUT_FILE = BASE_DIR / "groq_result_30rows.csv"

OUTPUT_BY_ROW = BASE_DIR / "groq_postprocess_wer_by_row.csv"
OUTPUT_SUMMARY = BASE_DIR / "groq_postprocess_wer_summary.csv"
OUTPUT_DETAIL = BASE_DIR / "groq_postprocess_wer_detail.csv"

REF_COL = "gloss_reference"
PRED_COL = "hermes_output"


def split_tokens(text):
    if pd.isna(text):
        return []
    return [t.strip() for t in str(text).split("|") if t.strip()]


def postprocess_model_output(text: str) -> str:
    """
    Fix repeated model output style mismatch:
    1. Bangkok / surrounding area style
    2. Wind speed style: remove extra ลม before เร็ว
    3. Unit punctuation: กิโลเมตร|ชั่วโมง -> กิโลเมตร.|ชั่วโมง
    """
    if pd.isna(text):
        return ""

    text = str(text).strip()

    # 1. Bangkok normalization
    text = text.replace("กรุงเทพฯ|พื้นที่|ใกล้เคียง", "กรุงเทพ|จังหวัด|ใกล้เคียง")
    text = text.replace("กรุงเทพมหานคร|พื้นที่|ใกล้เคียง", "กรุงเทพ|จังหวัด|ใกล้เคียง")
    text = text.replace("กรุงเทพฯ|จังหวัด|ใกล้เคียง", "กรุงเทพ|จังหวัด|ใกล้เคียง")
    text = text.replace("กรุงเทพ|พื้นที่|ใกล้เคียง", "กรุงเทพ|จังหวัด|ใกล้เคียง")

    # 2. Wind speed normalization
    wind_directions = [
        "ลมตะวันออกเฉียงเหนือ",
        "ลมตะวันออกเฉียงใต้",
        "ลมตะวันออก",
        "ลมตะวันตก",
        "ลมตะวันตกเฉียงเหนือ",
        "ลมตะวันตกเฉียงใต้",
        "ลมเหนือ",
        "ลมใต้",
    ]

    for wind in wind_directions:
        text = text.replace(f"{wind}|ลม|เร็ว", f"{wind}|เร็ว")

    # 3. Unit punctuation normalization
    # Senior web reference often uses กิโลเมตร.|ชั่วโมง
    text = text.replace("กิโลเมตร|ชั่วโมง", "กิโลเมตร.|ชั่วโมง")

    # Clean empty tokens
    parts = [p.strip() for p in text.split("|") if p.strip()]
    return "|".join(parts)


def analyze_wer(reference, prediction):
    ref_tokens = split_tokens(reference)
    pred_tokens = split_tokens(prediction)

    matcher = SequenceMatcher(None, ref_tokens, pred_tokens)

    insert_count = 0
    delete_count = 0
    replace_count = 0
    correct_count = 0
    detail_items = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        ref_part = ref_tokens[i1:i2]
        pred_part = pred_tokens[j1:j2]

        if tag == "equal":
            correct_count += len(ref_part)

        elif tag == "insert":
            insert_count += len(pred_part)
            detail_items.append({
                "error_type": "Insert",
                "reference_tokens": "",
                "predicted_tokens": "|".join(pred_part),
            })

        elif tag == "delete":
            delete_count += len(ref_part)
            detail_items.append({
                "error_type": "Delete",
                "reference_tokens": "|".join(ref_part),
                "predicted_tokens": "",
            })

        elif tag == "replace":
            replace_count += max(len(ref_part), len(pred_part))
            detail_items.append({
                "error_type": "Replace",
                "reference_tokens": "|".join(ref_part),
                "predicted_tokens": "|".join(pred_part),
            })

    total_ref = len(ref_tokens)
    total_errors = insert_count + delete_count + replace_count
    wer = total_errors / total_ref if total_ref > 0 else 0
    accuracy = 1 - wer

    return {
        "insert": insert_count,
        "delete": delete_count,
        "replace": replace_count,
        "correct": correct_count,
        "total_ref": total_ref,
        "total_errors": total_errors,
        "wer": wer,
        "accuracy": accuracy,
        "detail_items": detail_items,
    }


def main():
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    print("Input file:", INPUT_FILE)
    print("Columns:", df.columns.tolist())
    print("Rows:", len(df))

    if REF_COL not in df.columns:
        raise ValueError(f"Missing column: {REF_COL}")

    if PRED_COL not in df.columns:
        raise ValueError(f"Missing column: {PRED_COL}")

    by_row = []
    detail_rows = []

    before_totals = {
        "insert": 0,
        "delete": 0,
        "replace": 0,
        "correct": 0,
        "total_ref": 0,
        "total_errors": 0,
    }

    after_totals = {
        "insert": 0,
        "delete": 0,
        "replace": 0,
        "correct": 0,
        "total_ref": 0,
        "total_errors": 0,
    }

    for idx, row in df.iterrows():
        source_row = row.get("source_row", idx + 1)
        thai_sentence = row.get("thai_sentence", "")
        reference = row.get(REF_COL, "")
        prediction_raw = row.get(PRED_COL, "")
        prediction_post = postprocess_model_output(prediction_raw)

        before = analyze_wer(reference, prediction_raw)
        after = analyze_wer(reference, prediction_post)

        for key in before_totals:
            before_totals[key] += before[key]
            after_totals[key] += after[key]

        by_row.append({
            "source_row": source_row,
            "thai_sentence": thai_sentence,
            "gloss_reference": reference,
            "groq_output_raw": prediction_raw,
            "groq_output_postprocessed": prediction_post,

            "before_insert": before["insert"],
            "before_delete": before["delete"],
            "before_replace": before["replace"],
            "before_total_errors": before["total_errors"],
            "before_total_ref": before["total_ref"],
            "before_wer": before["wer"],
            "before_accuracy": before["accuracy"],

            "after_insert": after["insert"],
            "after_delete": after["delete"],
            "after_replace": after["replace"],
            "after_total_errors": after["total_errors"],
            "after_total_ref": after["total_ref"],
            "after_wer": after["wer"],
            "after_accuracy": after["accuracy"],

            "accuracy_change": after["accuracy"] - before["accuracy"],
        })

        for item in after["detail_items"]:
            detail_rows.append({
                "source_row": source_row,
                "thai_sentence": thai_sentence,
                "gloss_reference": reference,
                "groq_output_postprocessed": prediction_post,
                "error_type": item["error_type"],
                "reference_tokens": item["reference_tokens"],
                "predicted_tokens": item["predicted_tokens"],
            })

    before_wer = before_totals["total_errors"] / before_totals["total_ref"]
    before_accuracy = 1 - before_wer

    after_wer = after_totals["total_errors"] / after_totals["total_ref"]
    after_accuracy = 1 - after_wer

    summary_df = pd.DataFrame([
        {
            "version": "Before postprocessing",
            "rows_evaluated": len(df),
            "insert": before_totals["insert"],
            "delete": before_totals["delete"],
            "replace": before_totals["replace"],
            "correct": before_totals["correct"],
            "total_reference_tokens": before_totals["total_ref"],
            "total_errors": before_totals["total_errors"],
            "wer": before_wer,
            "accuracy": before_accuracy,
        },
        {
            "version": "After postprocessing",
            "rows_evaluated": len(df),
            "insert": after_totals["insert"],
            "delete": after_totals["delete"],
            "replace": after_totals["replace"],
            "correct": after_totals["correct"],
            "total_reference_tokens": after_totals["total_ref"],
            "total_errors": after_totals["total_errors"],
            "wer": after_wer,
            "accuracy": after_accuracy,
        },
    ])

    by_row_df = pd.DataFrame(by_row)
    detail_df = pd.DataFrame(detail_rows)

    summary_df.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")
    by_row_df.to_csv(OUTPUT_BY_ROW, index=False, encoding="utf-8-sig")
    detail_df.to_csv(OUTPUT_DETAIL, index=False, encoding="utf-8-sig")

    print("\n========== POSTPROCESS WER RESULT ==========")
    print("Rows evaluated:", len(df))

    print("\nBefore postprocessing")
    print("Insert:", before_totals["insert"])
    print("Delete:", before_totals["delete"])
    print("Replace:", before_totals["replace"])
    print("Total errors:", before_totals["total_errors"])
    print(f"WER: {before_wer:.4f}")
    print(f"Accuracy: {before_accuracy:.4f}")

    print("\nAfter postprocessing")
    print("Insert:", after_totals["insert"])
    print("Delete:", after_totals["delete"])
    print("Replace:", after_totals["replace"])
    print("Total errors:", after_totals["total_errors"])
    print(f"WER: {after_wer:.4f}")
    print(f"Accuracy: {after_accuracy:.4f}")

    print("\nImprovement")
    print(f"Accuracy change: {after_accuracy - before_accuracy:.4f}")
    print(f"WER change: {before_wer - after_wer:.4f}")

    print("\nSaved summary:", OUTPUT_SUMMARY)
    print("Saved by-row:", OUTPUT_BY_ROW)
    print("Saved detail:", OUTPUT_DETAIL)
    print("===========================================")


if __name__ == "__main__":
    main()