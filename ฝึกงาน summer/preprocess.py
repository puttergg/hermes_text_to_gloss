import pandas as pd
import re
from pathlib import Path


BASE_DIR = Path(r"D:\SIIT_Y1\ฝึกงาน summer")

SOURCE_CSV = Path(r"D:\SIIT_Y1\ฝึกงาน summer\ประเมินล่ามภาษามือ AI (Update) - ประเมินล่ามภาษามือ AI (Update).csv")

OUTPUT_CLEANED = BASE_DIR / "cleaned_weather_tsl_v2_normalized.csv"
OUTPUT_QUALITY = BASE_DIR / "cleaned_weather_tsl_quality_report_v2.csv"

OUTPUT_DUP_SUMMARY = BASE_DIR / "duplicate_summary_v2.csv"
OUTPUT_MANUAL_CONFLICTS = BASE_DIR / "manual_review_conflicts_v2.csv"
OUTPUT_SAFE_DUPLICATES = BASE_DIR / "safe_duplicates_v2.csv"

THAI_COL = "ประโยคภาษาไทย"
REF_COL = "ลำดับท่ามือที่ถูกต้อง"



def remove_hidden_chars(text):
    """
    ลบ hidden characters เท่านั้น
    ไม่เปลี่ยนความหมาย
    """
    if pd.isna(text):
        return ""

    text = str(text)

    hidden_chars = [
        "\ufeff",  # BOM
        "\u200b",  # zero width space
        "\u200c",
        "\u200d",
        "\u2060",
        "\xa0",   # non-breaking space
    ]

    for ch in hidden_chars:
        text = text.replace(ch, "")

    return text


def clean_spaces(text):
    """
    clean space แบบปลอดภัย
    """
    text = remove_hidden_chars(text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def clean_thai_sentence(text):
    """
    ประโยคไทย:
    - ลบ hidden chars
    - normalize newline
    - clean spaces
    - เก็บเป็น one-line เพื่อใช้ตรวจ duplicate ได้แม่นขึ้น
    """
    text = remove_hidden_chars(text)

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = []
    for line in text.split("\n"):
        line = clean_spaces(line)
        if line:
            lines.append(line)

    return " ".join(lines).strip()


def normalize_units(text):
    """
    Normalize inconsistent weather unit labels.
    Example:
    กม./ชม., กม/ชม, ก.ม./ช.ม., กิโลเมตร/ชั่วโมง
    -> กิโลเมตร|ชั่วโมง
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # wind speed units
    text = re.sub(r"ก\.?\s*ม\.?\s*/\s*ช\.?\s*ม\.?", "กิโลเมตร|ชั่วโมง", text)
    text = re.sub(r"กม\.?\s*/\s*ชม\.?", "กิโลเมตร|ชั่วโมง", text)
    text = re.sub(r"กิโลเมตร\s*/\s*ชั่วโมง", "กิโลเมตร|ชั่วโมง", text)
    text = re.sub(r"กิโลเมตร\s*ต่อ\s*ชั่วโมง", "กิโลเมตร|ชั่วโมง", text)

    # single unit cleanup
    text = re.sub(r"\bกม\.?\b", "กิโลเมตร", text)
    text = re.sub(r"\bชม\.?\b", "ชั่วโมง", text)

    return text


def normalize_number_ranges(text):
    """
    Normalize number ranges.
    Example:
    22-24 -> 22|ถึง|24
    #c 23-24 -> 23|ถึง|24
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # remove #c before number range
    text = re.sub(r"#c\s*(\d+)\s*-\s*(\d+)", r"\1|ถึง|\2", text)

    # normal range
    text = re.sub(r"(\d+)\s*-\s*(\d+)", r"\1|ถึง|\2", text)

    return text


def clean_reference_safe(text):
    """
    reference gloss:
    - hidden chars ออก
    - newline/*/+ -> |
    - normalize unit format เช่น กม./ชม. -> กิโลเมตร|ชั่วโมง
    - normalize number range เช่น 22-28 -> 22|ถึง|28
    - clean spaces around |
    - remove empty parts

    หมายเหตุ:
    การ normalize นี้ไม่ได้เปลี่ยน meaning ของ gloss  
    แต่ทำให้ token format สม่ำเสมอขึ้นสำหรับ WER evaluation
    """
    text = remove_hidden_chars(text)

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # safe separator conversion
    text = text.replace("\n", "|")
    text = text.replace("*", "|")
    text = text.replace("+", "|")

    # new normalization from senior feedback
    text = normalize_units(text)
    text = normalize_number_ranges(text)

    parts = []
    for part in text.split("|"):
        part = clean_spaces(part)
        if part:
            parts.append(part)

    return "|".join(parts).strip()




def main():
    print("Reading source CSV...")
    df = pd.read_csv(SOURCE_CSV, encoding="utf-8-sig")

    print("Columns found:")
    print(df.columns.tolist())

    if THAI_COL not in df.columns:
        raise ValueError(f"Missing Thai column: {THAI_COL}")

    if REF_COL not in df.columns:
        raise ValueError(f"Missing reference column: {REF_COL}")

    # Excel/source row number
    df = df.reset_index(drop=True)
    df["source_row"] = df.index + 2

    # keep original source text/reference
    df["original_thai_sentence"] = df[THAI_COL]
    df["original_reference"] = df[REF_COL]

    # safe cleaned output columns
    df["thai_sentence"] = df[THAI_COL].apply(clean_thai_sentence)
    df["gloss_reference"] = df[REF_COL].apply(clean_reference_safe)

    # IMPORTANT:
    # blank rows must not be treated as duplicate groups
    df["is_blank_row"] = (
        df["thai_sentence"].eq("") &
        df["gloss_reference"].eq("")
    )

    # cleaned dataset keeps only usable rows
    # ถ้าพี่อยากเก็บ blank rows ด้วย ให้เปลี่ยนเป็น cleaned_df = df.copy()
    cleaned_df = df[~df["is_blank_row"]].copy()

    cleaned_df = cleaned_df[
        [
            "source_row",
            "thai_sentence",
            "gloss_reference",
            "original_reference",
        ]
    ]

    cleaned_df.to_csv(OUTPUT_CLEANED, index=False, encoding="utf-8-sig")
    print(f"Saved: {OUTPUT_CLEANED}")




    sentence_counts = cleaned_df["thai_sentence"].value_counts(dropna=False)

    quality_df = cleaned_df.copy()
    quality_df["has_km_short_unit"] = quality_df["gloss_reference"].str.contains(r"กม|ชม", regex=True, na=False)
    quality_df["has_dash_range"] = quality_df["gloss_reference"].str.contains(r"\d+\s*-\s*\d+", regex=True, na=False)
    quality_df["has_hash_c"] = quality_df["gloss_reference"].str.contains(r"#c", regex=True, na=False)
    quality_df["has_duplicate_sentence"] = quality_df["thai_sentence"].map(sentence_counts).gt(1)
    quality_df["has_old_plus"] = quality_df["gloss_reference"].str.contains(r"\+", regex=True, na=False)
    quality_df["has_empty_sentence"] = quality_df["thai_sentence"].eq("")
    quality_df["has_empty_reference"] = quality_df["gloss_reference"].eq("")
    quality_df["line_count_reference"] = quality_df["gloss_reference"].apply(
        lambda x: len(str(x).split("|")) if str(x).strip() else 0
    )

    quality_df.to_csv(OUTPUT_QUALITY, index=False, encoding="utf-8-sig")
    print(f"Saved: {OUTPUT_QUALITY}")


    duplicate_summary_rows = []
    conflict_rows = []
    safe_rows = []

    grouped = cleaned_df.groupby("thai_sentence", dropna=False)

    for thai_sentence, group in grouped:
        duplicate_count = len(group)

        if duplicate_count <= 1:
            continue

        ref_groups = (
            group.groupby("gloss_reference", dropna=False)["source_row"]
            .apply(lambda rows: ", ".join(map(str, rows.tolist())))
            .reset_index()
        )

        unique_reference_count = len(ref_groups)

        references_for_review = []
        for i, row in ref_groups.iterrows():
            references_for_review.append(
                f"[ref_{i+1} rows {row['source_row']}] {row['gloss_reference']}"
            )

        duplicate_summary_rows.append({
            "thai_sentence": thai_sentence,
            "duplicate_count": duplicate_count,
            "unique_reference_count": unique_reference_count,
            "all_source_rows": ", ".join(map(str, group["source_row"].tolist())),
            "references_for_review": " ||| ".join(references_for_review),
        })

        if unique_reference_count > 1:
            # one row per Thai sentence conflict, not repeated source rows
            conflict_row = {
                "thai_sentence": thai_sentence,
                "duplicate_count": duplicate_count,
                "unique_reference_count": unique_reference_count,
                "all_source_rows": ", ".join(map(str, group["source_row"].tolist())),
            }

            for i, row in ref_groups.iterrows():
                conflict_row[f"reference_{i+1}_source_rows"] = row["source_row"]
                conflict_row[f"reference_{i+1}"] = row["gloss_reference"]

            conflict_rows.append(conflict_row)

        else:
            # one row per safe duplicated Thai sentence
            safe_rows.append({
                "thai_sentence": thai_sentence,
                "duplicate_count": duplicate_count,
                "unique_reference_count": unique_reference_count,
                "all_source_rows": ", ".join(map(str, group["source_row"].tolist())),
                "gloss_reference": ref_groups.iloc[0]["gloss_reference"],
            })

    duplicate_summary_df = pd.DataFrame(duplicate_summary_rows)
    conflict_df = pd.DataFrame(conflict_rows)
    safe_df = pd.DataFrame(safe_rows)

    duplicate_summary_df.to_csv(OUTPUT_DUP_SUMMARY, index=False, encoding="utf-8-sig")
    conflict_df.to_csv(OUTPUT_MANUAL_CONFLICTS, index=False, encoding="utf-8-sig")
    safe_df.to_csv(OUTPUT_SAFE_DUPLICATES, index=False, encoding="utf-8-sig")

    print(f"Saved: {OUTPUT_DUP_SUMMARY}")
    print(f"Saved: {OUTPUT_MANUAL_CONFLICTS}")
    print(f"Saved: {OUTPUT_SAFE_DUPLICATES}")


    print("\n========== SUMMARY ==========")
    print(f"Original source rows: {len(df)}")
    print(f"Blank rows removed from cleaned dataset: {df['is_blank_row'].sum()}")
    print(f"Cleaned usable rows: {len(cleaned_df)}")
    print(f"Unique Thai sentences: {cleaned_df['thai_sentence'].nunique(dropna=False)}")
    print(f"Duplicate sentence groups: {len(duplicate_summary_df)}")
    print(f"Manual conflict groups: {len(conflict_df)}")
    print(f"Safe duplicate groups: {len(safe_df)}")
    print("=============================")


if __name__ == "__main__":
    main()
