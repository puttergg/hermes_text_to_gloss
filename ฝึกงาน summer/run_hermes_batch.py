import os
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI


BASE_DIR = Path(r"D:\SIIT_Y1\ฝึกงาน summer")

INPUT_FILE = BASE_DIR / "clean_no_conflict_v3.csv"
FEWSHOT_CSV = BASE_DIR / "fewshot_20.csv"

OUTPUT_FILE = BASE_DIR / "groq_result_indirect_hermes.csv"
FAILED_FILE = BASE_DIR / "groq_failed_rows_indirect_hermes.csv"


api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is not set. Set it in PowerShell first.")

client = OpenAI(
    api_key="local-test-key",
    base_url="http://127.0.0.1:8642/v1",
)

MODEL = "hermes-agent"




THAI_COL = "thai_sentence"
REF_COL = "gloss_reference"




MAX_TEST_ROWS = 30
MAX_ATTEMPTS = 30



SYSTEM_PROMPT = """
คุณคือ Thai Weather Forecast to Thai Sign Language Gloss Converter

หน้าที่ของคุณคือแปลงประโยคพยากรณ์อากาศภาษาไทยให้เป็น Thai Sign Language gloss เท่านั้น

กฎสำคัญ:
- Output ต้องเป็นภาษาไทยเท่านั้น
- ห้าม romanize ภาษาไทย เช่น KUNG-THEP, FON, ROI-LA
- ห้ามใช้ภาษาอังกฤษ
- ห้ามใช้ภาษาจีน
- ห้ามอธิบาย
- ห้าม return JSON
- ห้ามใช้ markdown
- ห้ามใช้ backticks
- ใช้ | คั่นระหว่าง gloss token
- Output ต้องเป็น one line เท่านั้น
- ห้ามใช้คำเชื่อมที่ไม่ใช่ gloss เช่น กับ, และ, โดย, ของพื้นที่
- ช่วงตัวเลขต้องใช้รูปแบบ 31|ถึง|33 ไม่ใช่ 31-33
- Follow the few-shot output style.

กฎเรื่องความครบถ้วน:
- You must preserve and convert every weather component from the input:
  region, weather condition, rain percentage, special weather warning,
  low temperature, high temperature, wind direction, wind speed, sea wave, and units.
- Do not stop after region name.
- Do not summarize.
- Do not shorten the output.
- If the input contains temperature or wind information, it must appear in the output.
- If the input contains low temperature and high temperature, both must appear.
- If the input contains wind direction and wind speed, both must appear.
- If the input contains wind speed, include the full wind speed unit: กิโลเมตร|ชั่วโมง.
- Do not stop in the middle of a token.
- The output must end with a complete gloss token.
- Do not output incomplete tokens such as ก, เช, เที่, เปอร์เซ็น, อุณหภูมิต.
"""


def load_fewshot_examples(limit=3) -> str:
    df = pd.read_csv(FEWSHOT_CSV, encoding="utf-8-sig")
    df = df.dropna(subset=[THAI_COL, REF_COL])
    df = df.head(limit)

    examples = []
    for _, row in df.iterrows():
        examples.append(
            f"Input:\n{row[THAI_COL]}\nOutput:\n{row[REF_COL]}"
        )

    return "\n\n".join(examples)


def clean_hermes_output(text) -> str:
    """
    Clean model output formatting only.
    Do not change semantic labels.
    """
    if text is None:
        return ""

    text = str(text).strip()

    text = text.replace("```", "")
    text = text.strip("`").strip()

    text = text.replace("Output:", "").strip()
    text = text.replace("PREDICTED:", "").strip()

    text = text.replace("+", "|")

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\n", "|")

    parts = [p.strip().strip("`") for p in text.split("|") if p.strip()]
    return "|".join(parts)


fewshot_examples = load_fewshot_examples(limit=3)


def call_hermes(sentence: str, max_retries: int = 5) -> str:
    user_prompt = f"""
Study the following examples only to understand the gloss style.

<FEW_SHOT_EXAMPLES>
{fewshot_examples}
</FEW_SHOT_EXAMPLES>

Important:
- The numbers in the examples belong only to those examples.
- Use only the numbers from the CURRENT INPUT below.
- Rules in the current task override any conflicting format in the examples.

<CURRENT_INPUT>
{sentence}
</CURRENT_INPUT>

Convert CURRENT_INPUT into Thai Sign Language gloss.

Return exactly one pipe-separated line.
Do not include "Output:", explanations, reasoning, or any English text.
""".strip()

    payload = {
        "model": "hermes-agent",
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "temperature": 0,
        "max_tokens": 400,
    }


    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                 max_tokens=1200,
            )

            return clean_hermes_output(response.choices[0].message.content)

        except Exception as e:
            error_text = str(e)
            print("ERROR:", error_text)

            if "api key" in error_text.lower() or "401" in error_text:
                return "__INVALID_API_KEY__"

            if "429" in error_text or "quota" in error_text.lower() or "rate limit" in error_text.lower():
                wait_time = 45
                print(f"Groq quota/rate limit hit. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                continue

            if "503" in error_text or "unavailable" in error_text.lower():
                wait_time = 30
                print(f"Model busy. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                continue

            return "__REQUEST_ERROR__"

    return "__REQUEST_ERROR__"
def is_incomplete_output(output: str, thai_sentence: str) -> bool:
    output = str(output).strip()

    if not output:
        return True

    bad_endings = ["ก", "เช", "เที่", "เปอร์เซ็น", "อุณหภูมิต"]
    if any(output.endswith(bad) for bad in bad_endings):
        return True

    # If input has number range like 22-24, output should not keep 22-24
    # It should become 22|ถึง|24
    if "-" in output:
        return True

    # If input has wind speed, output should include wind unit
    if "ความเร็ว" in thai_sentence:
        if "กิโลเมตร" not in output and "กม" not in output:
            return True
        if "ชั่วโมง" not in output and "ชม" not in output:
            return True

    # If input has low/high temperature, output should include both
    if "อุณหภูมิต่ำสุด" in thai_sentence and "อุณหภูมิสูงสุด" in thai_sentence:
        if "อุณหภูมิต่ำ" not in output:
            return True
        if "อุณหภูมิสูง" not in output:
            return True

    return False


def save_failed_row(failed_rows, source_row, thai_sentence, gloss_reference, hermes_output, fail_reason):
    failed_rows.append({
        "source_row": source_row,
        "thai_sentence": thai_sentence,
        "gloss_reference": gloss_reference,
        "hermes_output": hermes_output,
        "fail_reason": fail_reason,
    })

    failed_df = pd.DataFrame(failed_rows)
    failed_df.to_csv(FAILED_FILE, index=False, encoding="utf-8-sig")


def main():
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    print("Input file:", INPUT_FILE)
    print("Output file:", OUTPUT_FILE)
    print("Failed file:", FAILED_FILE)
    print("Columns:", df.columns.tolist())
    print("Total rows:", len(df))

    if THAI_COL not in df.columns:
        raise ValueError(f"Missing column: {THAI_COL}")

    if REF_COL not in df.columns:
        raise ValueError(f"Missing column: {REF_COL}")

    results = []
    failed_rows = []

    # Resume successful rows only
    if OUTPUT_FILE.exists():
        old_df = pd.read_csv(OUTPUT_FILE, encoding="utf-8-sig")
        results = old_df.to_dict("records")
        done_source_rows = set(old_df["source_row"].astype(str))
        print(f"Found existing output. Resume mode. Already done: {len(done_source_rows)} rows")
    else:
        done_source_rows = set()

    # Load old failed rows for record only.
    # During testing, we DO NOT skip failed rows because many failures may be quota/incomplete.
    if FAILED_FILE.exists():
        old_failed_df = pd.read_csv(FAILED_FILE, encoding="utf-8-sig")
        failed_rows = old_failed_df.to_dict("records")
        print(f"Found existing failed rows: {len(failed_rows)} rows")
    else:
        failed_rows = []

    success_count = 0
    attempt_count = 0

    for i, row in df.iterrows():
        source_row = str(row.get("source_row", ""))

        if source_row in done_source_rows:
            continue

        attempt_count += 1
        if attempt_count > MAX_ATTEMPTS:
            print(f"Stopped after {MAX_ATTEMPTS} attempts.")
            break

        thai_sentence = row.get(THAI_COL, "")
        gloss_reference = row.get(REF_COL, "")

        print("\n" + "=" * 60)
        print(f"[attempt {attempt_count}/{MAX_ATTEMPTS}] dataset_index={i + 1}/{len(df)} source_row={source_row}")
        print("THAI:")
        print(thai_sentence)

        hermes_output = call_hermes(thai_sentence)

        # 1. Stop immediately if API key is invalid
        if hermes_output == "__INVALID_API_KEY__":
            print("STOP: Invalid Groq API key. Fix the API key before running again.")
            break

        # 2. Stop immediately if daily quota is reached
        if hermes_output == "__DAILY_QUOTA__":
            print("STOP:  Groq daily quota reached. Try again tomorrow or change model.")
            break

        # 3. Handle API/request failure
        if hermes_output in ["__TIMEOUT__", "__REQUEST_ERROR__", "__MODEL_BUSY__"]:
            print("SKIP ROW because Hermes failed:", hermes_output)

            save_failed_row(
                failed_rows,
                source_row,
                thai_sentence,
                gloss_reference,
                hermes_output,
                hermes_output,
            )
            continue

        # 4. Empty output
        if not str(hermes_output).strip():
            print("SKIP ROW because Hermes output is empty")

            save_failed_row(
                failed_rows,
                source_row,
                thai_sentence,
                gloss_reference,
                hermes_output,
                "__EMPTY_OUTPUT__",
            )
            continue

        # 5. Incomplete output
        if is_incomplete_output(hermes_output, thai_sentence):
            print("SKIP ROW because Hermes output is incomplete")
            print("INCOMPLETE OUTPUT:")
            print(hermes_output)

            save_failed_row(
                failed_rows,
                source_row,
                thai_sentence,
                gloss_reference,
                hermes_output,
                "__INCOMPLETE_OUTPUT__",
            )
            continue

        print("HERMES:")
        print(hermes_output)

        results.append({
            "source_row": source_row,
            "thai_sentence": thai_sentence,
            "gloss_reference": gloss_reference,
            "hermes_output": hermes_output,
        })

        result_df = pd.DataFrame(results)
        result_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

        success_count += 1
        if success_count >= MAX_TEST_ROWS:
            print(f"Test finished after {MAX_TEST_ROWS} successful rows.")
            break

        time.sleep(15)

    print("\n========== DONE ==========")
    print("Saved:", OUTPUT_FILE)
    print("Saved failed rows:", FAILED_FILE)
    print("Total successful rows:", len(results))
    print("Total failed rows:", len(failed_rows))
    print("==========================")


if __name__ == "__main__":
    main()
