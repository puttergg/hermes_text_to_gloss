import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

# =========================
# PATH SETUP
# =========================

BASE_DIR = Path(r"D:\SIIT_Y1\ฝึกงาน summer")

REFERENCE_FILE = BASE_DIR / "clean_no_conflict_v3.csv"

TEST_ROWS = 45

OUTPUT_FILE = BASE_DIR / f"qwen_hermes_result_{TEST_ROWS}rows_exp006_prompt3.csv"
FAILED_FILE = BASE_DIR / f"qwen_hermes_failed_{TEST_ROWS}rows_exp006_prompt3.csv"

# =========================
# HERMES LOCAL API SETUP
# =========================
# Flow:
# Python Code -> Hermes Agent -> Ollama RTT API -> qwen3.5:35b

client = OpenAI(
    api_key="local-test-key",
    base_url="http://127.0.0.1:8642/v1",
)

MODEL = "hermes-agent"

SOURCE_COL = "source_row"
THAI_COL = "thai_sentence"


# =========================
# LOAD DATA
# =========================

ref_df = pd.read_csv(REFERENCE_FILE, encoding="utf-8-sig")
ref_df.columns = [str(c).strip().replace("\ufeff", "") for c in ref_df.columns]

run_df = ref_df.head(TEST_ROWS).copy()

print("Input file:", REFERENCE_FILE)
print("Output file:", OUTPUT_FILE)
print("Rows to run:", len(run_df))
print("Columns:", ref_df.columns.tolist())


# =========================
# PROMPT
# =========================

def build_prompt(sentence: str) -> str:
    return f"""
คุณคือระบบแปลงประโยคพยากรณ์อากาศภาษาไทยเป็น Thai Sign Language gloss

กฎสำคัญ:
- ตอบเฉพาะ gloss ภาษาไทยเท่านั้น
- ใช้ | คั่นระหว่าง gloss token
- ห้ามอธิบาย
- ห้ามแสดง reasoning
- ห้ามแปลแบบ word-by-word
- ต้องใช้ style ให้เหมือน reference dataset

กฎ style:
- ใช้ กรุงเทพ|จังหวัด|ใกล้เคียง ไม่ใช่ กรุงเทพฯ|ปริมณฑล
- ใช้ อุณหภูมิต่ำ ไม่ใช่ อุณหภูมิต่ำสุด
- ใช้ อุณหภูมิสูง ไม่ใช่ อุณหภูมิสูงสุด
- ใช้ องศาเซลเซียส เป็น token เดียว
- ใช้ อุณหภูมิลดลง ไม่ใช่ อุณหภูมิ|ลด
- ร้อยละ 40 ให้เขียนเป็น เปอร์เซ็นต์|40
- ความเร็ว 10-20 กม./ชม. ให้เขียนเป็น เร็ว|10|ถึง|20|กิโลเมตร|ชั่วโมง
- ห้ามใส่คำเชื่อม เช่น มี, ของ, และ, โดย, ของพื้นที่

ตัวอย่าง:
ประโยค: มีฝนฟ้าคะนอง ร้อยละ 40 ของพื้นที่
Gloss: ฝนตกหลายพื้นที่|เปอร์เซ็นต์|40

ประโยค: มีฝนฟ้าคะนอง ร้อยละ 70 ของพื้นที่ และมีฝนตกหนักบางแห่ง
Gloss: ฝนตกหลายพื้นที่|เปอร์เซ็นต์|70|ตรง|ฝนตก|หนัก|ไม่แน่นอน

ประโยค: กรุงเทพมหานครและปริมณฑล อากาศเย็นในตอนเช้ากับมีลมแรง อุณหภูมิต่ำสุด 20-22 องศาเซลเซียส อุณหภูมิสูงสุด 29-31 องศาเซลเซียส ลมตะวันออกเฉียงเหนือ ความเร็ว 10-30 กม./ชม.
Gloss: กรุงเทพ|จังหวัด|ใกล้เคียง|หนาว|เช้า|กับ|ลม|อุณหภูมิต่ำ|20|ถึง|22|องศาเซลเซียส|ร้อน|อุณหภูมิสูง|29|ถึง|31|องศาเซลเซียส|ลมตะวันออกเฉียงเหนือ|เร็ว|10|ถึง|30|กิโลเมตร|ชั่วโมง

แปลงประโยคนี้:
{sentence}

Gloss:
""".strip()


def clean_output(text) -> str:
    if text is None:
        return ""

    text = str(text).strip()
    text = text.replace("```", "").strip("`").strip()
    text = text.replace("Gloss:", "").strip()
    text = text.replace("Output:", "").strip()
    text = text.replace("+", "|")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\n", "|")

    parts = [p.strip() for p in text.split("|") if p.strip()]
    return "|".join(parts)


# =========================
# CALL HERMES
# =========================

def call_qwen_through_hermes(sentence: str, max_retries: int = 3) -> str:
    prompt = build_prompt(sentence)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0,
                max_tokens=300,
            )

            output = response.choices[0].message.content
            return clean_output(output)

        except Exception as e:
            print(f"Attempt {attempt} failed:", e)
            time.sleep(3)

    return ""


# =========================
# RESUME MODE
# =========================

if OUTPUT_FILE.exists():
    old_df = pd.read_csv(OUTPUT_FILE, encoding="utf-8-sig")
    results = old_df.to_dict("records")
    done_source_rows = set(old_df[SOURCE_COL].astype(str).str.strip())
    print(f"Found existing output. Resume mode. Already done: {len(done_source_rows)} rows")
else:
    results = []
    done_source_rows = set()

failed_rows = []


# =========================
# RUN LOOP
# =========================

for i, row in run_df.iterrows():
    source_row = str(row[SOURCE_COL]).strip()
    sentence = str(row[THAI_COL]).strip()

    if source_row in done_source_rows:
        continue

    print("\n" + "=" * 60)
    print(f"[{len(results) + 1}/{TEST_ROWS}] source_row={source_row}")
    print("THAI:")
    print(sentence)

    output = call_qwen_through_hermes(sentence)

    print("OUTPUT:", output)
    print("-" * 50)

    if not output:
        failed_rows.append({
            SOURCE_COL: source_row,
            THAI_COL: sentence,
            "hermes_output": output,
            "fail_reason": "empty_output",
        })

        pd.DataFrame(failed_rows).to_csv(
            FAILED_FILE,
            index=False,
            encoding="utf-8-sig"
        )

        continue

    results.append({
        SOURCE_COL: source_row,
        THAI_COL: sentence,
        "hermes_output": output,
    })

    pd.DataFrame(results).to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    time.sleep(0.3)


print("\nDONE")
print("Saved to:", OUTPUT_FILE)
print("Failed saved to:", FAILED_FILE)
print("Total successful rows:", len(results))