import pandas as pd

INPUT_CSV = r"D:\SIIT_Y1\ฝึกงาน summer\cleaned_weather_tsl_resolved.csv"

FEWSHOT_CSV = r"D:\SIIT_Y1\ฝึกงาน summer\fewshot_20.csv"
TEST_CSV = r"D:\SIIT_Y1\ฝึกงาน summer\test_80.csv"

df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

df = df.dropna(subset=["thai_sentence", "gloss_reference"]).copy()

# shuffle reproducibly
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

split_index = int(len(df) * 0.2)

fewshot = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()

fewshot.to_csv(FEWSHOT_CSV, index=False, encoding="utf-8-sig")
test.to_csv(TEST_CSV, index=False, encoding="utf-8-sig")

print("Total rows:", len(df))
print("Few-shot 20%:", len(fewshot))
print("Test 80%:", len(test))
print("Saved:", FEWSHOT_CSV)
print("Saved:", TEST_CSV)