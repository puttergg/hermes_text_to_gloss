import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class EvalResult:
    exact_match: bool = False
    total_predicted_labels: int = 0
    total_reference_labels: int = 0
    matched_labels: int = 0
    predicted_lines: List[str] = field(default_factory=list)
    reference_lines: List[str] = field(default_factory=list)

    @property
    def precision(self) -> float:
        if self.total_predicted_labels == 0:
            return 0.0
        return self.matched_labels / self.total_predicted_labels

    @property
    def recall(self) -> float:
        if self.total_reference_labels == 0:
            return 0.0
        return self.matched_labels / self.total_reference_labels

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @property
    def accuracy(self) -> float:
        return 1.0 if self.exact_match else 0.0


def normalize_line(line: str) -> str:
    line = str(line).strip()
    line = re.sub(r"\s+", " ", line)
    line = re.sub(r"\s*\|\s*", "|", line)
    line = re.sub(r"\s*\+\s*", "+", line)
    return line


def split_labels_from_line(line: str) -> List[str]:
    line = normalize_line(line)

    if not line:
        return []

    # รองรับทั้ง format ใหม่ | และ format เก่า +
    parts = re.split(r"\s*(?:\||\+)\s*", line)

    return [p.strip() for p in parts if p.strip()]


def flatten_labels(text: str) -> List[str]:
    labels = []

    for line in str(text).splitlines():
        line = normalize_line(line)

        if not line:
            continue

        # แปลง + เป็น | ก่อน เพื่อรองรับทั้ง format เก่าและใหม่
        line = line.replace("+", "|")

        # แยก token ด้วย |
        parts = line.split("|")

        for part in parts:
            part = part.strip()
            if part:
                labels.append(part)

    return labels


def split_lines(text: str) -> List[str]:
    return [normalize_line(line) for line in str(text).splitlines() if normalize_line(line)]


def evaluate(predicted_text: str, reference_text: str) -> EvalResult:
    pred_lines = split_lines(predicted_text)
    ref_lines = split_lines(reference_text)

    pred_labels = flatten_labels(predicted_text)
    ref_labels = flatten_labels(reference_text)

    pred_counts = {}
    for label in pred_labels:
        pred_counts[label] = pred_counts.get(label, 0) + 1

    ref_counts = {}
    for label in ref_labels:
        ref_counts[label] = ref_counts.get(label, 0) + 1

    matched = 0

    for label, count in pred_counts.items():
        ref_count = ref_counts.get(label, 0)
        matched += min(count, ref_count)

    exact_match = pred_labels == ref_labels

    return EvalResult(
        exact_match=exact_match,
        total_predicted_labels=len(pred_labels),
        total_reference_labels=len(ref_labels),
        matched_labels=matched,
        predicted_lines=pred_lines,
        reference_lines=ref_lines,
    )
def print_evaluation(result: EvalResult) -> None:
    print("PREDICTED LINES:")
    for line in result.predicted_lines:
        print(f"  {line}")

    print("REFERENCE LINES:")
    for line in result.reference_lines:
        print(f"  {line}")

    print("ACCURACY:", "1.0 (exact match)" if result.exact_match else "0.0 (mismatch)")
    print(f"PREDICTED LABEL COUNT: {result.total_predicted_labels}")
    print(f"REFERENCE LABEL COUNT:  {result.total_reference_labels}")
    print(f"MATCHED LABELS:         {result.matched_labels}")
    print(f"PRECISION:              {result.precision:.4f}")
    print(f"RECALL:                 {result.recall:.4f}")
    print(f"F1:                     {result.f1:.4f}")