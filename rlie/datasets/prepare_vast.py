"""Convert VAST stance CSVs into RLIE's columnar JSON.

VAST labels are 0 (con / AGAINST), 1 (pro / FAVOR), and 2 (neutral). This
keeps only the two polar classes so the task stays binary. The comment is the
original ``post`` text. The topic is the crowdsourced ``new_topic``, falling
back to ``ori_topic`` and then ``topic_str``.

The bundled release has ``vast_train.csv`` and ``vast_dev.csv``. ``vast_test.csv``
is converted too when it is present. Train and dev also get stratified short
files, 500 and 200 rows, drawn with seed 0 so each class keeps its source share.

Example::

    python -m rlie.datasets.prepare_vast --input data/real/vast
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


SPLIT_OUTPUTS = {
    "vast_train.csv": "vast_train.json",
    "vast_dev.csv": "vast_dev.json",
    "vast_test.csv": "vast_test.json",
}
SHORT_OUTPUTS = {
    "vast_train.csv": ("vast_train_short.json", 500),
    "vast_dev.csv": ("vast_dev_short.json", 200),
}
SHORT_SEED = 0


def canonical_label(value) -> Optional[str]:
    """Map a VAST stance code to FAVOR or AGAINST. Neutral and unknown values are dropped."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and float(value) == int(value):
        value = int(value)
    text = str(value).strip().lower()
    try:
        number = float(text)
    except ValueError:
        number = None
    if number is not None and number == int(number):
        text = str(int(number))
    if text in {"0", "against", "con"}:
        return "AGAINST"
    if text in {"1", "favor", "favour", "pro"}:
        return "FAVOR"
    return None


def _clean_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def comment_text(row: pd.Series) -> str:
    """Original comment. The stopword-stripped ``text_s`` is only a fallback."""
    post = _clean_text(row.get("post"))
    if post:
        return post
    return _clean_text(row.get("text_s"))


def topic_text(row: pd.Series) -> str:
    """Crowdsourced topic, then the heuristic topic, then the tokenized topic string."""
    for field in ("new_topic", "ori_topic", "topic_str"):
        text = _clean_text(row.get(field))
        if text:
            return text
    return ""


def convert_frame(frame: pd.DataFrame) -> tuple[dict, dict]:
    """Filter one CSV frame to polar stances and return columnar JSON plus counts."""
    posts = []
    topics = []
    labels = []
    dropped_nonpolar = 0
    dropped_empty = 0
    for _, row in frame.iterrows():
        label = canonical_label(row.get("label"))
        if label is None:
            dropped_nonpolar += 1
            continue
        post = comment_text(row)
        topic = topic_text(row)
        if not post or not topic:
            dropped_empty += 1
            continue
        posts.append(post)
        topics.append(topic)
        labels.append(label)
    payload = {"post": posts, "topic": topics, "label": labels}
    counts = {
        "input_rows": int(len(frame)),
        "kept": len(labels),
        "favor": labels.count("FAVOR"),
        "against": labels.count("AGAINST"),
        "dropped_nonpolar": dropped_nonpolar,
        "dropped_empty": dropped_empty,
    }
    return payload, counts


def class_allocation(labels: Iterable[str], size: int) -> Dict[str, int]:
    """Give each class a share of ``size`` that matches its share of ``labels``.

    Counts use the largest-remainder method, so they sum to ``size`` and stay
    within the number of rows available for each class.
    """
    counts = Counter(labels)
    total = sum(counts.values())
    if size >= total:
        return dict(counts)
    raw = {label: size * counts[label] / total for label in counts}
    allocation = {label: min(counts[label], math.floor(raw[label])) for label in counts}
    leftover = size - sum(allocation.values())
    order = sorted(counts, key=lambda label: (raw[label] - math.floor(raw[label]), label), reverse=True)
    while leftover > 0:
        progressed = False
        for label in order:
            if leftover <= 0:
                break
            if allocation[label] < counts[label]:
                allocation[label] += 1
                leftover -= 1
                progressed = True
        if not progressed:
            break
    return allocation


def sample_payload(payload: dict, size: int, seed: int = SHORT_SEED) -> dict:
    """Sample ``size`` rows without replacement, preserving class proportions."""
    labels = list(payload["label"])
    if size >= len(labels):
        return {key: list(values) for key, values in payload.items()}
    allocation = class_allocation(labels, size)
    by_class: Dict[str, list] = {}
    for index, label in enumerate(labels):
        by_class.setdefault(label, []).append(index)
    generator = np.random.default_rng(seed)
    chosen = []
    for label in sorted(by_class):
        pool = np.asarray(by_class[label], dtype=int)
        picked = generator.choice(pool, size=allocation[label], replace=False)
        chosen.append(np.asarray(picked, dtype=int))
    selected = np.concatenate(chosen)
    generator.shuffle(selected)
    return {key: [values[int(index)] for index in selected] for key, values in payload.items()}


def write_split(payload: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)


def convert_directory(input_dir: str, output_dir: Optional[str] = None) -> Dict[str, dict]:
    """Convert every known VAST CSV in ``input_dir``. Missing optional splits are skipped."""
    destination = output_dir or input_dir
    written: Dict[str, dict] = {}
    found = False
    for source_name, output_name in SPLIT_OUTPUTS.items():
        source_path = os.path.join(input_dir, source_name)
        if not os.path.exists(source_path):
            continue
        found = True
        frame = pd.read_csv(source_path)
        payload, counts = convert_frame(frame)
        output_path = os.path.join(destination, output_name)
        write_split(payload, output_path)
        counts["output"] = output_path
        short = SHORT_OUTPUTS.get(source_name)
        if short is not None:
            short_name, short_size = short
            short_payload = sample_payload(payload, short_size, SHORT_SEED)
            short_path = os.path.join(destination, short_name)
            write_split(short_payload, short_path)
            short_counts = Counter(short_payload["label"])
            counts["short"] = {
                "output": short_path,
                "kept": len(short_payload["label"]),
                "favor": int(short_counts["FAVOR"]),
                "against": int(short_counts["AGAINST"]),
            }
        written[source_name] = counts
        print(
            f"{source_name}: kept {counts['kept']} "
            f"(FAVOR {counts['favor']}, AGAINST {counts['against']}); "
            f"dropped {counts['dropped_nonpolar']} non-polar and {counts['dropped_empty']} empty "
            f"-> {output_path}"
        )
        if "short" in counts:
            short_counts = counts["short"]
            print(
                f"  short: {short_counts['kept']} "
                f"(FAVOR {short_counts['favor']}, AGAINST {short_counts['against']}) "
                f"-> {short_counts['output']}"
            )
    if not found:
        expected = ", ".join(SPLIT_OUTPUTS)
        raise FileNotFoundError(f"No VAST CSV files in {input_dir}. Expected one of: {expected}")
    return written


def main(argv: Optional[Iterable[str]] = None) -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    default_dir = os.path.join(repo_root, "data", "real", "vast")
    parser = argparse.ArgumentParser(description="Convert VAST CSVs to binary FAVOR/AGAINST JSON.")
    parser.add_argument("--input", default=default_dir, help="Directory containing vast_train.csv and vast_dev.csv.")
    parser.add_argument("--output", default=None, help="Directory for the JSON files. Defaults to --input.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    convert_directory(args.input, args.output)


if __name__ == "__main__":
    main()
