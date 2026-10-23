"""VAST preprocessing and task wiring."""

import json
import os
import tempfile
import unittest

import pandas as pd
import yaml

from rlie.bert.templates import TASK_BERT_BUILDERS, vast_segments
from rlie.datasets.labels import vast_extract_label
from rlie.datasets.prepare_vast import canonical_label, class_allocation, convert_directory, convert_frame, sample_payload


class TestVastPreprocess(unittest.TestCase):
    def test_polar_labels_and_topic_fallback(self):
        self.assertEqual(canonical_label(0), "AGAINST")
        self.assertEqual(canonical_label(1), "FAVOR")
        self.assertIsNone(canonical_label(2))
        self.assertEqual(canonical_label("pro"), "FAVOR")
        frame = pd.DataFrame(
            [
                {"post": " I support this. ", "new_topic": " transit ", "ori_topic": "buses", "label": 1},
                {"post": "This is harmful.", "new_topic": "", "ori_topic": " companies ", "label": 0},
                {"post": "No opinion here.", "new_topic": "diet", "label": 2},
                {"post": "   ", "text_s": "", "new_topic": "empty", "label": 1},
                {"post": "", "text_s": " fallback text ", "new_topic": "", "ori_topic": "", "topic_str": "health", "label": 1},
            ]
        )
        payload, counts = convert_frame(frame)
        self.assertEqual(payload["label"], ["FAVOR", "AGAINST", "FAVOR"])
        self.assertEqual(payload["topic"], ["transit", "companies", "health"])
        self.assertEqual(payload["post"][0], "I support this.")
        self.assertEqual(payload["post"][2], "fallback text")
        self.assertEqual(counts["dropped_nonpolar"], 1)
        self.assertEqual(counts["dropped_empty"], 1)
        self.assertEqual(counts["kept"], 3)

    def test_directory_conversion_writes_columnar_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = pd.DataFrame(
                {
                    "post": ["yes", "no", "maybe"],
                    "new_topic": ["tax", "tax", "tax"],
                    "label": [1, 0, 2],
                }
            )
            frame.to_csv(os.path.join(tmp, "vast_train.csv"), index=False)
            written = convert_directory(tmp)
            self.assertEqual(set(written), {"vast_train.csv"})
            with open(os.path.join(tmp, "vast_train.json"), encoding="utf-8") as handle:
                payload = json.load(handle)
            with open(os.path.join(tmp, "vast_train_short.json"), encoding="utf-8") as handle:
                short = json.load(handle)
        self.assertEqual(payload["label"], ["FAVOR", "AGAINST"])
        self.assertEqual(len(payload["post"]), len(payload["topic"]))
        self.assertEqual(short["label"], ["FAVOR", "AGAINST"])

    def test_short_sample_keeps_source_class_shares(self):
        labels = ["FAVOR"] * 7 + ["AGAINST"] * 3
        self.assertEqual(class_allocation(labels, 5), {"FAVOR": 4, "AGAINST": 1})
        payload = {
            "post": [f"p{index}" for index in range(10)],
            "topic": [f"t{index}" for index in range(10)],
            "label": labels,
        }
        first = sample_payload(payload, 5, seed=0)
        second = sample_payload(payload, 5, seed=0)
        self.assertEqual(first, second)
        self.assertEqual(len(first["label"]), 5)
        self.assertEqual(first["label"].count("FAVOR"), 4)
        self.assertEqual(first["label"].count("AGAINST"), 1)
        self.assertEqual(len(set(first["post"])), 5)

    def test_task_template_and_label_parser(self):
        specs = os.path.join(os.path.dirname(__file__), "..", "rlie", "task_specs", "real", "vast", "task.yaml")
        with open(specs, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        self.assertEqual(tuple(data["bert_input"]["fields"]), tuple(TASK_BERT_BUILDERS["vast"].fields))
        self.assertEqual(vast_segments(pd.Series({"post": "A comment", "topic": "housing"})), ["A comment", "housing"])
        self.assertEqual(vast_extract_label("Reasoning. {Final answer: FAVOR}"), "FAVOR")
        self.assertEqual(vast_extract_label("Final answer: against"), "AGAINST")
        self.assertEqual(vast_extract_label("Final answer: not applicable"), "other")
