import json
import logging
import time
from json import JSONDecodeError
from typing import Dict, List, Optional, Sequence, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import pandas as pd
from openai import APIError, OpenAI
from tqdm import tqdm

from rlie.utils.logger_config import LoggerConfig
from rlie.llm.prompt import BasePrompt
from rlie.datasets.data_loader import _normalize_label
from rlie.llm.parsing import (
    allowed_answers,
    append_answer_constraint,
    derive_label_aliases,
    extract_response_text,
    parse_prediction_response,
    should_use_responses_api,
)
from rlie.utils import concurrent_config
from rlie.llm.parse_failure_recorder import record_attempt, record_failure, record_usage
from rlie.core.metrics import compute_accuracy_and_macro_f1


LOGGER = LoggerConfig.get_logger("rlie.rule_scorer")


class RuleScorer:
    """Score rule applicability with the configured LLM backend."""

    def __init__(
        self,
        config: Dict,
        prompt: BasePrompt,
        label_mapping: Dict[str, int],
        *,
        label_values: Optional[Sequence[str]] = None,
        disable_progress: Optional[bool] = None,
    ):
        if "model" not in config:
            raise ValueError("Evaluation config must include a model name")

        self.prompt = prompt
        self.model_name = config["model"]
        self.temperature = float(config.get("temperature", 0.0))
        self.request_timeout = float(config.get("request_timeout", 120))
        self.num_predict = int(config.get("num_predict", 64))
        self.output_dir = config.get("output_dir")

        # Concurrency priority: _concurrent_budget (injected) > max_concurrent_per_key (new) > max_concurrent (old, deprecated).
        if "_concurrent_budget" in config:
            self.max_concurrent = max(1, int(config["_concurrent_budget"]))
            LOGGER.debug("Using injected concurrency budget: %d", self.max_concurrent)
        elif "max_concurrent_per_key" in config:
            # max_concurrent_per_key must be scaled by the API key count.
            # The key count may be unknown here, so store a placeholder and adjust it once the real count is known.
            per_key = int(config["max_concurrent_per_key"])
            self.max_concurrent = max(1, per_key)
            LOGGER.debug("Using max_concurrent_per_key placeholder: %d (will adjust to the real key count)", self.max_concurrent)
        else:
            # Backward compatibility: use the old max_concurrent setting.
            self.max_concurrent = max(
                1,
                int(
                    config.get(
                        "max_concurrent",
                        config.get("max_workers", 1),
                    )
                ),
            )
            LOGGER.debug("Using legacy max_concurrent config: %d", self.max_concurrent)
        self.max_retries = max(1, int(config.get("max_retries", 3)))
        self.max_parse_retries = max(1, int(config.get("max_parse_retries", self.max_retries)))
        self.min_backoff = float(config.get("min_backoff", 1.0))
        self.max_backoff = float(config.get("max_backoff", 30.0))
        self.request_delay = float(config.get("request_delay", 0.0))
        # Disable the progress bar globally to avoid refresh noise from multiple threads; pass disable_progress=False to turn it back on.
        if disable_progress is None:
            self._disable_tqdm = False
        else:
            self._disable_tqdm = bool(disable_progress)
        # thinking tri-state:
        #   None  -> do not send enable_thinking
        #   True  -> force enable_thinking=true
        #   False -> force enable_thinking=false
        disable_flag = config.get("disable_thinking")
        if disable_flag is True:
            self._thinking_mode = False
        elif disable_flag is False:
            self._thinking_mode = True
        else:
            self._thinking_mode = None
        self._reasoning_effort = config.get("reasoning_effort")
        self._chat_template_kwargs = config.get("chat_template_kwargs")
        self._use_responses_api = should_use_responses_api(
            self.model_name,
            bool(config.get("use_responses_api", False)),
        )

        client_cfg = config.get("client", {})

        base_url = client_cfg.get("base_url")
        api_key = client_cfg["api_key"].strip()
        if not base_url or not api_key:
            raise ValueError("Remote evaluation requires client.base_url and api_key(s)")

        if "_concurrent_budget" not in config and "max_concurrent_per_key" in config:
            retry_overhead = float(config.get("retry_overhead", 0.15))
            self.max_concurrent = concurrent_config.calculate_total_concurrent_limit(
                config,
                1,
                retry_overhead,
            )
            LOGGER.debug(
                "Adjusted max_concurrent=%d to the real key count (keys=%d, retry_overhead=%.2f)",
                self.max_concurrent,
                1,
                retry_overhead,
            )

        self.model = self.model_name
        self._base_url = base_url
        self.client = OpenAI(base_url=base_url, api_key=api_key)

        # All tasks treated as binary for parsing; not applicable will be counted as negative.
        self._use_three_state = False
        self.metric_labels = sorted({int(idx) for idx in label_mapping.values()})

        self._build_label_sets(label_mapping, label_values)
        LoggerConfig.get_logger("httpx").setLevel(logging.WARNING)
        LoggerConfig.get_logger("httpcore").setLevel(logging.WARNING)
        for logger_name in ("openai", "openai._base_client", "openai._http"):
            LoggerConfig.get_logger(logger_name).setLevel(logging.WARNING)
            LoggerConfig.get_logger(logger_name).propagate = False

        LOGGER.info(
            "Rule checker initialized with remote model %s (max_concurrent=%d)",
            self.model,
            self.max_concurrent,
        )

    def _build_label_sets(
        self,
        label_mapping: Dict[str, int],
        label_values: Optional[Sequence[str]] = None,
    ):
        self.canonical_positive: List[str] = []
        self.canonical_negative: List[str] = []

        if label_values and len(label_values) >= 2:
            pos_value = label_values[0]
            neg_value = label_values[1]
        else:
            # fallback to mapping order
            positives = [text for text, idx in label_mapping.items() if idx == 1]
            negatives = [text for text, idx in label_mapping.items() if idx == 0]
            pos_value = positives[0] if positives else "positive"
            neg_value = negatives[0] if negatives else "negative"

        self.canonical_positive = [pos_value]
        self.canonical_negative = [neg_value]

        self.positive_labels = { _normalize_label(pos_value) }
        self.negative_labels = { _normalize_label(neg_value) }

        self.positive_synonyms = {"truthful", "yes", "positive", "1"}
        self.negative_synonyms = {"deceptive", "no", "negative", "0"}
        self.neutral_synonyms = {
            "not applicable",
            "na",
            "n a",
            "neutral",
            "unknown",
            "无法判断",
            "不适用",
            "not sure",
        }

        for label_text in self.positive_labels:
            self.positive_synonyms.update(derive_label_aliases(label_text))
        for label_text in self.negative_labels:
            self.negative_synonyms.update(derive_label_aliases(label_text))

    def _format_messages(self, rule: str, samples: pd.DataFrame, idx: int) -> List[Dict[str, str]]:
        messages = self.prompt.inference({rule: None}, samples, idx)
        return messages

    def _format_multiple_messages(
        self,
        hypotheses: Sequence[str],
        samples: pd.DataFrame,
        idx: int,
    ) -> List[Dict[str, str]]:
        messages = self.prompt.multiple_hypotheses_inference(list(hypotheses), samples, idx)
        return append_answer_constraint(
            messages,
            allowed_answers(
                self.canonical_positive,
                self.canonical_negative,
                include_not_applicable=False,
            ),
        )

    def _format_multiple_with_label_messages(
        self,
        weighted_hypotheses: Sequence[Tuple[str, float]],
        bias: float,
        samples: pd.DataFrame,
        idx: int,
        predicted_label: str,
    ) -> List[Dict[str, str]]:
        messages = self.prompt.multiple_hypotheses_inference_with_linear_regression_and_label(
            list(weighted_hypotheses),
            bias,
            predicted_label,
            samples,
            idx,
        )
        return append_answer_constraint(
            messages,
            allowed_answers(
                self.canonical_positive,
                self.canonical_negative,
                include_not_applicable=False,
            ),
        )

    def score_rule(
        self,
        rule: str,
        samples: pd.DataFrame,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> List[int]:
        total = len(samples)
        if total == 0:
            return []

        def _evaluate(row_idx: int) -> int:
            messages = self._format_messages(rule, samples, row_idx)
            return self._call_and_parse(
                messages,
                context={"rule": rule, "sample_idx": row_idx},
            )

        indices = list(range(total))
        results: List[int] = []

        if self.max_concurrent == 1:
            for idx in indices:
                results.append(_evaluate(idx))
                if progress_callback is not None:
                    progress_callback(1)
            return results

        # Sliding window: handle each request as it finishes so one slow request does not block the batch.
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            results = [0] * len(indices)
            future_to_pos = {
                executor.submit(_evaluate, idx): pos
                for pos, idx in enumerate(indices)
            }
            for future in as_completed(future_to_pos):
                pos = future_to_pos[future]
                try:
                    results[pos] = future.result()
                except Exception:
                    results[pos] = 0
                if progress_callback is not None:
                    progress_callback(1)
        return results

    def _call_and_parse(self, messages: List[Dict[str, str]], context: Optional[Dict[str, object]] = None) -> int:
        """Call the model and parse the result; retry on parsing failures."""
        record_attempt(self.output_dir, "evaluation")
        prompt_text = json.dumps(messages, ensure_ascii=False, indent=2)
        last_response: str = ""
        last_error: str = ""
        for attempt in range(1, self.max_parse_retries + 1):
            try:
                response = self._call_model(messages)
                last_response = response or ""
            except Exception as exc:  # pragma: no cover
                last_error = str(exc)
                if attempt >= self.max_parse_retries:
                    LOGGER.warning(
                        "Model call failed after %d attempts; returning not applicable. Error: %s",
                        attempt,
                        exc,
                    )
                    record_failure(
                        output_dir=self.output_dir,
                        stage="evaluation",
                        model=self.model,
                        attempts=attempt,
                        error=last_error,
                        prompt=prompt_text,
                        response=last_response,
                        round_idx=None,
                        rule=(context or {}).get("rule"),
                        sample_idx=(context or {}).get("sample_idx"),
                    )
                    return 0
                LOGGER.warning(
                    "Model call failed (attempt %d/%d); retrying once more. Error: %s",
                    attempt,
                    self.max_parse_retries,
                    exc,
                )
                continue
            try:
                return parse_prediction_response(
                    response,
                    self.canonical_positive,
                    self.canonical_negative,
                    self.positive_synonyms,
                    self.negative_synonyms,
                    self.neutral_synonyms,
                )
            except ValueError as exc:
                last_error = str(exc)
                if attempt >= self.max_parse_retries:
                    LOGGER.warning(
                        "Parse failed after %d attempts; returning not applicable. Error: %s. Full response: %s",
                        attempt,
                        exc,
                        response,
                    )
                    record_failure(
                        output_dir=self.output_dir,
                        stage="evaluation",
                        model=self.model,
                        attempts=attempt,
                        error=last_error,
                        prompt=prompt_text,
                        response=response or last_response,
                        round_idx=None,
                        rule=(context or {}).get("rule"),
                        sample_idx=(context or {}).get("sample_idx"),
                    )
                    return 0
                LOGGER.warning(
                    "Parse failed (attempt %d/%d); retrying once more. Full response: %s",
                    attempt,
                    self.max_parse_retries,
                    response,
                )
        return 0

    def _call_model(self, messages: List[Dict[str, str]]) -> str:
        backoff = self.min_backoff
        client = self.client
        for attempt in range(1, self.max_retries + 1):
            try:
                kwargs = {
                    "timeout": self.request_timeout,
                }

                if self._use_responses_api:
                    kwargs.update(
                        {
                            "model": self.model,
                            "input": messages,
                            "max_output_tokens": self.num_predict,
                            "temperature": self.temperature,
                            "store": False,
                        }
                    )
                    if self._reasoning_effort:
                        kwargs["reasoning"] = {"effort": self._reasoning_effort}
                    response = client.responses.create(**kwargs)
                    content = extract_response_text(response)
                else:
                    kwargs.update(
                        {
                            "model": self.model,
                            "messages": messages,
                            "temperature": self.temperature,
                        }
                    )
                    extra_body = {}
                    # The official DeepSeek API rejects unsupported extra parameters, so skip extra_body.
                    base_url = self._base_url
                    if not (base_url and str(base_url).startswith("https://api.deepseek.com")):
                        if self._thinking_mode is not None:
                            extra_body["enable_thinking"] = self._thinking_mode
                        if self._reasoning_effort:
                            extra_body["reasoning_effort"] = self._reasoning_effort
                        if self._chat_template_kwargs:
                            extra_body.setdefault("chat_template_kwargs", self._chat_template_kwargs)
                        if extra_body:
                            kwargs["extra_body"] = extra_body
                    response = client.chat.completions.create(**kwargs)
                    content = response.choices[0].message.content or ""
                record_usage(self.output_dir, "evaluation", getattr(response, "usage", None))
                # Delay between requests so a burst of concurrency does not hit API rate limits.
                if self.request_delay > 0:
                    time.sleep(self.request_delay)
                return content
            except APIError as exc:
                if self._maybe_toggle_thinking(exc):
                    continue
                if attempt >= self.max_retries:
                    LOGGER.error(
                        "Model request failed after %d attempts [sync]: model=%s; messages=%s; error=%s",
                        attempt,
                        self.model,
                        messages,
                        exc,
                    )
                    return "Final answer: not applicable"
                LOGGER.debug(
                    "Model request failed (attempt %d/%d) [sync]: model=%s; error=%s; messages=%s; backing off %.1fs",
                    attempt,
                    self.max_retries,
                    self.model,
                    exc,
                    messages,
                    backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, self.max_backoff)
                continue
            except JSONDecodeError as exc:  # API returned a non-JSON response
                if attempt >= self.max_retries:
                    LOGGER.error(
                        "JSON decode failed after %d attempts [sync]: model=%s; error=%s; "
                        "this usually means the API returned an error page instead of JSON; check the API service",
                        attempt,
                        self.model,
                        exc,
                    )
                    return "Final answer: not applicable"

                # Log the first failure at debug so it does not flood the log.
                log_func = LOGGER.debug if attempt == 1 else LOGGER.warning
                log_func(
                    "JSON decode failed (attempt %d/%d) [sync]: model=%s; error=%s",
                    attempt,
                    self.max_retries,
                    self.model,
                    exc,
                )
                # JSON errors are usually transient server problems, so back off longer.
                backoff = min(backoff * 3, self.max_backoff)
                time.sleep(backoff)
                continue
            except Exception as exc:  # pragma: no cover
                if attempt >= self.max_retries:
                    LOGGER.error(
                        "Unexpected error after %d attempts [sync]: model=%s; messages=%s; error=%s",
                        attempt,
                        self.model,
                        messages,
                        exc,
                    )
                    return "Final answer: not applicable"

                # Log the first failure at debug so it does not flood the log.
                log_func = LOGGER.debug if attempt == 1 else LOGGER.warning
                log_func(
                    "Unexpected error (attempt %d/%d) [sync]: model=%s; error=%s",
                    attempt,
                    self.max_retries,
                    self.model,
                    exc,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, self.max_backoff)
                continue

        LOGGER.error("Model request failed after retries")
        LOGGER.warning("Model request exhausted retries; falling back to not applicable")
        return "Final answer: not applicable"

    def _maybe_toggle_thinking(self, error: Exception) -> bool:
        message = str(error).lower()
        if "enable_thinking" not in message:
            return False
        if "must be set to false" in message and self._thinking_mode is True:
            LOGGER.info("Model %s requires thinking mode off; setting enable_thinking=false", self.model_name)
            self._thinking_mode = False
            return True
        if "unknown parameter" in message and self._thinking_mode is not None:
            LOGGER.info("Model %s does not support enable_thinking; removing that parameter", self.model_name)
            self._thinking_mode = None
            return True
        return False

    def batch_score_serial(
        self,
        rules: List[str],
        samples: pd.DataFrame,
        labels: Optional[Sequence[int]] = None,
    ) -> List[List[int]]:
        matrix: List[List[int]] = []
        total_samples = len(samples)

        if total_samples == 0:
            if not self._disable_tqdm:
                tqdm.write("No samples provided for scoring; skipping metrics.")
            return [[] for _ in rules]

        labels_array: Optional[List[int]] = list(labels) if labels is not None else None

        for idx, rule in enumerate(rules, start=1):
            bar_desc = f"Rule {idx}/{len(rules)}"
            bar = tqdm(
                total=total_samples,
                desc=bar_desc,
                unit="sample",
                leave=True,
                disable=self._disable_tqdm,
            )
            predictions = self.score_rule(rule, samples, progress_callback=bar.update)

            if labels_array is not None:
                effective_pairs = [
                    (label, pred)
                    for label, pred in zip(labels_array, predictions)
                    if pred != 0
                ]
                coverage = len(effective_pairs) / len(labels_array) if labels_array else 0.0
                if effective_pairs:
                    y_true = [label for label, _ in effective_pairs]
                    y_pred = [1 if pred == 1 else 0 for _, pred in effective_pairs]
                    acc, f1 = compute_accuracy_and_macro_f1(
                        y_true,
                        y_pred,
                        labels=self.metric_labels,
                    )
                    bar.set_postfix(
                        {
                            "acc": f"{acc:.3f}",
                            "f1": f"{f1:.3f}",
                            "cov": f"{coverage:.2f}",
                        }
                    )
                else:
                    bar.set_postfix({"acc": "nan", "f1": "nan", "cov": f"{coverage:.2f}"})

            bar.close()
            matrix.append(predictions)

        return matrix


    def batch_score_parallel(
        self,
        rules: List[str],
        samples: pd.DataFrame,
        labels: Optional[Sequence[int]] = None,
        max_rule_workers: Optional[int] = None,
    ) -> List[List[int]]:
        if not rules or len(samples) == 0:
            if not self._disable_tqdm and len(samples) == 0:
                tqdm.write("No samples provided for parallel scoring; skipping.")
            return [[] for _ in rules]

        original_max_concurrent = self.max_concurrent
        if max_rule_workers is not None:
            rule_workers, adjusted_concurrent = max_rule_workers, original_max_concurrent
        else:
            rule_workers, adjusted_concurrent = concurrent_config.allocate_concurrent_budget(
                total_limit=original_max_concurrent,
                outer_parallelism=len(rules),
                inner_parallelism=len(samples),
                min_inner=5,
            )

        if rule_workers > 1:
            self.max_concurrent = adjusted_concurrent
            LOGGER.info(
                "Cross-rule parallelism: %d rules x %d sample concurrency = %d actual total concurrency (budget=%d)",
                rule_workers,
                adjusted_concurrent,
                rule_workers * adjusted_concurrent,
                original_max_concurrent,
            )

        labels_array = list(labels) if labels is not None else None
        aggregate_bar = None
        aggregate_bar_lock = Lock()
        if rule_workers > 1:
            aggregate_bar = tqdm(
                total=len(rules) * len(samples),
                desc="Rule scoring",
                unit="sample",
                leave=True,
                disable=self._disable_tqdm,
            )

        def update_aggregate_progress(increment: int = 1):
            with aggregate_bar_lock:
                if aggregate_bar is not None:
                    aggregate_bar.update(increment)

        def score_single_rule(args):
            idx, rule = args
            bar_desc = f"Rule {idx}/{len(rules)}"
            bar = tqdm(
                total=len(samples),
                desc=bar_desc,
                unit="sample",
                leave=True,
                disable=self._disable_tqdm or rule_workers > 1,
            )

            try:
                progress_callback = (
                    bar.update if aggregate_bar is None else update_aggregate_progress
                )
                predictions = self.score_rule(
                    rule,
                    samples,
                    progress_callback=progress_callback,
                )
                if labels_array is not None:
                    effective_pairs = [
                        (label, pred)
                        for label, pred in zip(labels_array, predictions)
                        if pred != 0
                    ]
                    coverage = len(effective_pairs) / len(labels_array) if labels_array else 0.0
                    if effective_pairs:
                        y_true = [label for label, _ in effective_pairs]
                        y_pred = [1 if pred == 1 else 0 for _, pred in effective_pairs]
                        acc, f1 = compute_accuracy_and_macro_f1(
                            y_true,
                            y_pred,
                            labels=self.metric_labels,
                        )
                        bar.set_postfix(
                            {"acc": f"{acc:.3f}", "f1": f"{f1:.3f}", "cov": f"{coverage:.2f}"}
                        )
                    else:
                        bar.set_postfix({"acc": "nan", "f1": "nan", "cov": f"{coverage:.2f}"})

                return predictions
            finally:
                bar.close()

        matrix: List[List[int]] = []
        try:
            if rule_workers == 1:
                for args in enumerate(rules, start=1):
                    matrix.append(score_single_rule(args))
            else:
                with ThreadPoolExecutor(max_workers=rule_workers) as executor:
                    matrix = list(executor.map(score_single_rule, enumerate(rules, start=1)))
        finally:
            if aggregate_bar is not None:
                aggregate_bar.close()
            self.max_concurrent = original_max_concurrent

        return matrix


    def batch_score(
        self,
        rules: List[str],
        samples: pd.DataFrame,
        labels: Optional[Sequence[int]] = None,
    ) -> List[List[int]]:
        if len(rules) <= 2:
            LOGGER.debug("Using serial scoring mode (rules=%d)", len(rules))
            return self.batch_score_serial(rules, samples, labels)
        LOGGER.debug("Using parallel scoring mode (rules=%d)", len(rules))
        return self.batch_score_parallel(rules, samples, labels)


    def evaluate_rule_set(
        self,
        hypotheses: Sequence[str],
        samples: pd.DataFrame,
    ) -> List[int]:
        if not hypotheses:
            return []

        total_samples = len(samples)
        if total_samples == 0:
            return []

        bar = tqdm(
            total=total_samples,
            desc="LLM Eval",
            unit="sample",
            leave=True,
            disable=self._disable_tqdm,
        )

        def _evaluate(row_idx: int) -> int:
            messages = self._format_multiple_messages(hypotheses, samples, row_idx)
            return self._call_and_parse(messages, context={"sample_idx": row_idx})

        results: List[int] = []
        try:
            if self.max_concurrent == 1:
                for idx in range(total_samples):
                    results.append(_evaluate(idx))
                    bar.update(1)
            else:
                with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
                    results = [0] * total_samples
                    future_to_pos = {
                        executor.submit(_evaluate, idx): idx
                        for idx in range(total_samples)
                    }
                    for future in as_completed(future_to_pos):
                        pos = future_to_pos[future]
                        try:
                            results[pos] = future.result()
                        except Exception:
                            results[pos] = 0
                        bar.update(1)
        finally:
            bar.close()

        return results


    def evaluate_rule_set_with_label(
        self,
        weighted_hypotheses: Sequence[Tuple[str, float]],
        bias: float,
        samples: pd.DataFrame,
        predicted_labels: Sequence[str],
    ) -> List[int]:
        if not weighted_hypotheses or len(samples) == 0:
            return []
        if len(predicted_labels) != len(samples):
            raise ValueError("Predicted labels length must match number of samples for LLM evaluation")

        total_samples = len(samples)
        bar = tqdm(
            total=total_samples,
            desc="LLM Eval (with label)",
            unit="sample",
            leave=True,
            disable=self._disable_tqdm,
        )

        def _evaluate(row_idx: int) -> int:
            messages = self._format_multiple_with_label_messages(
                weighted_hypotheses,
                bias,
                samples,
                row_idx,
                predicted_labels[row_idx],
            )
            return self._call_and_parse(messages, context={"sample_idx": row_idx})

        results: List[int] = []
        try:
            if self.max_concurrent == 1:
                for idx in range(total_samples):
                    results.append(_evaluate(idx))
                    bar.update(1)
            else:
                with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
                    results = [0] * total_samples
                    future_to_pos = {
                        executor.submit(_evaluate, idx): idx
                        for idx in range(total_samples)
                    }
                    for future in as_completed(future_to_pos):
                        pos = future_to_pos[future]
                        try:
                            results[pos] = future.result()
                        except Exception:
                            results[pos] = 0
                        bar.update(1)
        finally:
            bar.close()

        return results


    def evaluate_rule_set_with_linear_regression(
        self,
        weighted_hypotheses: Sequence[Tuple[str, float]],
        bias: float,
        samples: pd.DataFrame,
    ) -> List[int]:
        if not weighted_hypotheses or len(samples) == 0:
            return []

        total_samples = len(samples)
        bar = tqdm(
            total=total_samples,
            desc="LLM Eval (linear regression)",
            unit="sample",
            leave=True,
            disable=self._disable_tqdm,
        )

        def _evaluate(row_idx: int) -> int:
            messages = self.prompt.multiple_hypotheses_inference_with_linear_regression(
                list(weighted_hypotheses),
                bias,
                samples,
                row_idx,
            )
            messages = append_answer_constraint(
                messages,
                allowed_answers(
                    self.canonical_positive,
                    self.canonical_negative,
                    include_not_applicable=False,
                ),
            )
            return self._call_and_parse(messages, context={"sample_idx": row_idx})

        results: List[int] = []
        try:
            if self.max_concurrent == 1:
                for idx in range(total_samples):
                    results.append(_evaluate(idx))
                    bar.update(1)
            else:
                with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
                    results = [0] * total_samples
                    future_to_pos = {
                        executor.submit(_evaluate, idx): idx
                        for idx in range(total_samples)
                    }
                    for future in as_completed(future_to_pos):
                        pos = future_to_pos[future]
                        try:
                            results[pos] = future.result()
                        except Exception:
                            results[pos] = 0
                        bar.update(1)
        finally:
            bar.close()

        return results
