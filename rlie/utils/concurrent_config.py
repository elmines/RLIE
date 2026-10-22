"""Concurrency configuration and budget allocation utilities.

Provides a single way to compute concurrency limits and allocate a budget
across nested parallel work.
"""
from rlie.utils.logger_config import LoggerConfig
from typing import Dict, List, Tuple


LOGGER = LoggerConfig.get_logger("rlie.concurrent_config")


def calculate_total_concurrent_limit(
    config: Dict,
    num_keys: int,
    retry_overhead: float = 0.15,
) -> int:
    """Compute the total concurrency limit, reserving headroom for retries.

    Args:
        config: Config dict; must contain max_concurrent_per_key.
        num_keys: Number of API keys.
        retry_overhead: Fraction of capacity reserved for retries (default 0.15, i.e. 15%).

    Returns:
        Effective total concurrency limit.

    Raises:
        ValueError: If max_concurrent_per_key is missing from the config.
    """
    per_key = config.get("max_concurrent_per_key")
    if per_key is None:
        raise ValueError(
            "Config must contain 'max_concurrent_per_key' parameter. "
            "The old 'max_concurrent' parameter has been deprecated."
        )

    per_key = int(per_key)
    if per_key <= 0:
        raise ValueError(f"max_concurrent_per_key must be positive, got {per_key}")

    retry_overhead = float(retry_overhead)
    if not 0.0 <= retry_overhead < 1.0:
        raise ValueError(f"retry_overhead must be in [0, 1), got {retry_overhead}")

    # total concurrency = per-key concurrency x key count x (1 - retry overhead)
    theoretical_limit = per_key * num_keys
    effective_limit = int(theoretical_limit * (1.0 - retry_overhead))

    LOGGER.info(
        "Concurrency limit: %d keys x %d per_key = %d theoretical max, "
        "after reserving %.0f%% for retries = %d effective max",
        num_keys,
        per_key,
        theoretical_limit,
        retry_overhead * 100,
        effective_limit,
    )

    return max(1, effective_limit)


def allocate_concurrent_budget(
    total_limit: int,
    outer_parallelism: int,
    inner_parallelism: int,
    min_inner: int = 5,
) -> Tuple[int, int]:
    """Allocate a concurrency budget across a two-level parallel structure.

    Args:
        total_limit: Total concurrency limit.
        outer_parallelism: Outer parallelism (for example, repeat count or model count).
        inner_parallelism: Desired inner parallelism (for example, rule count or sample count).
        min_inner: Minimum guaranteed inner concurrency (default 5).

    Returns:
        (outer_workers, inner_concurrent): outer worker count and inner concurrency.

    Examples:
        >>> allocate_concurrent_budget(85, 3, 5, min_inner=5)
        (3, 28)  # 3 repeats, 28 concurrent requests inside each repeat

        >>> allocate_concurrent_budget(28, 5, 20, min_inner=5)
        (5, 5)   # 5 rules, 5 concurrent requests inside each rule (limited by min_inner)
    """
    total_limit = max(1, int(total_limit))
    outer_parallelism = max(1, int(outer_parallelism))
    inner_parallelism = max(1, int(inner_parallelism))
    min_inner = max(1, int(min_inner))

    # Case 1: a single outer worker gets the entire budget for the inner work.
    if outer_parallelism == 1:
        inner_concurrent = min(total_limit, inner_parallelism)
        LOGGER.debug(
            "Single outer worker: total=%d -> inner=%d (desired=%d)",
            total_limit,
            inner_concurrent,
            inner_parallelism,
        )
        return 1, inner_concurrent

    # Case 2: the budget cannot give every outer worker min_inner.
    min_required = outer_parallelism * min_inner
    if total_limit < min_required:
        # Reduce outer parallelism.
        outer_workers = max(1, total_limit // min_inner)
        inner_concurrent = min_inner
        LOGGER.warning(
            "Budget too small: total=%d < %d x %d = %d, reducing outer parallelism %d -> %d",
            total_limit,
            outer_parallelism,
            min_inner,
            min_required,
            outer_parallelism,
            outer_workers,
        )
        return outer_workers, inner_concurrent

    # Case 3: normal allocation.
    # Keep every outer worker and split the budget evenly across the inner work.
    outer_workers = outer_parallelism
    inner_budget = total_limit // outer_parallelism
    inner_concurrent = max(min_inner, min(inner_budget, inner_parallelism))

    actual_total = outer_workers * inner_concurrent
    LOGGER.debug(
        "Two-level allocation: total=%d -> outer=%d x inner=%d = %d (desired inner=%d)",
        total_limit,
        outer_workers,
        inner_concurrent,
        actual_total,
        inner_parallelism,
    )

    return outer_workers, inner_concurrent
