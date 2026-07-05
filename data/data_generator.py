"""Generate synthetic textile return data."""

from pathlib import Path

import numpy as np
import pandas as pd

SOURCES = ["Internet/Shipping", "Store-Customer", "Store-Employee"]
SOURCE_PROBS = [0.75, 0.20, 0.05]

CONDITIONS = [
    "NotLiked",
    "ColorMismatch",
    "Undamaged",
    "Torn",
    "SeamDefect",
    "Stained",
    "MissingPart",
]

CONDITION_PROBS = {
    "Internet/Shipping": [0.55, 0.16, 0.11, 0.05, 0.04, 0.04, 0.05],
    "Store-Customer": [0.25, 0.08, 0.25, 0.07, 0.15, 0.08, 0.12],
    "Store-Employee": [0.04, 0.14, 0.12, 0.16, 0.18, 0.22, 0.14],
}

BRAND_TIERS = ["Premium", "Standard"]
BRAND_PROBS = [0.20, 0.80]

AGE_BUCKET_PROBS = [0.10, 0.40, 0.20, 0.06, 0.04, 0.20]
AGE_DECAY_LAMBDA = np.log(2) / 60  # median ~60 days in tail
MAX_AGE_DAYS = 1080  # 36 months


def _sample_age_days(rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample age in days according to the specified piecewise distribution."""
    buckets = rng.choice(6, size=n, p=AGE_BUCKET_PROBS)

    ages = np.empty(n, dtype=int)
    for i, bucket in enumerate(buckets):
        if bucket == 0:
            ages[i] = 0
        elif bucket == 1:
            ages[i] = rng.integers(1, 8)
        elif bucket == 2:
            ages[i] = rng.integers(8, 15)
        elif bucket == 3:
            ages[i] = rng.integers(15, 22)
        elif bucket == 4:
            ages[i] = rng.integers(22, 29)
        else:
            # Day 29+: truncated exponential decay
            span = MAX_AGE_DAYS - 29
            u = rng.random()
            offset = -np.log(1 - u * (1 - np.exp(-AGE_DECAY_LAMBDA * span))) / AGE_DECAY_LAMBDA
            ages[i] = min(29 + int(offset), MAX_AGE_DAYS)

    return ages


def generate_data(I: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic return data for I units."""
    rng = np.random.default_rng(seed)

    sources = rng.choice(SOURCES, size=I, p=SOURCE_PROBS)

    conditions = np.empty(I, dtype=object)
    for source in SOURCES:
        mask = sources == source
        count = mask.sum()
        if count > 0:
            conditions[mask] = rng.choice(
                CONDITIONS, size=count, p=CONDITION_PROBS[source]
            )

    age_days = _sample_age_days(rng, I)
    brand_tiers = rng.choice(BRAND_TIERS, size=I, p=BRAND_PROBS)

    return pd.DataFrame(
        {
            "unit_id": np.arange(1, I + 1),
            "source": sources,
            "condition": conditions,
            "age_days": age_days,
            "brand_tier": brand_tiers,
        }
    )


if __name__ == "__main__":
    df = generate_data()
    output_path = Path(__file__).parent / "synthetic_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")
