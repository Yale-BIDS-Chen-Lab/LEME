"""Score saved responses without changing their text or dropping empty outputs."""

import importlib.metadata
import math
from statistics import fmean

METRICS = ("rougeL", "bertscore")


def rouge_l_scores(predictions, references):
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return [scorer.score(reference, prediction)["rougeL"].fmeasure
            for prediction, reference in zip(predictions, references)]


def _bert_scores(predictions, references, batch_size, device):
    import bert_score
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    settings = {
        "model_type": "roberta-large", "num_layers": 17, "lang": "en",
        "idf": False, "rescale_with_baseline": False, "use_fast_tokenizer": False,
        "batch_size": batch_size, "device": device,
    }
    (precision, recall, f1), setting_hash = bert_score.score(
        predictions, references, return_hash=True, **settings,
    )
    settings["hash"] = setting_hash
    settings["versions"] = {name: importlib.metadata.version(name)
                            for name in ("bert-score", "transformers", "torch")}
    return {
        "bertscore_precision": precision.tolist(),
        "bertscore_recall": recall.tolist(),
        "bertscore_f1": f1.tolist(),
    }, settings


def percentile_intervals(scores, samples, seed):
    """Percentile intervals over resampled case means; points stay unchanged."""
    import numpy as np

    intervals = {}
    for name, values in scores.items():
        values = np.asarray(values, dtype=np.float64)
        # Reuse the same resampling indices for every metric.
        rng = np.random.default_rng(seed)
        means = np.empty(samples)
        chunk_size = min(256, max(1, 1_000_000 // len(values)))
        for start in range(0, samples, chunk_size):
            stop = min(start + chunk_size, samples)
            indices = rng.integers(0, len(values), size=(stop - start, len(values)))
            means[start:stop] = values[indices].mean(axis=1)
        low, high = np.quantile(means, [0.025, 0.975])
        intervals[name] = {"low": float(low), "high": float(high)}
    return intervals


def score_predictions(rows, metrics=METRICS, batch_size=64, device="auto",
                      bootstrap_samples=0, seed=0):
    rows = list(rows)
    if not rows:
        raise ValueError("No prediction records found.")
    if not metrics or len(set(metrics)) != len(metrics) or any(name not in METRICS for name in metrics):
        raise ValueError(f"Choose each metric at most once from {METRICS}.")
    if batch_size < 1 or bootstrap_samples < 0 or seed < 0:
        raise ValueError("Batch size must be positive; bootstrap samples and seed must be nonnegative.")
    ids = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Record {index} must contain reference and prediction fields.")
        if not isinstance(row.get("reference"), str) or not row["reference"].strip():
            raise ValueError(f"Record {index} has no nonempty reference text.")
        if not isinstance(row.get("prediction"), str):
            raise ValueError(f"Record {index} has no prediction text; empty strings are allowed.")
        if "id" in row:
            identifier = str(row["id"])
            if identifier in ids:
                raise ValueError(f"Duplicate prediction ID: {identifier}")
            ids.add(identifier)
    predictions = [row["prediction"] for row in rows]
    references = [row["reference"] for row in rows]
    scores, settings = {}, {}
    if "rougeL" in metrics:
        scores["rougeL_f1"] = rouge_l_scores(predictions, references)
        settings["rougeL"] = {"use_stemmer": True}
    if "bertscore" in metrics:
        bert_scores, bert_settings = _bert_scores(predictions, references, batch_size, device)
        scores.update(bert_scores)
        settings["bertscore"] = bert_settings
    for name, values in scores.items():
        if len(values) != len(rows) or any(not math.isfinite(value) for value in values):
            raise ValueError(f"Invalid {name} scores: expected {len(rows)} finite values.")
    summary = {
        "examples": len(rows),
        "empty_predictions": sum(not text.strip() for text in predictions),
        **{name: fmean(values) for name, values in scores.items()},
        "settings": settings,
    }
    if bootstrap_samples:
        summary["bootstrap"] = {
            "method": "percentile", "confidence": 0.95,
            "samples": bootstrap_samples, "seed": seed,
            "intervals": percentile_intervals(scores, bootstrap_samples, seed),
        }
    per_example = [{"id": row.get("id", index),
                    **{name: values[index] for name, values in scores.items()}}
                   for index, row in enumerate(rows)]
    return summary, per_example
