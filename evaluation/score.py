"""Compute ROUGE-L and BERTScore from saved benchmark predictions."""

import argparse
import hashlib
import json
from pathlib import Path

from evaluation.metrics import METRICS, score_predictions
from leme.io import read_records, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path, help="JSON or JSONL records with reference/prediction fields; gzip is supported")
    parser.add_argument("--output", type=Path, help="Summary JSON path; default: <predictions-stem>.scores.json")
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=list(METRICS))
    parser.add_argument("--device", default="auto", help="auto, cpu, or a PyTorch device such as cuda:0")
    parser.add_argument("--batch-size", type=int, default=64, help="BERTScore batch size")
    parser.add_argument("--bootstrap-samples", type=int, default=0, help="Optional case-bootstrap replicates for percentile 95%% intervals")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap random seed")
    args = parser.parse_args()
    if args.batch_size < 1 or args.bootstrap_samples < 0 or args.seed < 0:
        parser.error("Batch size must be positive; bootstrap samples and seed must be nonnegative")
    output = args.output or args.predictions.with_suffix(".scores.json")
    if output.suffix != ".json":
        parser.error("--output must end in .json")
    per_example_output = output.with_suffix(".jsonl")
    if args.predictions.resolve() in (output.resolve(), per_example_output.resolve()):
        parser.error("Score output paths must differ from the prediction input")
    try:
        rows = read_records(args.predictions)
        input_sha256 = hashlib.sha256(args.predictions.read_bytes()).hexdigest()
        summary, per_example = score_predictions(
            rows, metrics=args.metrics, batch_size=args.batch_size, device=args.device,
            bootstrap_samples=args.bootstrap_samples, seed=args.seed,
        )
    except ImportError as error:
        parser.error(f"Missing scoring dependency: {error}. Install requirements-scoring.txt")
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    summary["prediction_file"] = str(args.predictions)
    summary["prediction_sha256"] = input_sha256
    write_jsonl(per_example_output, per_example)
    write_json(output, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
