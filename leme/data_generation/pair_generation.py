"""Construct preferred/dispreferred response pairs from LLM scores."""

import argparse
import random

from leme.data_generation.client import add_api_args, azure_client, complete, prompt_text, score_value, select_rows
from leme.io import read_records, jsonl_writer


def preference_strategy(row, rng):
    if score_value(row["truthfulness"]) < 5:
        return "restore", False, row["truthfulness"]
    if rng.random() > 0.5:
        return "hallucinate", True, row["truthfulness"]
    if score_value(row["informativeness"]) == 5:
        return "deinform", True, row["informativeness"]
    return "enrich", False, row["informativeness"]


def make_pair(row, client, deployment, rng):
    strategy, original_chosen, explanation = preference_strategy(row, rng)
    prompt = prompt_text(f"preference_{strategy}.txt")
    prompt = prompt.replace("{question}", row["input"])
    prompt = prompt.replace("{ground_truth_answer}", row["response"]).replace("{explanation}", explanation)
    revised = complete(client, prompt, deployment, temperature=0.0)
    if revised == row["response"]:
        raise ValueError("The generated preference pair contains identical responses")
    chosen, rejected = (row["response"], revised) if original_chosen else (revised, row["response"])
    return {"conversations": [{"from": "human", "value": row["input"]}],
            "chosen": {"from": "gpt", "value": chosen}, "rejected": {"from": "gpt", "value": rejected}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    add_api_args(parser)
    args = parser.parse_args()
    rows = select_rows(read_records(args.input), args.start, args.limit)
    client = azure_client()
    rng = random.Random(args.seed)
    with jsonl_writer(args.output) as emit:
        for row in rows:
            emit(make_pair(row, client, args.deployment, rng))


if __name__ == "__main__":
    main()
