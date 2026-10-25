"""Score LEME-SFT responses for truthfulness and informativeness."""

import argparse

from leme.data_generation.client import add_api_args, azure_client, complete, prompt_text, score_value, select_rows
from leme.io import read_records, jsonl_writer


def score_row(row, client, deployment):
    question = row["input"]
    response = row.get("prediction", row.get("response"))
    if not isinstance(question, str) or not isinstance(response, str) or not response.strip():
        raise ValueError("Each record needs input and prediction or response text")
    suffix = question + "\n\nResponse: " + response
    truth = complete(client, prompt_text("score_truthfulness.txt") + suffix, deployment, temperature=0.0)
    informativeness = None
    if score_value(truth) == 5:
        informativeness = complete(client, prompt_text("score_informativeness.txt") + suffix,
                                  deployment, temperature=0.0)
        score_value(informativeness)
    return {"input": question, "response": response, "truthfulness": truth, "informativeness": informativeness}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    add_api_args(parser)
    args = parser.parse_args()
    rows = select_rows(read_records(args.input), args.start, args.limit)
    client = azure_client()
    with jsonl_writer(args.output) as emit:
        for row in rows:
            emit(score_row(row, client, args.deployment))


if __name__ == "__main__":
    main()
