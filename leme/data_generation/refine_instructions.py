"""Refine generated QA examples and export ShareGPT instruction records."""

import argparse

from leme.data_generation.client import add_api_args, azure_client, complete, prompt_text, select_rows
from leme.io import read_records, jsonl_writer


def refine_row(row, mode, client, deployment):
    question, answer = row["question"], row["answer"]
    if mode == "multiple_choice":
        prompt = prompt_text("reason_multiple_choice.txt").replace("{question}", question)
        response = complete(client, prompt, deployment)
        if "## Answer:" not in response or response.rsplit("## Answer:", 1)[1].strip() != answer:
            return None
        answer = response.replace("## Explanation:", "").strip()
    elif mode == "open_ended":
        prompt = prompt_text("refine_open_ended.txt").replace("{question}", question)
        prompt = prompt.replace("{answer}", answer).replace("{explanation}", row.get("explanation", ""))
        response = complete(client, prompt, deployment)
        if response.strip().upper() == "ERROR":
            return None
        answer = response.replace("## Your final answer:", "").strip()
    elif mode == "dialogue":
        prompt = prompt_text("validate_dialogue.txt").replace("{question}", question).replace("{answer}", answer)
        verdict = complete(client, prompt, deployment).lower()
        if verdict not in ("valid", "error"):
            raise ValueError("Expected valid or error for dialogue review")
        if verdict == "error":
            return None
    if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
        raise ValueError("Instruction and response must be non-empty text")
    return {"id": row["id"], "conversations": [
        {"from": "human", "value": question}, {"from": "gpt", "value": answer},
    ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", required=True, choices=("open_ended", "multiple_choice", "dialogue", "direct"))
    add_api_args(parser)
    args = parser.parse_args()
    rows = select_rows(read_records(args.input), args.start, args.limit)
    client = None if args.mode == "direct" else azure_client()
    retained = 0
    with jsonl_writer(args.output) as emit:
        for row in rows:
            result = refine_row(row, args.mode, client, args.deployment)
            if result is not None:
                emit(result)
                retained += 1
    print(f"Retained {retained} of {len(rows)} instructions")


if __name__ == "__main__":
    main()
