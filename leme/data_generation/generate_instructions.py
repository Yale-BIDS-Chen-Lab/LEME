"""Generate instruction examples using the original study prompts."""

import argparse
import json
import random

from leme.data_generation.client import (
    add_api_args, azure_client, complete, json_response, prompt_text, select_rows,
)
from leme.io import corpus_text, read_records, jsonl_writer

TASKS = ("open_ended", "multiple_choice", "usmle", "clinical", "dialogue")


def question_with_options(row):
    question = row["question"]
    options = row.get("options")
    if options:
        if isinstance(options, dict):
            options = [f"{label}. {options[label]}" for label in ("A", "B", "C", "D")]
        elif not isinstance(options, list) or len(options) != 4:
            raise ValueError("Expected four answer options")
        question += "\n\n" + "\n".join(options)
    return question


def generation_prompt(task, context, rng, examples=None):
    if task in ("open_ended", "multiple_choice"):
        name = "prompts_short_answer.txt" if task == "open_ended" else "prompts_mcq.txt"
        return prompt_text(name) + "\n\n" + context
    if task == "dialogue":
        return prompt_text("dialogue.txt").replace("{context}", context)
    if task == "clinical":
        prompt = prompt_text("prompts_cliniq.txt")
        seeds = json.loads(prompt_text("clinical_seed_questions.json"))
        for i, seed in enumerate(rng.sample(seeds, 5), start=1):
            prompt = prompt.replace(f"{{question_{i}}}", seed)
    else:
        if not examples or len(examples) < 3:
            raise ValueError("USMLE generation requires --examples with at least three question/options/answer records")
        prompt = prompt_text("prompts_usmle.txt")
        for i, example in enumerate(rng.sample(examples, 3), start=1):
            prompt = prompt.replace(f"{{question_{i}}}", example["question"])
            prompt = prompt.replace(f"{{answer_{i}}}", example["answer"])
            for label in ("A", "B", "C", "D"):
                prompt = prompt.replace(f"{{op_{i}{label.lower()}}}", example["options"][label])
    return prompt + "\n\n" + context


def generated_rows(task, result, context, index):
    if task == "dialogue":
        return [{"id": f"dialogue-{index:06d}",
                 "question": "TASK: The task is to response to the patient's health inquiry.\n\nINPUT:"
                             + result["patient"] + "\nOutput:\n",
                 "answer": result["doctor"]}]
    questions = result["questions"]
    if not isinstance(questions, list):
        raise ValueError("Expected a questions array")
    output = []
    for j, qa in enumerate(questions):
        question = question_with_options(qa)
        if task == "clinical":
            question += "\n\nInput:\n" + context
        row = {"id": f"{task}-{index:06d}-{j}", "question": question, "answer": qa["answer"]}
        if "explanation" in qa:
            row["explanation"] = qa["explanation"]
        output.append(row)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--input", help="Optional local JSON/JSONL corpus")
    parser.add_argument("--corpus", choices=("guidelines", "statpearls", "cases"))
    parser.add_argument("--examples", help="USMLE few-shot examples JSON")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    add_api_args(parser)
    args = parser.parse_args()
    examples = read_records(args.examples) if args.examples else None
    if args.task == "usmle" and (not examples or len(examples) < 3):
        parser.error("USMLE generation requires --examples with at least three examples")
    if args.input:
        rows = read_records(args.input)
    else:
        from datasets import load_dataset
        corpus = args.corpus or ("statpearls" if args.task in ("open_ended", "multiple_choice") else "cases")
        rows = list(load_dataset("Yale-BIDS-Chen/LEME-Corpus", corpus, split="train"))
    rows = select_rows(rows, args.start, args.limit)
    rng = random.Random(args.seed)
    client = azure_client()
    count = 0
    with jsonl_writer(args.output) as emit:
        for offset, row in enumerate(rows, start=args.start):
            context = corpus_text(row)
            prompt = generation_prompt(args.task, context, rng, examples)
            response = json_response(complete(client, prompt, args.deployment))
            for result in generated_rows(args.task, response, context, offset):
                emit(result)
                count += 1
    print(f"Generated {count} examples")


if __name__ == "__main__":
    main()
