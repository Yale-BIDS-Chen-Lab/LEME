"""Generate benchmark predictions using the study's LEME inference settings."""

import argparse
import json
from pathlib import Path
from statistics import fmean

from evaluation.data import TASKS, load_benchmark, make_prompt
from evaluation.metrics import rouge_l_scores
from leme.inference import resolve_model
from leme.io import write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--model", "--model_name", default="leme-sft")
    parser.add_argument("--data-file", help="Optional local JSON or JSONL benchmark")
    parser.add_argument("--output", required=True, help="Prediction JSONL path")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--limit", type=int, help="Run only the first N examples")
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.tensor_parallel_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Token, GPU, and example limits must be positive")
    max_length = args.max_model_len if args.max_model_len is not None else (4096 if args.task == "treatment_planning" else 2048)
    if max_length <= args.max_new_tokens:
        parser.error("--max-model-len must leave room for both the prompt and generated tokens")
    try:
        rows = load_benchmark(args.task, args.data_file)
    except ValueError as error:
        parser.error(str(error))
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        parser.error("No benchmark examples found")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=resolve_model(args.model), dtype="bfloat16", max_model_len=max_length,
        gpu_memory_utilization=0.9 if args.task == "title_prediction" else 0.95,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.task == "treatment_planning",
    )
    tokenizer = llm.get_tokenizer()
    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": make_prompt(args.task, row["input"])}],
        add_generation_prompt=True, tokenize=False,
    ) for row in rows]
    stop_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if stop_id is None or stop_id == tokenizer.unk_token_id:
        parser.error("The selected tokenizer has no Llama end-of-turn token")
    outputs = llm.generate(prompts, SamplingParams(
        temperature=0.0, max_tokens=args.max_new_tokens, stop_token_ids=[stop_id],
    ))
    if len(outputs) != len(rows):
        raise RuntimeError("Prediction count does not match benchmark size")
    predictions = [{**row, "prediction": output.outputs[0].text} for row, output in zip(rows, outputs)]
    score = fmean(rouge_l_scores([row["prediction"] for row in predictions],
                                [row["reference"] for row in predictions]))
    write_jsonl(args.output, predictions)
    metrics = {"model": resolve_model(args.model), "task": args.task,
               "examples": len(predictions), "rougeL_f1": score}
    write_json(Path(args.output).with_suffix(".metrics.json"), metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
