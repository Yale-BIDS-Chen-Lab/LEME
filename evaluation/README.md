# Benchmark inference

Run commands from the repository root after installing `requirements-evaluation.txt`.

```bash
python -m evaluation.run --task public_eye_health_qa --model leme-sft \
  --output model_predictions/leme-sft/public_eye_health_qa.jsonl
```

The runner downloads the selected configuration from
[LEME-Benchmarks](https://huggingface.co/datasets/Yale-BIDS-Chen/LEME-Benchmarks).
Available configurations are `public_eye_health_qa`, `patient_consultation`,
and `title_prediction`. Use `--model leme-dpo` for the preference-tuned model.

Inference uses the checkpoint's chat template, BF16 weights, temperature 0,
the Llama end-of-turn token, and at most 1,024 generated tokens. QA and title
prediction use a 2,048-token context limit; treatment planning uses 4,096.
The default tensor-parallel size is 2. Adjust `--tensor-parallel-size`,
`--max-model-len`, and `--max-new-tokens` for the available hardware and inputs.

Each output row contains the input, reference, and prediction. A companion
`.metrics.json` file contains mean ROUGE-L F1. The paper also reports BERTScore
and clinical evaluations; those are separate from this runner.

## Local benchmark files

AAO Patient QA is not distributed because of licensing restrictions.
EyeRounds Treatment Planning is pending redistribution permission. To evaluate
either task, supply a local file you are authorized to use:

```bash
python -m evaluation.run --task treatment_planning --model leme-sft \
  --data-file /path/to/treatment_planning.json \
  --output model_predictions/leme-sft/treatment_planning.jsonl
```

Local JSON arrays and JSONL files can use `input`/`reference`. The original
archive's `query`/`answer`, `question`/`answer`, and treatment `input`/`output`
fields are also accepted. Use `--limit 5` to check a small batch first.
