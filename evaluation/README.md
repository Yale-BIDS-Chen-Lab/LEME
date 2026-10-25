# Benchmark evaluation

## Generate predictions

Run commands from the repository root after installing `requirements-evaluation.txt`.

```bash
python -m evaluation.run --task patient_consultation --model leme-sft \
  --output model_predictions/leme-sft/patient_consultation.jsonl
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
`.metrics.json` file contains mean ROUGE-L F1.

## Score saved predictions

Install the scoring dependencies and run the scorer on a saved prediction file:

```bash
pip install -r requirements-scoring.txt
python -m evaluation.score \
  --predictions model_predictions/leme-sft/patient_consultation.jsonl \
  --device cuda:0 \
  --batch-size 16 \
  --bootstrap-samples 10000
```

This reads the saved responses and does not load a LEME model. Run it after the
inference process exits. With no `--device`, scoring uses CUDA when available
and otherwise CPU. For CPU scoring, use `--device cpu --batch-size 8`.

The input can be JSON, JSONL, or gzip-compressed JSON/JSONL. Each record must
contain `reference` and `prediction` strings; `id` is retained when present.
Empty predictions remain in the evaluation. Reference text must be nonempty.

| Metric | Settings |
| --- | --- |
| ROUGE-L | F1, stemming enabled |
| BERTScore | `roberta-large`, layer 17, English, IDF off, baseline rescaling off, slow tokenizer |

Scores use the original stored strings and the encoder's standard tokenization
and maximum-length truncation. Reported precision, recall, and F1 are each the
arithmetic mean of the corresponding per-example scores.

By default, both metrics are computed. Use `--metrics rougeL` or
`--metrics bertscore` to select one. The scorer writes:

- `<input-stem>.scores.json`: means, example counts, input SHA-256, and settings.
  BERTScore settings include its configuration hash and software versions.
- `<input-stem>.scores.jsonl`: per-example scores in the input order.

Use `--output path/to/scores.json` to choose a summary path. Its companion
per-example file uses the same stem with `.jsonl`. Existing inference
`.metrics.json` files and prediction files are preserved.

Bootstrap intervals are optional. `--bootstrap-samples 10000` resamples cases
with replacement, preserving the original sample size, and reports the 2.5th
and 97.5th percentiles of the resampled means. The point estimates remain the
original per-example means. Use `--seed` to set the bootstrap seed (default 0).
Clinical evaluations are described separately in the paper.

## Local benchmark files

| Task | Public data |
| --- | --- |
| Patient QA · AAO | Not distributed due to licensing restrictions |
| Treatment Planning · EyeRounds | Not distributed due to licensing restrictions |

To evaluate these tasks, supply a local file you are authorized to use:

```bash
python -m evaluation.run --task treatment_planning --model leme-sft \
  --data-file /path/to/treatment_planning.json \
  --output model_predictions/leme-sft/treatment_planning.jsonl
```

Local JSON arrays and JSONL files can use `input`/`reference`. The original
archive's `query`/`answer`, `question`/`answer`, and treatment `input`/`output`
fields are also accepted. Use `--limit 5` to check a small batch first.



