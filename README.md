# LEME: Open Large Language Models for Ophthalmology with Advanced Reasoning and Clinical Validation

[Paper](https://doi.org/10.1038/s41746-026-03104-4) · [Models and datasets](https://huggingface.co/Yale-BIDS-Chen)

LEME is a family of open-weight ophthalmology language models based on
Meta-Llama-3-70B-Instruct. LEME-SFT uses 211,149 instruction examples derived
from ophthalmology guidelines, StatPearls, and published case reports.
LEME-DPO further aligns responses using 29,747 preference pairs evaluated
for truthfulness and informativeness.

![Study design](assets/figure_1_.png)

The study combines ophthalmology corpus selection, instruction tuning,
preference optimization, five open-ended benchmarks, and downstream clinical
validation. The code below supports model inference, public benchmark
prediction, and training-data preparation and generation.

## Resources

| Resource | Content |
| --- | --- |
| [LEME-SFT-70B](https://huggingface.co/Yale-BIDS-Chen/LEME-SFT-70B) | 70B model after supervised fine-tuning |
| [LEME-DPO-70B](https://huggingface.co/Yale-BIDS-Chen/LEME-DPO-70B) | 70B model after preference optimization |
| [LEME-Corpus](https://huggingface.co/datasets/Yale-BIDS-Chen/LEME-Corpus) | 3,435 guideline snippets, 12,919 StatPearls snippets, and 5,745 case summaries |
| [LEME-Training-Data](https://huggingface.co/datasets/Yale-BIDS-Chen/LEME-Training-Data) | SFT: 211,149 instructions; DPO: 29,747 preference pairs |
| [LEME-Benchmarks](https://huggingface.co/datasets/Yale-BIDS-Chen/LEME-Benchmarks) | Public Eye Health QA, Patient Consultation, and Abstract-to-Title Generation; 1,084 examples |

The paper evaluates five benchmark tasks. AAO Patient QA is not distributed
because of licensing restrictions. Redistribution permission for EyeRounds
Treatment Planning is being sought; its data are not included in the public
benchmark repository.

## Installation

Use Python 3.10 or 3.11 and an environment suitable for running a BF16 70B model.

```bash
pip install -r requirements.txt
```

Install `requirements-evaluation.txt` for vLLM benchmark inference, or
`requirements-generation.txt` for the Azure OpenAI data-generation examples.

## Quick start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Yale-BIDS-Chen/LEME-SFT-70B"  # or Yale-BIDS-Chen/LEME-DPO-70B
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto",
)
messages = [{"role": "user", "content": "Explain the difference between dry and wet AMD."}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_dict=True, return_tensors="pt"
).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

The same inference flow is available from the command line:

```bash
python -m leme.inference --model leme-sft --prompt "Explain the difference between dry and wet AMD."
```

Use `--model leme-dpo` for the preference-tuned model. The model card examples
and this quick start use the checkpoint's chat template and greedy decoding.

## Benchmark inference

```bash
pip install -r requirements-evaluation.txt
python -m evaluation.run --task public_eye_health_qa --model leme-sft \
  --output model_predictions/leme-sft/public_eye_health_qa.jsonl
```

The runner also supports `patient_consultation` and `title_prediction` from
Hugging Face. It uses the study's task instructions, temperature 0, and up to
1,024 generated tokens, and saves predictions and ROUGE-L F1. See
[evaluation/README.md](evaluation/README.md) for local-file evaluation,
hardware options, and the unavailable benchmark tasks.

## Data preparation and training

Load the released training configurations directly:

```python
from datasets import load_dataset
sft = load_dataset("Yale-BIDS-Chen/LEME-Training-Data", "sft", split="train")
dpo = load_dataset("Yale-BIDS-Chen/LEME-Training-Data", "dpo", split="train")
```

To prepare files and dataset definitions for LLaMA-Factory:

```bash
python -m leme.training.prepare_data --output-dir data/training
```

See [training instructions](leme/training/README.md) for the DPO example
configuration, and [data construction](leme/data_generation/README.md) for
corpus selection, instruction generation, consistency filtering, and
preference-pair construction with the original prompts.

## Benchmarks & Results


| # | Benchmark (open-ended) | Data Size | What It Measures | Key Take-away |
|---|-----------------------|-----------|------------------|---------------|
| 1 | **Patient QA** | 76 Q-A pairs (AAO) | Accuracy & completeness on real patient questions answered by board-certified ophthalmologists | LEME-SFT & DPO surpass baselines; best ROUGE-L 0.1953, BERTScore 0.8716. |
| 2 | **Public Eye Health QA** | 281 Q-A pairs (NIH NEI) | Ability to deliver lay-friendly medical info | LEME-SFT & DPO surpass baselines; best ROUGE-L 0.1935, BERTScore 0.8480. |
| 3 | **Patient Consultation** | 303 queries (iCliniq) | Robustness to informal patient questions | LEME-SFT & DPO surpass baselines; best ROUGE-L 0.1478, BERTScore 0.8661. |
| 4 | **Treatment Planning** | 245 cases (EyeRounds) | Generating concise management plans from clinical cases | LEME-SFT & DPO surpass baselines; best ROUGE-L 0.1017, BERTScore 0.8229. |
| 5 | **Abstract-to-Title Generation** | 500 research abstracts (PubMed) | Summarisation & research-assistant capability | LEME-SFT hits ROUGE-L 0.4416 (best) and BERTScore 0.8898. |

**Overall performance**
![Figure 2](assets/figure_2_.png)

*Average across the five tasks (ROUGE-L / BERTScore)*  
- **LEME-SFT 70B:** **0.2160 / 0.8601**  
- **LEME-DPO 70B:** 0.1905 / 0.8512  
- Best publicly available baseline (LLaMA-3 70B): 0.1281 / 0.8354  
- Best proprietary baseline (GPT-4o): 0.1828 / 0.8422  

See the paper for the complete benchmark and clinical evaluation results.

## Contact

For general inquiries, please contact Hyunjae Kim at `hyunjae.kim@yale.edu`.

## License

LEME code, models, data, and other related materials may be used for non-commercial research purposes. Any derivative works must be made publicly available through a public platform, such as GitHub or Hugging Face. For commercial-use inquiries, please contact us (`agreements@yale.edu` or `qingyu.chen@yale.edu`). Please see the [LICENSE file](LICENSE) for the full terms.

Model and dataset repositories preserve their applicable third-party terms.
See [NOTICE](NOTICE) for source-code attribution and retained upstream licenses.

## Citation

```bibtex
@article{kim2026leme,
  title={LEME: open large language models for ophthalmology with advanced reasoning and clinical validation},
  author={Kim, Hyunjae and Ai, Xuguang and Srinivasan, Sahana and Gilson, Aidan and Singer, Maxwell B and Pushpanathan, Krithi and Xie, Qianqian and Park, Jungwoo and Applebaum, Serina and Yang, Gabriel Dawei and others},
  journal={npj Digital Medicine},
  year={2026},
  publisher={Nature Publishing Group UK, London}
}
```
