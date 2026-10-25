"""Adapt public benchmark records and the original local JSON files."""

from leme.io import read_records

PUBLIC_TASKS = ("public_eye_health_qa", "patient_consultation", "title_prediction")
TASKS = ("patient_qa", *PUBLIC_TASKS, "treatment_planning")
QA_INSTRUCTION = (
    "The task is to provide an appropriate and professional response to a clinical inquiry "
    "submitted by an individual seeking advice from an Ophthalmologist.\n\nQuestion: "
)
TITLE_INSTRUCTION = (
    "The task is to generate the most appropriate paper title based on the given "
    "ophthalmology-related abstract.\n\nInput: "
)
PLAN_INSTRUCTION = "Briefly summarize the treatments that should be administered to the given patient's note.\n\n"


def normalize_record(row, task, index):
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    value = row.get("predict_title", row) if task == "title_prediction" else row
    question = next((value[k] for k in ("input", "query", "question") if k in value), None)
    answer = next((value[k] for k in ("reference", "answer", "output") if k in value), None)
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Record {index} has no input text.")
    if not isinstance(answer, str):
        raise ValueError(f"Record {index} has no reference text.")
    result = {"id": row.get("id", f"{task}-{index:04d}"), "input": question, "reference": answer}
    for key in ("source", "pmid"):
        if key in row:
            result[key] = row[key]
    return result


def load_benchmark(task, data_file=None):
    if data_file:
        rows = read_records(data_file)
    elif task in PUBLIC_TASKS:
        from datasets import load_dataset
        rows = load_dataset("Yale-BIDS-Chen/LEME-Benchmarks", task, split="test")
    else:
        raise ValueError(f"{task} is not distributed on the Hub; provide an authorized local file with --data-file.")
    return [normalize_record(row, task, i) for i, row in enumerate(rows)]


def make_prompt(task, question):
    prefix = TITLE_INSTRUCTION if task == "title_prediction" else PLAN_INSTRUCTION if task == "treatment_planning" else QA_INSTRUCTION
    return prefix + question
