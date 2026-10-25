"""Download released SFT/DPO data and create LLaMA-Factory dataset metadata."""

import argparse
from pathlib import Path

from leme.io import write_json


def dataset_info():
    tags = {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"}
    return {
        "leme_sft": {"file_name": "sft.json", "formatting": "sharegpt",
                     "columns": {"messages": "conversations"}, "tags": tags},
        "leme_dpo": {"file_name": "dpo.json", "formatting": "sharegpt", "ranking": True,
                     "columns": {"messages": "conversations", "chosen": "chosen", "rejected": "rejected"},
                     "tags": tags},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/training")
    parser.add_argument("--revision", default="main", help="Optional pinned Hugging Face revision")
    args = parser.parse_args()
    from datasets import load_dataset

    folder = Path(args.output_dir)
    for name in ("sft", "dpo"):
        rows = load_dataset("Yale-BIDS-Chen/LEME-Training-Data", name, split="train", revision=args.revision)
        write_json(folder / f"{name}.json", list(rows))
        print(f"{name}: {len(rows):,} examples")
    write_json(folder / "dataset_info.json", dataset_info())


if __name__ == "__main__":
    main()
