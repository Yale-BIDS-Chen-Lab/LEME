"""Select ophthalmology-related snippets from an existing text corpus."""

import argparse

from leme.data_generation.client import add_api_args, azure_client, complete, select_rows
from leme.io import corpus_text, read_records, jsonl_writer

INSTRUCTION = (
    "Your task is to determine if a given text snippet from a study material pertains to "
    "ophthalmology or vision/eye health. Respond with *Yes* if the topic is (potentially) "
    "related to ophthalmology; otherwise, respond with *No*. Don't output anything else "
    "other than *Yes* or *No*.\n\n### Text\n"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSON array or JSONL of snippets")
    parser.add_argument("--output", required=True)
    add_api_args(parser)
    args = parser.parse_args()
    rows = select_rows(read_records(args.input), args.start, args.limit)
    client = azure_client()
    selected = 0
    with jsonl_writer(args.output) as emit:
        for index, row in enumerate(rows):
            answer = complete(client, INSTRUCTION + corpus_text(row), args.deployment).lower()
            if answer not in ("yes", "no"):
                raise ValueError(f"Invalid classification for row {args.start + index}: expected Yes or No")
            if answer == "yes":
                emit(row)
                selected += 1
    print(f"Selected {selected} of {len(rows)} snippets")


if __name__ == "__main__":
    main()
