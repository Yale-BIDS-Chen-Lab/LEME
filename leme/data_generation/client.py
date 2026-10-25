"""Azure OpenAI configuration and generation response parsing."""

import json
import os
from pathlib import Path
import re

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


def azure_client():
    names = ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION")
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise ValueError("Set environment variables: " + ", ".join(missing))
    from openai import AzureOpenAI
    return AzureOpenAI(api_key=os.environ[names[0]], azure_endpoint=os.environ[names[1]],
                       api_version=os.environ[names[2]], max_retries=9)


def add_api_args(parser):
    parser.add_argument("--deployment", default=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)


def select_rows(rows, start=0, limit=None):
    if start < 0 or (limit is not None and limit < 1):
        raise ValueError("Start must be nonnegative and limit must be positive")
    return rows[start:] if limit is None else rows[start:start + limit]


def complete(client, prompt, deployment, temperature=0.8):
    result = client.chat.completions.create(
        model=deployment, messages=[{"role": "user", "content": prompt}],
        temperature=temperature, max_tokens=2048,
    )
    text = result.choices[0].message.content
    if not isinstance(text, str) or not text.strip():
        raise ValueError("The generation API returned an empty response")
    return text.strip()


def json_response(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def score_value(text):
    if not isinstance(text, str):
        raise ValueError("Missing scoring response")
    match = re.search(r"(?:^|\n)\s*#?\s*Score:\s*([1-5])(?:\s|$)", text, re.IGNORECASE)
    if not match:
        raise ValueError("Expected a score from 1 to 5 in '# Score: N' format")
    return int(match.group(1))


def prompt_text(name):
    return (PROMPT_DIR / name).read_text(encoding="utf-8")
