"""Run a single prompt with the checkpoint's chat template."""

import argparse


MODELS = {
    "leme-sft": "Yale-BIDS-Chen/LEME-SFT-70B",
    "leme-dpo": "Yale-BIDS-Chen/LEME-DPO-70B",
}


def resolve_model(name):
    return MODELS.get(name, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="leme-sft", help="LEME alias or Hugging Face model ID")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = resolve_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
    messages = [{"role": "user", "content": args.prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
