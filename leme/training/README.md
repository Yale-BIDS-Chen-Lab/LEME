# Training

LEME was trained with [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory).
Install it and its distributed-training dependencies in a suitable GPU
environment following the framework's instructions.

Download the released data and prepare the ShareGPT dataset definitions:

```bash
python -m leme.training.prepare_data --output-dir data/training
```

This preserves the SFT conversations and DPO chosen/rejected messages from
[LEME-Training-Data](https://huggingface.co/datasets/Yale-BIDS-Chen/LEME-Training-Data),
and writes `sft.json`, `dpo.json`, and `dataset_info.json`. Use `--revision`
to select a specific dataset commit.

The DPO example starts from the official LEME-SFT checkpoint:

```bash
FORCE_TORCHRUN=1 llamafactory-cli train leme/training/configs/leme_dpo.yaml
```

Run this command from the repository root. The example uses full fine-tuning,
BF16, DeepSpeed ZeRO-3 with CPU offloading, one epoch, and a learning rate of
5e-7. With eight GPUs its effective batch size is 32. Configure GPU selection
and distributed execution for the available hardware.

The paper reports SFT from Meta-Llama-3-70B-Instruct for three epochs with
effective batch size 64 and learning rate 5e-6, followed by the DPO stage.
Refer to the paper and LLaMA-Factory documentation for the training method.
