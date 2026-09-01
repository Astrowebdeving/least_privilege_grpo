# Model adapter release

The LoRA adapters are published at
[`tu101/least-privilege-model`](https://huggingface.co/tu101/least-privilege-model).
They target the Apache-2.0 `Qwen/Qwen3.5-4B` base model and use PEFT LoRA rank
32, alpha 64, and dropout 0.

The recommended endpoints are:

- `main/seed-1-checkpoint-1500`: primary preregistered policy.
- `continuation/selected-A0-seed1-checkpoint-400`: selected post-hoc corrective
  continuation from the primary policy.

The repository also contains endpoint adapters for all three primary seeds,
all 12 base-start reward-ablation cells, and all completed production-policy
continuation reward-arm cells. Those additional adapters are experiment
replicates, not independently selected deployment models.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "Qwen/Qwen3.5-4B"
adapter_id = "tu101/least-privilege-model"

tokenizer = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(base_id, device_map="auto")
model = PeftModel.from_pretrained(
    base,
    adapter_id,
    subfolder="main/seed-1-checkpoint-1500",
)
```

See the model repository's `adapter_manifest.json` for paths, campaign roles,
training seeds, steps, byte sizes, and SHA-256 hashes. The release includes final
experiment endpoints, not optimizer states or every intermediate autosave.
