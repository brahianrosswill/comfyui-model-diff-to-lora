<h1 align="center">Model Diff to LoRA for ComfyUI</h1>

<p align="center">
  Bake any chain of LoRAs (and any other model edits) into a single distributable <code>.safetensors</code> LoRA.<br>
  Load 3 LoRAs at the strengths you want, run this node, and walk away with one combined LoRA file you can ship anywhere.
</p>

<p align="center">
  <a href="https://buymeacoffee.com/lorasandlenses"><img src="https://img.shields.io/badge/Buy%20me%20a%20coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee"></a>
</p>

---

## What it does

Give the node two `MODEL` objects:

- `model_before` — your unmodified base model
- `model_after` — the same base with whatever LoRAs (and/or other edits) loaded onto it

The node walks every trainable weight, computes the per-tensor difference, runs SVD to factor each diff into low-rank LoRA `up` / `down` matrices, and saves the result as a standard `.safetensors` LoRA file.

The output LoRA, applied at strength 1.0 to the original base, reproduces the "after" model's behaviour. So whatever combination of LoRAs you had loaded gets baked into one portable file.

## Primary use case: combine multiple LoRAs into one

The most common reason to reach for this node:

```
[Load Checkpoint] ──► model_before
       │
       ▼
[LoRA Loader: identity.safetensors @ 1.0]
       │
       ▼
[LoRA Loader: style.safetensors @ 0.7]
       │
       ▼
[LoRA Loader: details.safetensors @ 0.5] ──► model_after
       │
       └─────────────────────► Model Diff to LoRA ──► combined.safetensors
                                          ▲
                                          │
                  model_before ────────────┘
```

You experiment with stacking LoRAs at the strengths that produce the look you want — identity at 1.0, style at 0.7, details at 0.5 — and once you're happy, this node bakes that *exact* combination into a single file. Now you can:

- Share one LoRA instead of "use these three with these strengths"
- Get back to the same look with one drop-down instead of three loaders + three strength sliders
- Use the combined LoRA as the base for further mixing
- Save a "preset" of your favourite multi-LoRA stack

## Other things it captures

It's not LoRAs-only. The node captures any difference between the two `MODEL` objects, so it also handles:

- **Block-level scaling** (e.g. layer-scaler nodes that boost / suppress specific transformer blocks)
- **External merges** — if you merged two checkpoints in another tool and want a small LoRA that captures the merge delta rather than shipping a whole new checkpoint
- **IP-Adapter and other patch-style edits** — anything ComfyUI applies as a `model_patcher.patches` entry gets correctly baked in
- **Custom one-off patches** — useful for sharing edits with someone who doesn't have your patch code

## Inputs

| Input | Type | Purpose |
|---|---|---|
| `model_before` | MODEL | The baseline (unmodified) model |
| `model_after` | MODEL | The same model with whatever edits applied |
| `rank` | INT | Target LoRA rank (typical: 4–64; higher = more faithful, larger file) |
| `threshold` | FLOAT | Skip layers whose diff norm is below this — keeps the LoRA small by ignoring noise |
| `output_name` | STRING | Filename for the saved `.safetensors` |
| `output_path` | STRING | Where to save (relative to ComfyUI/models/loras by default) |
| `svd_device` | DROPDOWN | `cuda` (fast) or `cpu` (universal). GPU is dramatically faster on big models. |
| `enable` | BOOLEAN | Toggle to actually run the extraction. Default off so the node sits dormant until you want it to fire. |

## Outputs

| Output | Type | Purpose |
|---|---|---|
| `lora_path` | STRING | Absolute path to the saved file |
| `info` | STRING | Multi-line summary: layers extracted / skipped, compression ratio, SVD device, file size |

## Install

1. Drop the `comfyui-model-diff-to-lora` folder into `ComfyUI/custom_nodes/`.
2. Restart ComfyUI.
3. Add the **Model Diff to LoRA** node from the `loaders` / `lora` area.
4. Wire `before` and `after` MODEL inputs, set the rank, click Queue.

No pip installs needed beyond what ComfyUI already ships with.

## Notes

- **Rank choice.** Higher rank = larger file but closer reproduction of the diff. For most edits, rank 16–32 captures >95% of the signal. For very subtle edits, rank 4 may be enough.
- **Threshold.** The default skips layers where the difference is essentially numerical noise — keeps file size sensible. Drop to 0 to extract everything.
- **GPU SVD.** If your "before" and "after" are big (Flux, SDXL), CPU SVD can take minutes. CUDA SVD is typically seconds to tens of seconds. Use CUDA unless you're memory-constrained.
- **The node honours patches.** ComfyUI applies LoRAs and other modifications as `model_patcher.patches` rather than baking them into weights. This node detects patches on the input MODEL objects and applies them via `comfy.lora.calculate_weight()` before computing the diff — so your chained LoRAs and runtime patches are correctly captured.

## Support

If this saves you time or unlocks a workflow you couldn't ship before, consider buying me a coffee:

<a href="https://buymeacoffee.com/lorasandlenses"><img src="https://img.shields.io/badge/Buy%20me%20a%20coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee"></a>
