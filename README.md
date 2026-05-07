<h1 align="center">Model Diff to LoRA for ComfyUI</h1>

<p align="center">
  Extract a LoRA from the difference between two <code>MODEL</code> objects.<br>
  Capture the result of any model edit — block-level tweaks, merges, IP-Adapter chains, custom patches — into a single distributable <code>.safetensors</code> file.
</p>

<p align="center">
  <a href="https://buymeacoffee.com/lorasandlenses"><img src="https://img.shields.io/badge/Buy%20me%20a%20coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee"></a>
</p>

---

## What it does

Give the node two `MODEL` objects: one is your "before" (typically the unmodified base), the other is your "after" (with whatever edits you've applied — chained LoRAs, block-level scaling, merge results, IP-Adapter, custom patches, anything). The node walks every trainable weight, computes the per-tensor difference, runs SVD to factor each diff into low-rank LoRA `up` / `down` matrices, and saves the result as a standard `.safetensors` LoRA.

The output LoRA, applied at strength 1.0 to the "before" model, reproduces the "after" model's behaviour. So whatever transformation you'd been doing live in your workflow gets baked into a single portable file.

## When you'd use it

- **Distillation of a node chain** — you've got a workflow that loads model + 3 LoRAs + a layer-scaler + an IP-Adapter, and you want to ship the combined effect as one LoRA.
- **Merge-to-LoRA** — you merged two checkpoints in some external tool and want a smaller LoRA that captures the merge delta.
- **Custom patch capture** — you wrote a one-off model patch and want to share it with someone who doesn't have your patch code.
- **Block-edit baking** — you scaled specific transformer blocks and want to save that as a portable file.

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
