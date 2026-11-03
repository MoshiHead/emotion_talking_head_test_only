# Emotion-Based Talking Head Generation (Standalone)

A fully **self-contained** pipeline that generates **emotion-controlled talking head videos**
using IMTalker and a **pre-trained emotion LoRA adapter** hosted on Hugging Face:

- Repository: `bitwisemind/emotion_lora_adapter_v1`
- File: `last.ckpt`

All required source code (`generator/`, `renderer/`) is bundled in this repository -- **no
external repository or training pipeline is needed**. It reuses the exact avatar testing /
video-generation method that runs after each training epoch in the full pipeline
(`05_train_emotion_lora.ipynb`), but **without any LoRA training** and without the real-time
PersonaPlex conversation pipeline.

## Quick Start (RunPod, any GPU)

```bash
git clone https://github.com/MoshiHead/emotion_talking_head_test_only.git
cd emotion_talking_head_test_only
jupyter lab   # or open 07_emotion_talking_head_inference.ipynb in VS Code
```

Then run all cells in `07_emotion_talking_head_inference.ipynb`. The notebook will:

1. Verify the bundled source code and locate the repository root
2. Set up the environment -- installs the right CUDA PyTorch build for your GPU
   (cu128 covers RTX 3090/4090/5090, A100, A40, A6000, L40S, H100; cu126 automatic fallback;
   preinstalled RunPod PyTorch builds are used as-is)
3. Download all required models automatically (see table below)
4. Load the avatar (base generator -> style LoRA merge -> emotion LoRA adapter)
5. Generate the talking head video

## Configuration

Edit the **CONFIGURATION** cell to set:

| Setting | Description | Default |
|---------|-------------|---------|
| `REF_IMAGE` | Reference image path | `assets/3robert.jpeg` |
| `AUDIO` | Input audio path (.wav) | `assets/audio_3.wav` |
| `EMOTION` | `neutral`, `happy`, `sad`, `angry`, `surprise`, `fear`, or `all` | `happy` |
| `CFG_SCALE` | Audio guidance scale | `3.0` |
| `NFE` | Flow-matching steps | `10` |
| `SEED` | Random seed | `42` |
| `CROP` | Face-alignment crop (keep `False` for robot faces) | `False` |

Videos are written to `emotion_talking_head_outputs/`.

## Repository Structure

| Path | Description |
|------|-------------|
| `07_emotion_talking_head_inference.ipynb` | The standalone notebook (run this) |
| `generator/` | IMTalker generator source (FM, FMT, wav2vec2, generate, LoRA, options) |
| `renderer/` | IMTalker renderer source (IMTRenderer and its modules) |
| `assets/` | Default reference image + test audio |
| `requirements.txt` | Python dependencies |
| `checkpoints/` | Created at runtime -- downloaded models land here (git-ignored) |
| `emotion_talking_head_outputs/` | Created at runtime -- generated videos land here (git-ignored) |

## Models Downloaded Automatically

| Model | Source | Purpose |
|-------|--------|---------|
| `generator.ckpt`, `renderer.ckpt`, `config.yaml` | `cbsjtu01/IMTalker` | Base IMTalker avatar |
| `wav2vec2-base-960h` | `cbsjtu01/IMTalker` | Audio feature encoder |
| `live_winner/lora/ditto_blink_lora_withaudio_r64_096_continue_2h_last.ckpt` | `niloy629/hdtf_preprocess` (dataset) | Style LoRA merged into the base generator (matches emotion-adapter training) |
| `last.ckpt` | `bitwisemind/emotion_lora_adapter_v1` | Trained emotion LoRA adapter |

## Emotion IDs

| ID | Emotion |
|----|---------|
| 0 | neutral |
| 1 | happy |
| 2 | sad |
| 3 | angry |
| 4 | surprise |
| 5 | fear |

## Notes

- `EMOTION_LORA_RANK = 32` / `EMOTION_LORA_ALPHA = 64.0` must match the training values.
- Keep `CROP = False` for robot / non-human faces (skips face-alignment cropping).
- Changing only the emotion requires re-running just the Generate cell -- models stay loaded
  in memory.
- Requires a CUDA GPU, `ffmpeg` on PATH (preinstalled on RunPod images), and ~2 GB of disk
  space for model downloads.
- CLI alternative: `python generator/test_emotions.py --checkpoint checkpoints/emotion_lora/last.ckpt
  --ref_img assets/3robert.jpeg --audio assets/audio_3.wav --out_dir outputs --cfg 3.0`
