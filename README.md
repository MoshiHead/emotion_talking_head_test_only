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
2. Create a dedicated virtual environment (`venv/`) with the proven pinned stack
   `torch==2.8.0 torchvision==0.23.0` (cu128 covers RTX 3090/4090/5090, A100, A40, A6000,
   L40S, H100; cu126 automatic fallback) -- the same venv + subprocess architecture as the
   live-pipeline notebooks, so the notebook kernel never imports numpy/torch and
   **no kernel restart is ever needed**
3. Download all required models automatically (see table below)
4. Run `generator/test_emotions.py` inside the venv (base generator -> style LoRA merge ->
   emotion LoRA adapter -> generate)
5. Display the talking head video(s) inline

## Terminal Usage (no Jupyter)

Prefer the terminal? `run_emotion_talking_head.sh` runs the complete pipeline --
venv setup + model downloads + video generation -- with one command:

```bash
bash run_emotion_talking_head.sh              # defaults: happy emotion
EMOTION=all bash run_emotion_talking_head.sh  # all 6 emotions
```

See [TERMINAL.md](TERMINAL.md) for all configuration options and examples.

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
| `run_emotion_talking_head.sh` | Complete terminal pipeline script (venv setup + downloads + generation) |
| `TERMINAL.md` | Terminal usage instructions for the shell script |
| `generator/` | IMTalker generator source (FM, FMT, wav2vec2, generate, LoRA, options) |
| `renderer/` | IMTalker renderer source (IMTRenderer and its modules) |
| `assets/` | Default reference image + test audio |
| `requirements.txt` | Python dependencies |
| `checkpoints/` | Created at runtime -- downloaded models land here (git-ignored) |
| `emotion_talking_head_outputs/` | Created at runtime -- generated videos land here (git-ignored) |
| `venv/` | Created at runtime -- the dedicated virtual environment (git-ignored) |

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
- Changing the emotion / inputs: edit the CONFIGURATION cell, then re-run the Generate cell.
  The model reloads each run (same behaviour as the CLI script).
- **No kernel restarts / no NumPy 2.x issues**: all heavy work runs inside the dedicated
  `venv/` (created by the environment cell) as a subprocess -- the notebook kernel never
  imports numpy/torch/cv2. The venv pins `numpy<2.0.0` and `scipy<1.13` and verifies the
  full import stack (cv2, scipy, librosa, matplotlib, face_alignment, torchdiffeq) with
  automatic repairs, so even a NumPy 2.x base image works on the first top-to-bottom run.
- **First run**: creating the venv downloads PyTorch (~3 GB) plus dependencies -- a few
  minutes, only once. The venv and checkpoints persist on RunPod volumes.
- Requires a CUDA GPU, `ffmpeg` on PATH (preinstalled on RunPod images), and ~8 GB of disk
  space (venv + model downloads).
- CLI alternative: `venv/bin/python generator/test_emotions.py --checkpoint checkpoints/emotion_lora/last.ckpt
  --ref_img assets/3robert.jpeg --audio assets/audio_3.wav --out_dir outputs --cfg 3.0`
