# Terminal Usage — Emotion Talking Head Generation

Run the complete pipeline with **ONE command** — no Jupyter required.
`run_emotion_talking_head.sh` does everything the notebook does:

1. Creates a dedicated virtual environment (`venv/`) with the proven pinned stack
   (`torch==2.8.0 torchvision==0.23.0`, `numpy<2.0.0`, `scipy<1.13`) — the system
   Python is never touched, no kernel, no restarts
2. Downloads all required models from Hugging Face (skipped if already present)
3. Generates the emotion talking-head video(s) via `generator/test_emotions.py`

---

## Quick Start (RunPod / any Linux CUDA host)

```bash
git clone https://github.com/MoshiHead/emotion_talking_head_test_only.git
cd emotion_talking_head_test_only
bash run_emotion_talking_head.sh
```

**Requirements:** CUDA GPU, `python3` (3.10–3.12), `ffmpeg` on PATH (preinstalled on
RunPod images), ~8 GB disk. The first run downloads PyTorch (~3 GB) + the models —
a few minutes, done only once. `venv/` and `checkpoints/` persist on RunPod volumes,
so subsequent runs start generating immediately.

---

## Configuration (environment variables)

All settings are optional environment variables; defaults match the notebook:

| Variable | Description | Default |
|----------|-------------|---------|
| `REF_IMAGE` | Reference image path | `assets/3robert.jpeg` |
| `AUDIO` | Input audio path (`.wav`) | `assets/audio_3.wav` |
| `EMOTION` | `neutral`, `happy`, `sad`, `angry`, `surprise`, `fear`, an id `0`–`5`, or `all` | `happy` |
| `CFG_SCALE` | Audio guidance scale | `3.0` |
| `NFE` | Flow-matching ODE steps | `10` |
| `SEED` | Random seed | `42` |
| `CROP` | `1` = face-alignment crop (keep `0` for robot faces) | `0` |
| `OUTPUT_DIR` | Where videos are saved | `emotion_talking_head_outputs/` |
| `EMOTION_LORA_RANK` | LoRA rank — must match training | `32` |
| `EMOTION_LORA_ALPHA` | LoRA alpha — must match training | `64.0` |
| `USE_STYLE_LORA` | `0` disables the style LoRA merge | `1` |

Emotion IDs: `0=neutral`, `1=happy`, `2=sad`, `3=angry`, `4=surprise`, `5=fear`.

---

## Examples

```bash
# All 6 emotions in one run
EMOTION=all bash run_emotion_talking_head.sh

# Custom image + audio, angry emotion
REF_IMAGE=/path/to/face.jpg AUDIO=/path/to/speech.wav EMOTION=angry \
    bash run_emotion_talking_head.sh

# Higher quality (more ODE steps, stronger audio guidance)
NFE=25 CFG_SCALE=4.0 bash run_emotion_talking_head.sh

# Without the style LoRA
USE_STYLE_LORA=0 bash run_emotion_talking_head.sh

# Human face photo with alignment cropping
REF_IMAGE=/path/to/portrait.jpg CROP=1 bash run_emotion_talking_head.sh
```

---

## What Gets Created

| Path | Contents |
|------|----------|
| `venv/` | Dedicated virtual environment (git-ignored, reusable) |
| `checkpoints/` | Downloaded models: base IMTalker + wav2vec2 + style LoRA + emotion LoRA (git-ignored, reusable) |
| `emotion_talking_head_outputs/` | Generated videos — `<id>_<emotion>.mp4` |

---

## Troubleshooting

- **No NumPy 2.x issues**: all heavy work runs inside `venv/` with pinned
  `numpy<2.0.0` + `scipy<1.13`; the system Python is never touched.
- **Black frames in video**: re-run the script. If the base `generator.ckpt`
  download was interrupted, delete `checkpoints/generator.ckpt` first so it
  re-downloads.
- **`ffmpeg not found`**: install it — `apt-get install -y ffmpeg` on
  Ubuntu/RunPod (it is preinstalled on the standard PyTorch templates).
- **CUDA unavailable**: the script installs torch cu128 wheels
  (sm_75..sm_120: RTX 3090/4090/5090, A100, A40, A6000, L40S, H100) with a cu126
  fallback. Check `nvidia-smi` shows your GPU.
- **Existing venv kept as-is**: if the venv already has a working CUDA PyTorch,
  the script skips the ~3 GB re-download (video encoding no longer depends on
  torchvision).
