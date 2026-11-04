#!/usr/bin/env bash
# ============================================================================
# Emotion-Based Talking Head Generation -- complete terminal pipeline
#
# One command does everything the 07_emotion_talking_head_inference.ipynb
# notebook does, without Jupyter:
#   1. Create a dedicated virtual environment (venv/) with the proven pinned
#      torch stack + dependencies (no kernel, no restarts)
#   2. Download all required models from Hugging Face (skipped if present)
#   3. Generate the emotion talking-head video(s) via generator/test_emotions.py
#
# Configuration via environment variables, e.g.:
#   EMOTION=all bash run_emotion_talking_head.sh
#   REF_IMAGE=/path/to/face.jpg AUDIO=/path/to/speech.wav EMOTION=angry \
#       bash run_emotion_talking_head.sh
# ============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Locate the repository root (the folder containing this script)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f "generator/FM.py" ] || [ ! -f "renderer/models.py" ]; then
    echo "[error] could not locate generator/ and renderer/ next to this script" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------
REF_IMAGE="${REF_IMAGE:-$SCRIPT_DIR/assets/3robert.jpeg}"
AUDIO="${AUDIO:-$SCRIPT_DIR/assets/audio_3.wav}"
EMOTION="${EMOTION:-happy}"                    # neutral|happy|sad|angry|surprise|fear|all|0-5
CFG_SCALE="${CFG_SCALE:-3.0}"
NFE="${NFE:-10}"
SEED="${SEED:-42}"
CROP="${CROP:-0}"                              # 1 = face-alignment crop (human faces only)
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/emotion_talking_head_outputs}"
EMOTION_LORA_RANK="${EMOTION_LORA_RANK:-32}"   # must match training
EMOTION_LORA_ALPHA="${EMOTION_LORA_ALPHA:-64.0}"
USE_STYLE_LORA="${USE_STYLE_LORA:-1}"          # 0 = disable the style LoRA merge

# Hugging Face sources (same as the notebook)
HF_EMOTION_LORA_REPO="bitwisemind/emotion_lora_adapter_v1"
HF_EMOTION_LORA_FILE="last.ckpt"
HF_STYLE_LORA_REPO="niloy629/hdtf_preprocess"
HF_STYLE_LORA_FILE="live_winner/lora/ditto_blink_lora_withaudio_r64_096_continue_2h_last.ckpt"

CKPT_DIR="$SCRIPT_DIR/checkpoints"
WAV2VEC_DIR="$CKPT_DIR/wav2vec2-base-960h"
RENDERER_CKPT="$CKPT_DIR/renderer.ckpt"
BASE_GENERATOR_CKPT="$CKPT_DIR/generator.ckpt"
STYLE_LORA_CKPT="$CKPT_DIR/$HF_STYLE_LORA_FILE"
EMOTION_LORA_CKPT="$CKPT_DIR/emotion_lora/$HF_EMOTION_LORA_FILE"

VENV_DIR="$SCRIPT_DIR/venv"
VENV_PYTHON="$VENV_DIR/bin/python"

echo "reference image : $REF_IMAGE"
echo "audio           : $AUDIO"
echo "emotion         : $EMOTION"
echo "generation      : cfg=$CFG_SCALE  nfe=$NFE  seed=$SEED  crop=$CROP"
echo "output dir      : $OUTPUT_DIR"
echo

# ---------------------------------------------------------------------------
# 1. ffmpeg check
# ---------------------------------------------------------------------------
command -v ffmpeg >/dev/null 2>&1 || {
    echo "[error] ffmpeg not found on PATH (required for video encoding)." >&2
    exit 1
}

# ---------------------------------------------------------------------------
# 2. Virtual environment
# ---------------------------------------------------------------------------
if [ ! -x "$VENV_PYTHON" ]; then
    echo "[venv] creating virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
    "$VENV_PYTHON" -m pip install -q --upgrade pip setuptools wheel
else
    echo "[venv] reusing existing virtual environment at $VENV_DIR"
fi

# ---------------------------------------------------------------------------
# 3. Pinned CUDA PyTorch (torch 2.8.0 + torchvision 0.23.0 -- the proven stack
#    from the live-pipeline notebooks / live.md; newer torchvision removed
#    torchvision.io.write_video, so do NOT install unpinned latest)
# ---------------------------------------------------------------------------
torch_cuda_ok() {
    "$VENV_PYTHON" -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"
}

if ! torch_cuda_ok; then
    echo "[venv] installing pinned torch 2.8.0 + torchvision 0.23.0 (first run downloads ~3 GB) ..."
    installed=0
    for index_url in \
        "https://download.pytorch.org/whl/cu128" \
        "https://download.pytorch.org/whl/cu126"; do
        echo "[venv] trying $(basename "$index_url") wheels ..."
        if "$VENV_PYTHON" -m pip install -q torch==2.8.0 torchvision==0.23.0 --index-url "$index_url" && torch_cuda_ok; then
            installed=1
            break
        fi
        echo "  wheel index failed, trying next option ..."
    done
    if [ "$installed" -eq 0 ]; then
        "$VENV_PYTHON" -m pip install -q torch==2.8.0 torchvision==0.23.0
    fi
    torch_cuda_ok || { echo "[error] CUDA still unavailable inside the venv after installing PyTorch." >&2; exit 1; }
else
    echo "[venv] CUDA-capable PyTorch already installed -- keeping it (no re-download)"
fi

"$VENV_PYTHON" -c "import torch; print('torch (venv):      ', torch.__version__); print('GPU:               ', torch.cuda.get_device_name(0))"


# ---------------------------------------------------------------------------
# 4. Python dependencies (installed INTO THE VENV)
# ---------------------------------------------------------------------------
MISSING=$("$VENV_PYTHON" - <<'PY'
import importlib.util
mods = ["pytorch_lightning", "torchdiffeq", "einops", "transformers", "librosa",
        "face_alignment", "av", "tensorboard", "timm", "tqdm", "huggingface_hub",
        "yaml", "cv2", "matplotlib"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
print(" ".join(missing))
PY
)

if [ -n "$MISSING" ]; then
    echo "[venv pip] installing missing packages ..."
    "$VENV_PYTHON" -m pip install -q \
        "pytorch-lightning==2.2.1" \
        "torchdiffeq==0.2.5" \
        "einops" \
        "transformers>=4.38,<4.42" \
        "librosa" \
        "face_alignment==1.4.1" \
        "av==12.0.0" \
        "tensorboard" \
        "timm==1.0.9" \
        "tqdm" \
        "huggingface_hub" \
        "pyyaml" \
        "opencv-python<4.10.0.0" \
        "matplotlib<3.9.0"
else
    echo "[venv] all required packages already installed"
fi

# ---------------------------------------------------------------------------
# 5. NumPy / SciPy pins (cv2 & friends are built against NumPy 1.x;
#    scipy<1.13 is the last NumPy-1.x-compatible line)
# ---------------------------------------------------------------------------
"$VENV_PYTHON" -m pip install -q "numpy<2.0.0" "scipy<1.13"
"$VENV_PYTHON" -c "import importlib.metadata as m; print('numpy (venv):       ', m.version('numpy')); print('scipy (venv):       ', m.version('scipy'))"

# ---------------------------------------------------------------------------
# 6. Verify the full import stack inside the venv (with auto-repair)
# ---------------------------------------------------------------------------
STACK_ERR="$(mktemp)"
STACK_CODE="import numpy, cv2, scipy.integrate, librosa, matplotlib.pyplot, face_alignment, torchdiffeq"
if ! "$VENV_PYTHON" -c "$STACK_CODE" 2>"$STACK_ERR"; then
    echo "[venv] import stack check failed -- attempting targeted repairs ..."
    grep -q "numba" "$STACK_ERR" && "$VENV_PYTHON" -m pip install -q "numba<0.62" || true
    grep -q "scipy" "$STACK_ERR" && "$VENV_PYTHON" -m pip install -q "scipy<1.13" || true
    if grep -q -e "cv2" -e "numpy.core.multiarray" "$STACK_ERR"; then
        "$VENV_PYTHON" -m pip install -q --force-reinstall --no-deps "opencv-python<4.10.0.0" || true
    fi
    if ! "$VENV_PYTHON" -c "$STACK_CODE" 2>"$STACK_ERR"; then
        echo "[error] venv import stack still failing:" >&2
        tail -n 5 "$STACK_ERR" >&2
        rm -f "$STACK_ERR"
        exit 1
    fi
fi
rm -f "$STACK_ERR"
echo "[venv] import stack check: OK (numpy, cv2, scipy.integrate, librosa, matplotlib.pyplot, face_alignment, torchdiffeq)"
echo "Environment OK (venv)"


# ---------------------------------------------------------------------------
# 7. Download all required models (skipped if already present)
# ---------------------------------------------------------------------------
echo
echo "[models] checking / downloading required models ..."
export CKPT_DIR WAV2VEC_DIR RENDERER_CKPT BASE_GENERATOR_CKPT STYLE_LORA_CKPT \
       EMOTION_LORA_CKPT USE_STYLE_LORA HF_STYLE_LORA_REPO HF_STYLE_LORA_FILE \
       HF_EMOTION_LORA_REPO HF_EMOTION_LORA_FILE

"$VENV_PYTHON" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

ckpt_dir = Path(os.environ["CKPT_DIR"])
ckpt_dir.mkdir(parents=True, exist_ok=True)

def download(repo_id, filename, local_dir, repo_type="model"):
    dest = Path(local_dir) / filename
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name}")
        return dest
    print(f"[download] {filename} from {repo_id} ...")
    hf_hub_download(repo_id=repo_id, filename=filename,
                    local_dir=str(local_dir), repo_type=repo_type)
    print(f"    ready: {dest}")
    return dest

# 1) Base IMTalker models (renderer, generator, config)
for fname in ["config.yaml", "renderer.ckpt", "generator.ckpt"]:
    download("cbsjtu01/IMTalker", fname, ckpt_dir)

# 2) wav2vec2-base-960h audio encoder
for rel in [
    "wav2vec2-base-960h/config.json",
    "wav2vec2-base-960h/pytorch_model.bin",
    "wav2vec2-base-960h/preprocessor_config.json",
    "wav2vec2-base-960h/feature_extractor_config.json",
]:
    download("cbsjtu01/IMTalker", rel, ckpt_dir)

# 3) Style LoRA (dataset repository)
if os.environ["USE_STYLE_LORA"] == "1":
    download(os.environ["HF_STYLE_LORA_REPO"], os.environ["HF_STYLE_LORA_FILE"],
             ckpt_dir, repo_type="dataset")

# 4) Trained emotion LoRA adapter
download(os.environ["HF_EMOTION_LORA_REPO"], os.environ["HF_EMOTION_LORA_FILE"],
         Path(os.environ["EMOTION_LORA_CKPT"]).parent)

# Final verification
required = [Path(os.environ["RENDERER_CKPT"]),
            Path(os.environ["BASE_GENERATOR_CKPT"]),
            Path(os.environ["WAV2VEC_DIR"]) / "pytorch_model.bin",
            Path(os.environ["EMOTION_LORA_CKPT"])]
if os.environ["USE_STYLE_LORA"] == "1":
    required.append(Path(os.environ["STYLE_LORA_CKPT"]))
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit(f"[error] missing required files: {missing}")
print("[models] all models ready")
PY

# ---------------------------------------------------------------------------
# 8. Generate the emotion talking-head video(s)
# ---------------------------------------------------------------------------
EMOTION_ARGS=$("$VENV_PYTHON" - "$EMOTION" <<'PY'
import sys
spec = sys.argv[1].strip().lower()
names = {0: "neutral", 1: "happy", 2: "sad", 3: "angry", 4: "surprise", 5: "fear"}
ids = {v: k for k, v in names.items()}
if spec == "all":
    print(" ".join(names.values()))
elif spec.isdigit() and int(spec) in names:
    print(names[int(spec)])
elif spec in ids:
    print(spec)
else:
    raise SystemExit(f"[error] unknown emotion {spec!r}. Valid: {list(ids)} or 'all'")
PY
)

CMD=(
    "$VENV_PYTHON" "generator/test_emotions.py"
    "--checkpoint" "$EMOTION_LORA_CKPT"
    "--renderer_ckpt" "$RENDERER_CKPT"
    "--base_generator_ckpt" "$BASE_GENERATOR_CKPT"
    "--wav2vec_model_path" "$WAV2VEC_DIR"
    "--ref_img" "$REF_IMAGE"
    "--audio" "$AUDIO"
    "--out_dir" "$OUTPUT_DIR"
    "--cfg" "$CFG_SCALE"
    "--nfe" "$NFE"
    "--seed" "$SEED"
    "--lora_rank" "$EMOTION_LORA_RANK"
    "--lora_alpha" "$EMOTION_LORA_ALPHA"
    "--emotions" $EMOTION_ARGS
)
if [ "$USE_STYLE_LORA" = "1" ]; then
    CMD+=("--style_lora_ckpt" "$STYLE_LORA_CKPT")
fi
if [ "$CROP" = "1" ]; then
    CMD+=("--crop")
fi

echo
echo "[generate] running: ${CMD[*]}"
echo
"${CMD[@]}"

echo
echo "[done] video(s) saved in: $OUTPUT_DIR"
ls -1 "$OUTPUT_DIR"/*.mp4 2>/dev/null || true
