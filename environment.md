# Environment Setup

## Quick install (conda + pip)

Run from the **project root** (so that `-e .` resolves correctly):

```bash
conda env create -f environment.yml
conda activate enlight
pip install -e .          # editable install of the enlight package
```

Or manually:

```bash
conda create -n enlight python=3.10 -y
conda activate enlight
pip install --upgrade pip
pip install -e .
pip install open-clip-torch==2.23.0
pip install "transformers>=4.45.0,<5.0" "tokenizers>=0.19,<0.21" "huggingface_hub>=0.23.2,<1.0"
pip install "accelerate>=1.1.0" "peft==0.13.2" "protobuf==3.20.3"
```

> **Version constraints explained:**
> - `transformers<5.0` — transformers 5.x requires `huggingface_hub>=1.3`, which conflicts with the rest of the stack
> - `tokenizers>=0.19,<0.21` — transformers 4.45+ dropped support for tokenizers 0.15.x
> - `huggingface_hub>=0.23.2,<1.0` — explicit upper bound prevents pip from pulling in a 5.x-compatible version

---

## Training dependencies

```bash
pip install -e ".[train]"   # deepspeed==0.12.6, ninja, wandb
```

### flash-attn (requires matching CUDA toolkit)

`flash-attn` must be compiled against your system's CUDA toolkit, not just the PyTorch CUDA runtime.

```bash
# Check your CUDA version first
nvcc --version          # should be 11.8 or 12.x
nvidia-smi              # confirm driver supports that CUDA version

# Install (no-build-isolation lets it link against your system CUDA)
pip install flash-attn==2.3.3 --no-build-isolation
```

If `nvcc` is missing, load the module (on SLURM/HPC clusters):

```bash
module load cuda/11.8     # or cuda/12.1 depending on your cluster
pip install flash-attn==2.3.3 --no-build-isolation
```

If compilation is slow (10–30 min), use a prebuilt wheel from
https://github.com/Dao-AILab/flash-attention/releases — pick the wheel
matching your Python, PyTorch, and CUDA versions, then:

```bash
pip install flash_attn-2.3.3+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

---

## Common version conflicts

### bitsandbytes on older GPUs (compute < 7.5)

```bash
pip install bitsandbytes --prefer-binary --extra-index-url \
    https://huggingface.github.io/bitsandbytes-foundation/whl/cu118
```

### gradio version lock

If you see `pydantic` conflicts with `gradio==4.16.0`, pin pydantic v1:

```bash
pip install "pydantic<2.0"
```

---

## Verify installation

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import open_clip; print(open_clip.__version__)"
python -c "import transformers; print(transformers.__version__)"
```
