# Environment Setup

## Quick install (conda + pip)

```bash
conda env create -f environment.yml
conda activate enlight
```

Or manually:

```bash
conda create -n enlight python=3.10 -y
conda activate enlight
pip install --upgrade pip
pip install -e .
pip install open-clip-torch==2.23.0
pip install "accelerate>=1.1.0"
pip install "peft==0.13.2"
pip install "transformers>=4.45.0" "protobuf==3.20.3"
```

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

### protobuf

`transformers>=4.45` pulls in a newer protobuf that conflicts with some
gRPC-based tools. Pin it explicitly:

```bash
pip install "protobuf==3.20.3"
```

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
