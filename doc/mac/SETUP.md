## install PIXI SETUP

```bash
pixi lock
pixi install
```

2. Verify MPS:

```bash
pixi run python -c \
  "import torch; print(torch.__version__); print('MPS:', torch.backends.mps.is_available())"
```

3. Build the Mac Cython extension:

```bash
pixi run python scripts/setup_taas_cython.py build_ext --inplace
```

###

```bash
git pull
pixi install --locked
```

```bash
pixi run python -c \
  "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

pixi run python scripts/setup_taas_cython.py build_ext --inplace
```
