# Raw + CNN Package (PC + K230)

This folder now contains two runtimes:
- `infer.py`: PC `.pth` inference (install `requirements.txt` first).
- `run_k230_infer.py`: K230 `.kmodel` inference script (board-side).

## 1) PC `.pth` inference

```bash
pip install -r requirements.txt
python infer.py
```

Quick smoke test (10 samples):

```bash
python infer.py --max_samples 10 --output predictions_pc_quick.csv
```

VSCode (workspace root `canmv_k230_deploy`):
- Run task: `Install Raw CNN Dependencies`
- Start config: `Raw CNN Quick Test (10 samples)` in `.vscode/launch.json`

## 2) Build K230 deploy assets

Use host build script in this folder:

```bash
pip install -r requirements_k230_host.txt
```

If you see `Failed to get hostfxr path`, install .NET runtime:

```bash
winget install --id Microsoft.DotNet.Runtime.7 -e --silent --accept-package-agreements --accept-source-agreements
```

Then build:

```bash
python build_kmodel.py --skip_compile
```

This generates:
- `model/model_run020.onnx`
- `model/scaler_run020.json`
- `model/calibration_input.npy`

Then, with `nncase` installed:

```bash
python build_kmodel.py
```

This additionally generates:
- `model/model_run020.kmodel`

## 3) Run on K230

Copy this folder (or at least `run_k230_infer.py`, `k230_config.json`, `model/*.kmodel`, `model/*.json`, `test_data/*.csv`) to board storage, then run:

```python
python run_k230_infer.py
```

Board script outputs:
- predictions
- model inference time
- total pipeline time
- `predictions_k230.csv`
