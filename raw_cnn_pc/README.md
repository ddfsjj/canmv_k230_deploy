# Raw CNN PC Folder

This folder is for PC-side work in VSCode:
- run `.pth` inference
- convert `.pth` to `.kmodel`
- auto-export deploy artifacts into `../raw_cnn_k230`

Detailed model replacement and deployment guide:
- `MODEL_CHANGE_TO_K230_GUIDE.md`

## 1) Install dependencies

```bash
pip install -r requirements.txt
```

For kmodel export:

```bash
pip install -r requirements_k230_host.txt
```

## 2) PC inference

```bash
python infer.py --config infer_config.json --max_samples 10 --output predictions_pc_quick.csv
```

## 3) Export to K230 folder

Use either command:

```bash
python build_kmodel.py
```

or

```bash
build_to_k230.bat
```

Generated files are written to:
- `../raw_cnn_k230/model/model_run020.onnx`
- `../raw_cnn_k230/model/model_run020.kmodel`
- `../raw_cnn_k230/model/scaler_run020.json`
- `../raw_cnn_k230/model/calibration_input.npy`
