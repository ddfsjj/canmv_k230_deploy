# K230 Model Assets

这个目录可以存放多版历史 KModel、ONNX、scaler 和 meta 文件，但正式部署不按整个目录打包。

正式部署只打包 `raw_cnn_k230/configs/runtime.json` 当前引用到的资产：

```text
models[].assets.kmodel
models[].assets.scaler_json
```

当前默认 profile：

```text
cnn_tcn_uart_two_inputs
```

当前 `runtime.json` 引用的板端资产：

```text
model/cnn-tcn/cnn_tcn_20260629_084904_u8_u8_kld_512.kmodel
model/cnn-tcn/cnn_tcn_20260629_084904_u8_u8_kld_512_scaler.json
```

上线前以部署包 manifest 为准：

```text
deploy_pkg/raw_cnn_k230/DEPLOY_MANIFEST.json
```

不要根据 `raw_cnn_k230/model/` 目录里存在的全部文件判断本次会部署哪些模型。

## 换模型流程

1. 把新的 `.kmodel` 和 scaler `.json` 放到合适的模型子目录。
2. 修改 `raw_cnn_k230/configs/runtime.json`：

```text
models[].assets.kmodel
models[].assets.scaler_json
```

3. 运行：

```bash
python scripts/validate_runtime_config.py
python scripts/make_deploy_package.py --clean
python scripts/verify_deploy_package.py
```

4. 检查 `DEPLOY_MANIFEST.json` 里记录的 KModel 和 scaler 是否是本次要上线的版本。

## 历史资产

没有被 `runtime.json` 引用的 `.kmodel/.onnx/.json/.pkl/.pth/.meta.json` 只作为历史实验或回滚参考。

等现场版本稳定后，可以再把不用的历史资产移到 archive 目录或删除。
