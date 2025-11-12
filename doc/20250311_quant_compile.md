# 量子化およびコンパイル作業レポート（2025-03-11）

## 対象
- モデル: `deimv2_{variant}_stage1.onnx` (`variant ∈ {atto, femto, pico, n}`)
- HAR: `dst_models/deimv2_{variant}_stage1.har`
- キャリブレーションデータ: `/local/shared_with_docker/calib/calib_set_{a,f,p,n}/`  
  - 形式: `float32` の `*.npy`（形状 `(1, 3, H, W)`）。スクリプト側で NHWC に変換。

## 追加スクリプト
- `tools/quantize_compile_stage1.py`
  - Calib データ読込 → NHWC 変換 → `runner.optimize()`。
  - 量子化後 HAR (`*_quantized.har`) と HEF (`*.hef`) を自動生成する流れを共通化。
  - `--max-samples`（既定 128）でキャリブ枚数制限を指定。

## 実行手順（コマンド例）
```bash
cd /local/shared_with_docker/halio8-model
export HOME=/local/shared_with_docker/halio8-model
export HAILO_CLIENT_LOGS_ENABLED=false
export HAILO_SDK_LOG_DIR=/local/shared_with_docker/halio8-model/.hailo_logs
export HAILO_DISABLE_MO_SUB_PROCESS=1
python tools/quantize_compile_stage1.py --max-samples 64
```
- 上記の環境変数設定により、ログ出力権限エラーや GPU 判定サブプロセスの PermissionError を回避。

## 実行結果（Codex CLI 環境）
| Variant | 量子化 (`optimize`) | HEF (`runner.compile`) | 備考 |
| --- | --- | --- | --- |
| atto | ✅ `dst_models/deimv2_atto_stage1_quantized.har` 生成 | ❌ `HailoToolsException: Couldn't launch Hailo tool securely` | ループバックソケット禁止のためコンパイル失敗 |
| femto/pico/n | ❌（atto 処理途中で停止） | - | スクリプト上は対応済み。環境を変えて再実行すれば生成可 |

### 失敗の詳細
- `runner.compile()` 内で呼ばれる `hailo_tools_runner.establish_server_client_channels()` が、`socket.connect()` 時に `Operation not permitted` を返し、`HailoToolsException` で終了。
- Codex CLI 環境では TCP/ローカルソケット作成が禁止されているため、Hailo コンパイラのセキュア通信が確立できない。
- 同時に GPU も見えず、SDK が「GPU なし」と判断して最適化レベルを 0 に強制ダウン。これは今回の検証環境依存の挙動。

##今後の対応
1. 公式 SDK コンテナをホスト PC 上で起動（`hailo_ai_sw_suite_docker_run.sh` をそのまま実行）し、同じスクリプトをコンテナ内で再実行する。  
   - この構成なら GPU やローカルソケットが使用できるため、量子化～コンパイルが完走する。
2. `dst_models/deimv2_{variant}_stage1_quantized.har` が生成済みであれば、他環境で `runner.compile()` のみ再実行することも可能。
3. キャリブデータが 1024 枚未満・GPU 無しの場合、SDK は最適化レベルを 0 に切り替えるため、精度面で影響がある場合はキャリブ枚数の追加か GPU を用意する。

##補足
- Stage1 HAR の出力は `hailo_stage1_feat` 固定。Stage2 は `hailo_stage1_feat` + `images` を受け取るため、ホスト側で Stage2 処理を完結できる。
- スクリプトではキャリブデータの値域チェックを行っていないため、Hailo の推奨どおり前処理（正規化）をデバイス側で行う場合は `.hailo` スクリプトに `normalization` コマンドを追加することを推奨。
