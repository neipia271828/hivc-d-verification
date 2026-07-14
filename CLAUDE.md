# AGENTS.md — エージェント運用メモ

## デプロイ・実行環境の前提

- **実験はGPUサーバーで実行し、結果はローカルMacに取得して閲覧するのが基本運用**。
- ローカルMacは開発・編集・実験結果のプレビュー用。実験本体はGPUサーバーで動かす。
- 可視化はGPUサーバー上のHTTPサーバーやngrokを使わず、ローカルで `scripts/local_preview.py` を起動する。
- `local_preview.py` は `127.0.0.1` のみで待ち受け、外部公開は行わない。

## Git運用ルール

- リモート: `git@github-neipia:neipia271828/hivc-d-verification.git`
  （SSHエイリアス `github-neipia` = neipia271828 アカウント）
- ローカルで変更したファイルは **コミット＆プッシュしないとGPUサーバーに反映されない**。
- GPUサーバー側は `git pull origin main` で最新コードを取得する。
- HTTPSリモート（`https://github.com/neipia271828/...`）に戻さないこと。
  `suzukihinata-dev` アカウントではプッシュ権限がない。

## GPUサーバーへのコード同期手順

```bash
# ローカルMac側
git add -A && git commit -m "..." && git push origin main

# GPUサーバー側（SSH接続後）
cd /path/to/hivc-d-verification
git pull origin main
```

## 実験・結果確認の一連の流れ

```bash
# 1. GPUサーバーで実験を起動
python3 scripts/gpu_run.py

# 2. 完了したらローカルMacにログを取得
python3 scripts/download_gpu_logs.py

# 3. ローカルでプレビュー（ネットワーク不要）
python3 scripts/local_preview.py

# 4. ブラウザで http://127.0.0.1:8765/ を開く
```

## ローカル実験ログ取得

`scripts/download_gpu_logs.py` は `configs/gpu_server.yaml` のSSH接続情報を使い、
GPUサーバー上の実験出力ディレクトリを `hivc_sim/results/turn_game/downloads/<run-id>/` へ `rsync` する。

```bash
# 既定の実験出力を取得
python3 scripts/download_gpu_logs.py

# run名を指定
python3 scripts/download_gpu_logs.py --run-id 2026-07-13-run1

# リモート出力ディレクトリを変更
python3 scripts/download_gpu_logs.py --remote-output-dir hivc_sim/results/turn_game/experiment

# ドライラン
python3 scripts/download_gpu_logs.py --dry-run
```

取得時に `manifest.json` が生成され、取得元GPUホスト・リモートディレクトリ・取得日時・ファイル一覧を記録する。

## ローカルプレビュー

`scripts/local_preview.py` を起動すると、取得済みのrunディレクトリをスキャンし、
ブラウザで条件・ゲーム・ターンを選択して再生できる。

```bash
# 既定ポート（8765）
python3 scripts/local_preview.py

# ポート変更
python3 scripts/local_preview.py --port 8080
```

- 待受ホストは既定で `127.0.0.1`（`--host` で変更可能だが、公開バインドは推奨しない）。
- `/api/runs` でrun一覧を取得する。
- `/api/runs/<run-id>/file/<filename>` でCSVファイルを配信する。
- ローカルHTMLは `scripts/local_preview.html` 。

## 旧ライブビューアーの扱い

旧ライブビューアー関連ファイル（`live_server.py`、`visualize_game.html`、`deploy_viz_to_gpu.sh`）は
`archive/legacy-live-viewer/` にアーカイブされている。通常運用では使わない。

## 一回開発が終わる毎に実装内容をdevelopment-log.mdに追記していくこと
