# AGENTS.md — エージェント運用メモ

## デプロイ・実行環境の前提

- **実験・ライブサーバーはGPUサーバー上で起動するのが基本運用**。
  ローカルMacは開発・編集用であり、実験本体はGPUサーバーで動かす。
- ライブ可視化はGPUサーバー上の `live_server.py --ngrok` で配信し、
  ngrokの公開URL経由でブラウザからアクセスする（ケース2運用）。
- `visualize_game.html` はページ読み込み時に `/api/files` から
  ファイル一覧を自動取得し、ドロップダウンでファイルを選択する。
  ファイルパスの手動入力は不要。ngrok無料プラン対応のため
  `ngrok-skip-browser-warning` ヘッダーを送信する。
- ライブモードは HTTP Range リクエストで差分取得し、
  指数バックオフ付きの無制限自動リトライで接続を維持する。

## Git運用ルール

- リモート: `git@github-neipia:neipia271828/hivc-d-verification.git`
  （SSHエイリアス `github-neipia` = neipia271828 アカウント）
- ローカルで変更したファイル（`scripts/visualize_game.html` 等）は
  **コミット＆プッシュしないとGPUサーバーに反映されない**。
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

> **注意**: GPUサーバー上の `projects/hivc-d-verification` は git 管理されて
> いない場合がある（`.git` が無い ＝ `git pull` 不可）。その場合は下記の
> `deploy_viz_to_gpu.sh` で可視化コードを scp 配備する。

## 可視化コードのGPU配備（git不可な環境向け）

`scripts/deploy_viz_to_gpu.sh` が「バックアップ → scp転送 → 旧サーバー/ngrok
停止 → 新版で再起動 → 検証」を1コマンドで実行する（Macから実行）。

```bash
# 転送＋再起動＋ローカル検証
scripts/deploy_viz_to_gpu.sh

# 公開URLでも検証（/api/status と /visualize の新版マーカーを確認）
PUBLIC_URL=https://<ngrok-url> scripts/deploy_viz_to_gpu.sh

# 転送のみ（サーバー無停止）
scripts/deploy_viz_to_gpu.sh --no-restart

# ヘルプ / 全オプション
scripts/deploy_viz_to_gpu.sh --help
```

接続情報・対象ファイルは環境変数で上書きできる（デフォルトは現行GPU設定）:

| 変数 | 既定値 | 説明 |
|------|--------|------|
| `SSH_USER` / `SSH_HOST` / `SSH_PORT` | `student222` / `172.16.51.202` / `2222` | 接続先 |
| `SSH_KEY` | `~/.ssh/id_ed25519_kmc_gpu` | 秘密鍵 |
| `REMOTE_REPO` | `projects/hivc-d-verification` | GPU上のリポジトリ（`$HOME` 相対） |
| `PORT` | `8765` | live_server のポート |
| `LIVE_JSONL` | `hivc_sim/results/turn_game/experiment/stream.jsonl` | `--file` で固定するJSONL |
| `FILES` | `scripts/live_server.py scripts/visualize_game.html` | 転送するファイル |
| `PUBLIC_URL` | （未設定） | 指定時、ngrok公開URL経由でも検証 |

内部的にリモート処理は `ssh ... bash -s`（stdin実行）で行う。これはリモート
シェルのコマンドラインを `bash -s` のみにして、`pkill -f "scripts/live_server.py"`
が自セッションにマッチして落ちる事故（exit 255）を防ぐため。上書き前に
GPU側 `/tmp/<name>.bak.<timestamp>` へ必ずバックアップする。

## ライブ可視化の起動手順（GPUサーバー側）

```bash
# サーバーを起動（--file 指定不要、結果ディレクトリを自動スキャン）
python3 scripts/live_server.py --port 8765 --ngrok

# 別ターミナルで実験を起動
python3 scripts/qwen_two_agent_experiment.py \
  --config configs/experiment.yaml \
  --live-jsonl hivc_sim/results/turn_game/experiment/stream.jsonl
```

ブラウザで表示された `https://<ngrok-url>/visualize` を開くと、
ファイル一覧がドロップダウン表示され、最新のJSONLが自動選択される。
ライブ/リプレイをトグルボタンで切り替え可能。

### サーバーAPI エンドポイント

| パス | 説明 |
|------|------|
| `/visualize` | ビジュアライザーHTML |
| `/api/files` | データファイル一覧（JSONL/CSV、メタデータ付き） |
| `/api/file?path=<rel>` | ファイル配信（Range リクエスト対応） |
| `/api/status` | ヘルスチェック |
| `/stream.jsonl` | 後方互換: デフォルトJSONL配信 |
