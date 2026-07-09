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
