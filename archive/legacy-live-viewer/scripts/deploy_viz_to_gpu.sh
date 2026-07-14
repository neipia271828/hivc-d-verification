#!/usr/bin/env bash
#
# deploy_viz_to_gpu.sh — ローカルの可視化コードをGPUサーバーへ配備して再起動する。
#
# GPU側の projects/hivc-d-verification は git 管理されていない（.git なし）ため
# `git pull` が使えない。本スクリプトは修正済みファイルを scp 転送し、
# 旧 live_server.py + ngrok を停止して新版で起動し直し、動作を検証する。
#
# 使い方:
#   scripts/deploy_viz_to_gpu.sh                 # デフォルト設定で配備＋再起動
#   scripts/deploy_viz_to_gpu.sh --no-restart    # 転送のみ（再起動しない）
#   PUBLIC_URL=https://<ngrok>.ngrok-free.dev scripts/deploy_viz_to_gpu.sh   # 公開URLも検証
#
# 主要な設定は環境変数で上書き可能:
#   SSH_HOST / SSH_USER / SSH_PORT / SSH_KEY   … 接続先
#   REMOTE_REPO   … GPU上のリポジトリパス
#   PORT          … live_server のポート
#   LIVE_JSONL    … --file で固定するデフォルトJSONL（リポジトリ相対）
#   FILES         … 転送するファイル（リポジトリ相対、スペース区切り）
#   PUBLIC_URL    … 指定すると ngrok 公開URL経由でも検証する
#
set -euo pipefail

# ---- 設定（環境変数で上書き可） --------------------------------------------
SSH_USER="${SSH_USER:-student222}"
SSH_HOST="${SSH_HOST:-172.16.51.202}"
SSH_PORT="${SSH_PORT:-2222}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519_kmc_gpu}"
REMOTE_REPO="${REMOTE_REPO:-projects/hivc-d-verification}"   # $HOME からの相対
PORT="${PORT:-8765}"
LIVE_JSONL="${LIVE_JSONL:-hivc_sim/results/turn_game/experiment/stream.jsonl}"
VENV="${VENV:-.venv}"
NGROK_DIR="${NGROK_DIR:-\$HOME/bin}"          # GPU上でngrokがあるディレクトリ（PATHに追加）
LOG="${LOG:-/tmp/hivc_server.log}"
FILES="${FILES:-scripts/live_server.py scripts/visualize_game.html}"
PUBLIC_URL="${PUBLIC_URL:-}"

RESTART=1
for arg in "$@"; do
  case "$arg" in
    --no-restart) RESTART=0 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "不明な引数: $arg" >&2; exit 2 ;;
  esac
done

LOCAL_REPO="$(cd "$(dirname "$0")/.." && pwd)"
SSH_TARGET="${SSH_USER}@${SSH_HOST}"
SSH_OPTS=(-p "$SSH_PORT" -i "$SSH_KEY" -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new)
SCP_OPTS=(-P "$SSH_PORT" -i "$SSH_KEY" -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new)

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# ---- 1. 接続確認 -----------------------------------------------------------
say "GPUへ接続確認 (${SSH_TARGET}:${SSH_PORT})"
ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "cd ~/$REMOTE_REPO && pwd" \
  || { echo "接続またはリポジトリパスの確認に失敗しました" >&2; exit 1; }

# ---- 2. バックアップ -------------------------------------------------------
TS="$(date +%Y%m%d_%H%M%S)"
say "GPU側の対象ファイルをバックアップ (/tmp/*.bak.$TS)"
ssh "${SSH_OPTS[@]}" "$SSH_TARGET" bash -s -- "$REMOTE_REPO" "$TS" $FILES <<'REMOTE'
set -e
repo="$1"; ts="$2"; shift 2
cd "$HOME/$repo"
for f in "$@"; do
  if [ -f "$f" ]; then
    cp "$f" "/tmp/$(basename "$f").bak.$ts"
    echo "  backup: $f -> /tmp/$(basename "$f").bak.$ts"
  else
    echo "  (skip, 未存在): $f"
  fi
done
REMOTE

# ---- 3. 転送 ---------------------------------------------------------------
say "修正版ファイルを転送"
for f in $FILES; do
  echo "  scp $f"
  scp "${SCP_OPTS[@]}" "$LOCAL_REPO/$f" "$SSH_TARGET:$REMOTE_REPO/$f"
done

if [ "$RESTART" -eq 0 ]; then
  say "--no-restart 指定のため再起動をスキップしました"
  exit 0
fi

# ---- 4. 停止 → 起動 → ローカル検証 ----------------------------------------
# 注意: remote は `bash -s`（stdin実行）にしているため、リモートシェルの
# コマンドライン自体は "bash -s ..." となり、pkill -f "scripts/live_server.py"
# が自分自身にマッチして自滅する事故を防いでいる。
say "旧サーバー/ngrokを停止し、新版で再起動"
ssh "${SSH_OPTS[@]}" "$SSH_TARGET" bash -s -- \
    "$REMOTE_REPO" "$PORT" "$LIVE_JSONL" "$VENV" "$NGROK_DIR" "$LOG" <<'REMOTE'
set +e
repo="$1"; port="$2"; live_jsonl="$3"; venv="$4"; ngrok_dir="$5"; log="$6"
cd "$HOME/$repo"

echo "--- 停止 ---"
pkill -f "scripts/live_server.py" && echo "killed live_server" || echo "no live_server proc"
pkill -f "ngrok http $port"       && echo "killed ngrok"       || echo "no ngrok proc"
sleep 3

echo "--- 起動 ---"
# shellcheck disable=SC1090
source "$venv/bin/activate"
eval "export PATH=$ngrok_dir:\$PATH"
echo "python: $(command -v python3)"
echo "ngrok:  $(command -v ngrok || echo 'NOT FOUND (ngrok未検出)')"
nohup python3 scripts/live_server.py \
  --file "$live_jsonl" --port "$port" --ngrok > "$log" 2>&1 &
disown
sleep 10

echo "--- ローカル /api/status ---"
curl -s -m 5 "http://localhost:$port/api/status" && echo || echo "(ローカルstatus取得失敗)"
echo "--- プロセス ---"
ps -o pid,cmd -C python3,ngrok 2>/dev/null | grep -E "live_server|ngrok http" || echo "(プロセスが見当たりません!)"
echo "--- ログ末尾 ---"
tail -n 15 "$log" 2>/dev/null || true
REMOTE

# ---- 5. 公開URL検証（任意） ------------------------------------------------
if [ -n "$PUBLIC_URL" ]; then
  say "公開URL経由で検証: $PUBLIC_URL"
  H='ngrok-skip-browser-warning: true'
  code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' -H "$H" "$PUBLIC_URL/api/status" || echo "000")
  echo "  /api/status -> HTTP $code (200なら新版稼働)"
  n=$(curl -s -m 10 -H "$H" "$PUBLIC_URL/visualize" | grep -c '/api/files' || true)
  echo "  /visualize の /api/files 参照: $n 箇所 (>=1 なら新版HTML)"
fi

say "完了"
