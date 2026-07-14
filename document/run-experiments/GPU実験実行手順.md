# GPU実験CLIの使い方

HIVC-Dの実験は、ローカルMacのリポジトリルートから `uv run` の4コマンドで操作する。
実験本体はGPUサーバーで実行し、完了したログをMacへ取得してローカルGUIで閲覧する。

GPU上でHTTPサーバーやngrokを起動する必要はない。GUIはMacの `127.0.0.1` のみに公開される。

## 基本フロー

リポジトリルートで次の順に実行する。

```bash
# 1. ローカルのコードをcommit・pushし、GPUへ同期
uv run sync

# 2. GPUで実験を開始
uv run experiment

# 3. 実験完了後、ログをMacへ取得
uv run download

# 4. MacでGUIビジュアライザーを起動
uv run visualize
```

`uv run experiment` の既定値は `--conditions hivc_d --games 1` で、1エピソードだけ実行する。

## 1. コードをGPUへ同期する

```bash
uv run sync
```

未コミット変更がある場合、次の処理が自動実行される。

1. `git add -A`
2. `git commit`
3. `git push origin main`
4. GPUサーバーで `git pull --ff-only origin main`
5. MacとGPUのHEADが一致することを確認

自動commitのメッセージを指定する場合:

```bash
uv run sync --message "fix: update experiment workflow"
```

未コミット変更を含めず、現在のHEADだけを同期する場合:

```bash
uv run sync --allow-dirty
```

実際には変更せず、実行予定だけを確認する場合:

```bash
uv run sync --dry-run
```

コードや設定を変更した後は、必ず再度 `uv run sync` を実行してから実験を開始する。

## 2. 実験を開始・確認する

### 1エピソードを実行する

```bash
uv run experiment
```

run IDは `episode-YYYYMMDD-HHMMSS` 形式で自動生成される。実験はGPU上でバックグラウンド実行されるため、開始後にSSH接続を維持する必要はない。

### 条件、ゲーム数、seedを指定する

```bash
uv run experiment \
  --conditions control consulting hivc_d \
  --games 30 \
  --seed 42
```

run IDを固定する場合:

```bash
uv run experiment --run-id episode-reproduction-01
```

同じrun IDは再利用できない。再実行時は別のrun IDを指定するか、自動生成を使う。

### 状態を確認する

```bash
uv run experiment --status
```

直前に開始したrunについて、次のいずれかが表示される。

- `running pid=...`: 実行中
- `completed exit_code=0`: 正常終了
- `completed exit_code=...`: エラー終了
- `stopped without exit_code`: 停止または異常終了

特定のrunを確認する場合:

```bash
uv run experiment --status --run-id episode-20260714-120000
```

### ログを追跡する

```bash
uv run experiment --logs
```

`tail -f` でログを表示する。終了するには `Ctrl+C` を押す。特定のrunは `--run-id` で指定できる。

### 実験を停止する

```bash
uv run experiment --stop
```

停止対象を明示する場合:

```bash
uv run experiment --stop --run-id episode-20260714-120000
```

実験は同時に1件だけ起動できる。`ERROR: 別の実験が実行中です: <run-id> (pid=...)` が表示された場合は、表示されたrunを `--status` または `--logs` で確認し、必要な場合だけ `--stop` する。

### 実行コマンドだけを確認する

```bash
uv run experiment --dry-run
```

## 3. 完了ログをMacへ取得する

```bash
uv run download
```

直前に開始したrun、またはGPU上の最新runを取得する。実験が未完了、もしくは終了コードが0以外の場合は取得せず停止する。

特定のrunを取得する場合:

```bash
uv run download --run-id episode-20260714-120000
```

同名のローカルディレクトリを置き換える場合:

```bash
uv run download --run-id episode-20260714-120000 --overwrite
```

ログは次の場所へ保存される。

```text
hivc_sim/results/turn_game/downloads/<run-id>/
```

各runには取得元、取得時刻、ファイル一覧を記録した `manifest.json` も生成される。

## 4. ローカルGUIで可視化する

```bash
uv run visualize
```

ブラウザが自動的に開き、次のURLで取得済みrunを閲覧できる。

```text
http://127.0.0.1:8765/
```

GUIを終了するには、起動したターミナルで `Ctrl+C` を押す。

ポートを変更する場合:

```bash
uv run visualize --port 8080
```

ブラウザを自動で開かない場合:

```bash
uv run visualize --no-open
```

## エピソードをもう一度実行する

コードを変更した場合:

```bash
uv run sync --message "fix: describe the change"
uv run experiment
```

コード変更がなく、別runとして同じ設定を再実行する場合:

```bash
uv run experiment
```

完了後:

```bash
uv run experiment --status
uv run download
uv run visualize
```

## 困ったとき

各コマンドの全オプションは `--help` で確認できる。

```bash
uv run sync --help
uv run experiment --help
uv run download --help
uv run visualize --help
```

- `先に uv run sync を完了してください`: Macで `uv run sync` を実行してから再試行する。
- `JSONスキーマ修正版が未同期です`: ローカルの修正版を `uv run sync` でGPUへ反映する。
- `別の実験が実行中です`: 表示されたrun IDを `uv run experiment --status --run-id <run-id>` で確認する。
- `実験はまだ完了していません`: `uv run experiment --status` で完了を待つ。
- `実験が失敗しています`: `uv run experiment --logs` でエラーを確認する。
- SSH agentの鍵エラー: `ssh-add ~/.ssh/id_ed25519_neipia` を実行してから `uv run sync` を再試行する。

旧 `live_server.py`、`visualize_game.html`、`deploy_viz_to_gpu.sh` は通常運用では使用しない。
