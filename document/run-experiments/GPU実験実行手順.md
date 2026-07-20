# GPU実験CLIの使い方

HIVC-Dの実験は、ローカルMacのリポジトリルートから `uv run` の各コマンドで操作する。
実験本体はGPUサーバーで実行し、完了したログをMacへ取得してローカルGUIで閲覧する。

GPU上でHTTPサーバーやngrokを起動する必要はない。GUIはMacの `127.0.0.1` のみに公開される。

## 基本フロー

リポジトリルートで次の順に実行する。

```bash
# 1. ローカルのコードをcommit・pushし、GPUへ同期
uv run sync

# 2. GPUで1ゲームのsmokeを開始
uv run experiment

# 3. smoke完了後、ログをMacへ取得して科学的妥当性を検証
uv run download
uv run validate-smoke hivc_sim/results/turn_game/downloads/<smoke-run-id> --applicability required

# 4. 合格したsmoke runを指定して本実験を開始
uv run experiment --conditions control consulting hivc_d --games 100 --smoke-run-id <smoke-run-id>

# 5. MacでGUIビジュアライザーを起動
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

### smoke科学的妥当性ゲート

本実験の開始前に、完了したsmokeのCSV実測値を検証する。列が存在するだけでは合格にならない。

```bash
uv run download --run-id <smoke-run-id>
uv run validate-smoke \
  hivc_sim/results/turn_game/downloads/<smoke-run-id> \
  --applicability required
```

次の全条件を満たす場合だけ終了コード0になる。

- alpha/betaの測定Vが、同一の一様0.2ベクトルをコピーしたものではない
- `v_alignment_required=true` のターンが1件以上ある
- 必須V提案機会（`v_proposal_rate` の分母）が1件以上ある
- ID付きの受諾済みV*が1件以上ある
- `invalid_discussion_output_rate < 0.10` であり、実発言が存在する

失敗時は各gateの観測値と診断をJSONで表示し、終了コード1を返す。診断を保存する場合は `--report <path>` を使う。

V測定を行わないlegacy実験だけは、対象外であることを明示できる。

```bash
uv run validate-smoke <legacy-run-dir> --applicability not-applicable
uv run experiment --games 100 --scientific-gate not-applicable
```

`not-applicable` はV固有gateを意図的に無効化するため、通常の `soft_value` 実験には使用しない。

### 条件、ゲーム数、seedを指定する

```bash
uv run experiment \
  --conditions control consulting hivc_d \
  --games 30 \
  --seed 42 \
  --smoke-run-id episode-smoke-01
```

2ゲーム以上の新規runでは `--smoke-run-id` が必須で、GPU側にある指定smokeを再検証してから本実験runを作成する。gate不合格時はGPU本実験を起動しない。`--resume` は同一runのshard再開なので、この事前指定を再要求しない。

run IDを固定する場合:

```bash
uv run experiment --run-id episode-reproduction-01
```

同じrun IDは再利用できない。再実行時は別のrun IDを指定するか、自動生成を使う。
直列ランナーも既存CSV、`value_manifest.json`、`run_metadata.json`、既存 `stream.jsonl` を含む出力先を拒否する。`run_metadata.json` にはrun ID、git commit、開始・完了状態、およびCSV・stream・value manifestのハッシュが保存される。workflowが事前作成する `run_id`、`started_at`、`command.txt`、`run.log`、`pid` だけのディレクトリは使用可能である。

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
uv run validate-smoke --help
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
