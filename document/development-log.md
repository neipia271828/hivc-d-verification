# 2026-07-13

- 評価妥当性修正を実装
  - `hivc_sim/turn_game_metrics.py`:
    - `_game_key` ヘルパーを追加し、`condition` + `seed`（ない場合 `game_id`）でゲームを識別
    - `plan_revision_quality` / `route_switch_quality` をゲーム識別子ごとにグループ化し、turn 昇順で処理
    - `terminal_metrics` / `rescue_wait_failure_rate` も同じゲーム識別子で集計
  - `hivc_sim/turn_game.py`:
    - `pod_ready_status` を beta 可視資源のみの部分診断に変更し、`_escape_conditions_met` による全条件判定をテキスト生成から外す
    - `role_specific_evidence` の beta ラベルを「beta確認済みの発進準備」に変更
    - 全条件充足時も「艇・資源の準備完了（浸水・船体は安全担当に確認）」と返し、alpha 確認の必要性を示す
  - `REQUIREMENTS.md` §4.6:
    - beta の観測情報を「通信・脱出艇と可視資源の準備条件」に明確化
    - 最終発進可否は alpha の安全情報が必要であることを追記
  - `hivc_sim/turn_game_cli.py`:
    - CSV 出力の `lineterminator` を `\n` に固定し、`git diff --check` 違反を解消
  - `hivc_sim/tests/test_turn_game_metrics.py`:
    - ゲーム境界・条件境界・インターリーブに対する `plan_revision_quality` / `route_switch_quality` の回帰テストを追加
  - `hivc_sim/tests/test_turn_game.py`:
    - beta 診断が `flooding` 変化に不変であること、可視条件不足のみを列挙すること、浸水・船体数値を `format_state` が含まないこと、行動 F の勝敗はゲーム内で異なることを検証するテストを追加
  - `hivc_sim/results/turn_game/{heuristic,mcts,random}_games.csv` と `*_summary.csv` を再生成
  - `pytest hivc_sim/tests -q` が 41 テストで通過

# 2026-07-13

- ローカル実験ログプレビュー設計を実装
- `scripts/download_gpu_logs.py` 新規作成
  - `configs/gpu_server.yaml` の SSH 接続情報を使い、GPU サーバーの実験出力を `hivc_sim/results/turn_game/downloads/<run-id>/` へ `rsync`
  - 取得対象の `all_games.csv` / 条件別 CSV / `summary.csv` の有無を検証
  - `manifest.json` を生成して取得元メタデータを記録
  - `--dry-run` / `--run-id` / `--remote-output-dir` / `--local-dir` / `--overwrite` 対応
- `scripts/local_preview.py` 新規作成
  - ローカル run ディレクトリをスキャンする軽量 HTTP サーバー
  - 既定の待受は `127.0.0.1:8765`（外部公開なし）
  - `/api/runs`, `/api/runs/<run>/files`, `/api/runs/<run>/file/<name>`, `/api/runs/<run>/summary` エンドポイント
- `scripts/local_preview.html` 新規作成
  - run 選択、ファイル選択、条件タブ、ゲーム選択、ターンタイムライン
  - 施設状態パネル、イベント、勝敗、エージェント議論、投票・合意、探索ベース評価、summary.csv 集計を表示
  - 前後ターン移動・自動再生対応
- `configs/experiment.yaml` の `live_jsonl` を `null` に変更
- `scripts/gpu_run.py` からライブサーバー / ngrok 関連を削除
  - 実験起動、ログ tail、状態確認、停止のみに整理
- `configs/gpu_server.yaml` から `remote_ngrok_path` / `live_port` / `live_jsonl` を削除
- `README.md` / `CLAUDE.md` を新オフライン運用に更新
- 旧ライブビューアー (`scripts/live_server.py`, `visualize_game.html`, `deploy_viz_to_gpu.sh`) は `archive/legacy-live-viewer/` へアーカイブ済み
- 合成サンプル CSV で `local_preview.py` の API 配信と HTML 読み込みを検証

# 2026-07-13

- `REQUIREMENTS.md` v2 ターン制合意形成ゲームに実装を更新
  - `hivc_sim/turn_game.py`:
    - 状態変数に `pod_readiness`, `pod_integrity`, `rescue_eta` を追加
    - 行動に `E`（脱出艇整備）と `F`（自力脱出）を追加
    - シナリオ群 `comms_favored`, `escape_favored`, `ambiguous`, `route_reversal` を実装
    - イベントに `relay_short`, `pod_flooding`, `current_change`, `backup_power_found`, `hull_fracture` を追加
    - 通信救助の勝利を `rescue_eta` 経過後に変更
    - 自力脱出の成否判定を実装
    - 終端スコアに `pod_readiness`, `pod_integrity` を追加
    - ルート追跡（`planned_route`, `optimal_route`, `route_switch`）と役別診断情報を追加
  - `hivc_sim/turn_game_metrics.py`:
    - `route_choice_accuracy`, `route_switch_quality`, `premature_launch_rate`, `rescue_wait_failure_rate`, `cross_role_evidence_use` 指標を追加
  - `scripts/llm_turn_game_common.py`:
    - 新行動・新状態・ルート追跡・役別診断情報をプロンプトと記録項目に反映
  - `scripts/qwen_turn_game_agent_smoke.py`, `scripts/qwen_two_agent_turn_game_smoke.py`:
    - プロンプト・状態表示・正規表現を新行動に対応
  - `hivc_sim/tests/test_turn_game.py`:
    - 通信救助、自力脱出、未達発進のテストを追加・更新
  - `hivc_sim/rollout_validation.py`:
    - 新シミュレータで §8 事前ロールアウト検証が通ることを確認

# 2026-07-13

- レビュー指摘を反映した修正
  - `hivc_sim/turn_game.py`:
    - ターン上限を `rescue_eta` 待機より先に判定し、5ターンを超えて通信救助が進行しないよう修正
    - `_post_start_state()` を導入し、基本消費・イベント効果を反映した「発進直前状態」で脱出可否を判定
    - `route_reversal` シナリオに決定済みイベント列 (`POD_FLOODING`, `SIGNAL_WINDOW`, `BACKUP_POWER_FOUND`) を追加し、勝ち筋の方針転換を保証
    - シナリオに `event_sequence` と `jitter_scale` フィールドを追加
  - `scripts/llm_turn_game_common.py`:
    - `format_state` をエージェント別にマスクし、alpha/beta に非対称な状態情報を提示
  - `scripts/download_gpu_logs.py`:
    - `run_id` のパス区切り文字と `..` を拒否し、`resolve()` 後に `local_dir` 配下か検証
  - `hivc_sim/tests/test_turn_game.py`:
    - `route_reversal` の方針転換をロールアウトで検証するテストを追加
  - `hivc_sim/turn_game_cli.py`:
    - `--scenario` オプションを追加

# 2026-07-13

- 追加レビュー指摘を反映した修正
  - `hivc_sim/turn_game.py`:
    - `_escape_conditions_met()` を「post 状態を受け取る判定」に、`_escape_conditions_met_current()` を「現在 state から post 状態を作る判定」に分離し、step() 内での二重適用を解消
    - `heuristic_policy` と `pod_ready_status()` を `_escape_conditions_met_current()` 使用に更新
    - `pod_ready_status()` から `flooding`（浸水過多）の記述を削除し、beta には可視の `pod_readiness`, `pod_integrity`, `oxygen`, `power` のみを診断文に含める
  - `hivc_sim/turn_game_metrics.py`:
    - `cross_role_evidence_use` を per-speaker 化し、相手の `role_specific_evidence` 中に含まれ、自分の evidence には含まれない語を、発言・最終理由で参照したターンを計測
    - `enrich_turn_row` で `cross_role_evidence_used` を追加記録
  - `scripts/local_preview.py`:
    - `_run_dir()` / `_read_file()` に `run_id` / `filename` の単一パス要素検証と `resolve()` 後の `downloads_dir` 配下検証を追加し、親ディレクトリ読み出しを防止

# 2026-07-13

- 追加レビュー指摘を反映した修正（経路指標の見直し）
  - `hivc_sim/turn_game_metrics.py`:
    - `route_switch_quality` を「イベント発生ターンで `prev_planned_route != optimal_route` となったターン」を分母に入れるよう修正
    - 当該ターンの `planned_route == optimal_route` を成功とし、「切り替えるべきだったのに切り替えなかった」失敗を分母に含める
  - `hivc_sim/turn_game.py`:
    - `optimal_route` を通信救助ルート・自力脱出ルートを強制した別々のロールアウト期待値差で定義するよう修正
    - `comms_forced_policy` / `escape_forced_policy` を新設し、A/B/D 等の支援行動がどちらの勝ち筋に属するかを推定せず、即時 Q 値比較ではなく経路全体の期待値を評価
    - `_route_value` を追加して強制ルートの平均終端スコアを計算
    - `play_policy_game` と `scripts/llm_turn_game_common.py` の `run_one_game` を `optimal_route` 新シグネチャに対応
    - `route_reversal` シナリオの初期値を `base_communication=0, base_pod_readiness=0, base_pod_integrity=0`、イベント列を `POD_FLOODING, BACKUP_POWER_FOUND, SIGNAL_WINDOW, BACKUP_POWER_FOUND, NONE` に変更し、`optimal_route` が `escape -> comms` に反転することを安定化
    - `heuristic_policy` で発進条件を満たしている場合は酸素・電力修理より先に `EXECUTE_ESCAPE` を選択するよう順序を調整
  - `hivc_sim/tests/test_turn_game.py`:
    - `test_route_reversal_changes_optimal_route` を `optimal_route(state, ...)` 呼び出しと `BACKUP_POWER_FOUND` イベント確認に更新
