# 2026-07-15

- P1/P2 修正（質問閉包の追加修正・その4）
  - `scripts/llm_turn_game_common.py`:
    - 回答すべき未回答質問がある場合に `reply_to_message_id` を省略した発言を自動補完しない
    - `reply_to_message_id` がない場合は `missing_reply_to_message_id_while_answer_required` を `transcript` に記録し、`forced_decision_reason` に反映
    - 無効な `reply_to_message_id` 時は `forced_decision_reason` に `invalid_reply_to_message_id` を反映（既存の理由があっても `absolute_budget_limit_reached` で上書きしない）
  - `hivc_sim/tests/test_turn_game.py`:
    - `reply_to_message_id` 省略の回帰テストを追加
    - 無効な返信ID後の `forced_decision_reason` が `invalid_reply_to_message_id` になることを検証
    - テスト数を 54 に更新

- P1/P2 修正（質問閉包の追加修正・その3）
  - `scripts/llm_turn_game_common.py`:
    - `can_ask_question` 条件を「自分宛の未回答質問がない かつ 残り予算で回答できる」に戻す
    - 回答すべき未回答質問があるのに `question_objection` を返した場合は `invalid_response_while_answer_required` として `transcript` に記録し、強制意思決定
    - 無効な `reply_to_message_id`（存在しない、または `addressed_to != speaker`）では質問を閉じず、`reply_to_message_id_invalid` フラグを記録
  - `hivc_sim/tests/test_turn_game.py`:
    - 回答すべき未回答質問がある発言で新しい質問を返すケースの回帰テストを追加
    - 質問者自身や宛先外が `reply_to_message_id` を指定しても質問を閉じないケースのテストを修正
    - テスト数を 53 に更新

- P1/P2 修正（質問閉包の追加修正・その2）
  - `scripts/llm_turn_game_common.py`:
    - `reply_to_message_id` による質問クローズ条件を「ID 一致 かつ q['addressed_to'] == speaker」のみに厳密化
    - 宛先エージェント以外が回答しようとした場合は `open_questions` を維持し、`transcript` に `reply_to_message_id_invalid` フラグを記録
    - `can_ask_question` 条件から「自分宛て未回答質問がないこと」を削除し、残り予算で全未回答分に回答できる場合に質問を許可するよう変更
  - `hivc_sim/tests/test_turn_game.py`:
    - `addressed_to` の補正結果を `== "beta"` に厳密化
    - 質問者自身や相手以外が `reply_to_message_id` を指定しても質問を閉じない回帰テストを追加
  - `pytest hivc_sim/tests -q` が 52 テストで通過

- P1/P2 修正（質問閉包の追加修正）
  - `scripts/llm_turn_game_common.py`:
    - `extract_json_discussion` で `speech_act == question_objection` なら `requires_response` をモデル値に関わらず常に `true` に正規化
    - `run_one_game` で `is_question` 時の `addressed_to` が相手プレイヤーでない場合は `other_speaker` へ補正
  - `hivc_sim/tests/test_turn_game.py`:
    - `requires_response: false` でも `question_objection` は回答待ちになるテストを追加
    - `addressed_to: "gamma"` でも相手に補正されて回答が返る回帰テストを追加
  - `pytest hivc_sim/tests -q` が 51 テストで通過

- P1/P2 修正
  - `scripts/llm_turn_game_common.py`:
    - `run_one_game` で未回答質問を残りの自由議論枠で回答し、絶対上限でのみ例外に変更
    - 例外時も `alpha`/`beta` の投票を実行し、無効な票は `None` にして `best_action` をフォールバック票として使わない
    - `max_discussion_turns < n_speakers` の場合は実効値を `n_speakers` へ繰り上げ、ログへ明示
    - `allocate_discussion_budgets` で合計が `max_discussion_turns` / `discussion_token_budget` を超えないよう修正
    - トークン配分を発言数配分に比例させ、端数は早い機会から配分
  - `hivc_sim/tests/test_turn_game.py`:
    - `allocate_discussion_budgets` の上限・比例配分テストを更新
  - `pytest hivc_sim/tests -q` が 48 テストで通過

# 2026-07-14

- REQUIREMENTS.md 更新（質問・応答の閉包と新評価指標）に合わせて実装を更新
  - `scripts/llm_turn_game_common.py`:
    - `extract_json_discussion` で `reply_to_message_id`, `addressed_to`, `requires_response` をパース
    - `get_discussion_message` を dict 返却に変更し、質問・応答メタデータを運ぶ
    - `discussion_prompt` に未回答質問への回答指示と予算警告を追加
    - `allocate_discussion_budgets` を新設し、実際の `opportunity_count` で発言数・トークン予算を配分
    - `run_one_game` を §7.1.3 質問と応答の閉包に対応:
      - `message_id`, `addressed_to`, `requires_response`, `reply_to_message_id` をトランスクリプトに記録
      - 未回答質問がある間は意思決定機会へ進まず、宛先エージェントの次の発言で回答を要求
      - 質問を出す際は回答1発言分のメッセージ・トークン予算を確保
      - 予算不足等で回答できない場合は `forced_decision_with_open_question` と理由を記録
      - ターン行に `unanswered_question_count`, `question_response_latency`, `forced_decision_with_open_question` を追加
  - `hivc_sim/turn_game_metrics.py`:
    - `unanswered_question_rate`, `question_response_latency_metric`, `forced_decision_with_open_question_rate` を追加
    - `compute_summary_metrics` に新指標を含める
  - `hivc_sim/tests/test_turn_game.py`:
    - `allocate_discussion_budgets` の配分ルールを検証
    - `extract_json_discussion` の質問メタデータパースを検証
    - `run_one_game` の質問→回答→意思決定の閉包をモックで回帰テスト
  - `hivc_sim/tests/test_turn_game_metrics.py`:
    - 新評価指標の単体テストを追加
  - `pytest hivc_sim/tests -q` が 48 テストで通過

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
