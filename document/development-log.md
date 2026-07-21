# 2026-07-17 (review remediation)

- Gate A review 指摘に対する修正
  - `scripts/qwen_parallel_experiment.py`:
    - `_merge_value_manifests` で `(seed, condition)` 集合が期待と完全一致することを検査
    - 同一キーの重複・内容衝突・condition 欠落を `assignment_completeness=false` とし診断情報を追加
    - `game_profile_assignments` の重複排除を停止し、検査経路で検出できるように変更
  - `scripts/llm_turn_game_common.py`:
    - V 提案を出しただけでは自動的に `accept` しないように変更。プロポーザは `v_star_response: accept` を明示的に返すか、後続の V 応答で受諾する
    - `discussion_prompt` / `v_proposal_required_prompt` の例に明示的 self-accept を含めるよう更新
    - `v_star_failure_reason` が予算枯渇時に `v_negotiation_budget_exhausted` で上書きされ、直前の未解決理由は `v_star_unresolved_reason` に残す
    - `extract_json_discussion` / `_is_valid_discussion_payload` で必須キー・型・値（speech_act, message, action, reason, addressed_to, reply_to_message_id）を検証
    - JSON 契約違反出力を有効 `transcript` へ追加せず、`invalid_discussion_outputs` 監査経路へ分離
    - ターン CSV に `v_star_unresolved_reason`, `invalid_discussion_outputs` を追加
  - `hivc_sim/turn_game_metrics.py`:
    - `silent_unanswered_question_count` を `compute_summary_metrics` に統合
  - `hivc_sim/tests`:
    - `test_qwen_parallel_experiment.py` に manifest の condition 欠落・重複・内容衝突・欠落キーの Gate A テストを追加
    - `test_v_flow.py` に explicit self-accept、counter、V 予算枯渇理由、V 測定リトライのテストを追加
    - `test_turn_game.py` に JSON 前後テキスト拒否・必須キー欠落・型不一致・無効発話の監査経路テストを追加
    - `test_turn_game_metrics.py` に strict schema completeness と `silent_unanswered_question_count` のテストを追加
    - `test_profiles.py` に swap / orthogonal YAML ロードテストを追加
    - `pytest hivc_sim/tests -q` が 148 tests passed
  - `python3 -m py_compile` 対象ファイル構文確認と `git diff --check` を実施

# 2026-07-17

- RoleとValue分離によるV整合検証の要件実装
  - `hivc_sim/profiles.py`:
    - `ValueCriteriaSchema` dataclass で共通Vオントロジーを定義
    - `DEFAULT_VALUE_CRITERIA_SCHEMA` を `value-manifest-2` 用に ID / version / sha256 / criteria / schema_level 付きで提供
    - `Value.from_dict` が `ValueCriteriaSchema` を使って `initial_priority_weights` を検証・正規化
  - `scripts/llm_turn_game_common.py`:
    - `_normalize_v`, `_normalize_v_proposal` で完全な criteria 集合を必須化
    - `extract_json_v_measurement` が `ordered_criteria`, `weights`, `confidence` を検証
    - `v_alignment_required` を L1距離・ top criterion 不一致・ action mismatch の3条件で決定
    - V protocol 状態機械 (`I_SHARE` → `V_COMPARE` → `V_PROPOSE`/`V_RESPOND`/`V_NOT_REQUIRED` → `A_CHECK` → `FINAL_VOTE`) を実装
    - `v_measurement_prompt`/`v_proposal_required_prompt`/`v_proposal_response_prompt`/`discussion_prompt`/`decision_opportunity_prompt` を共通Vオントロジーに更新
    - `run_one_game` で `v_before` を自由議論前に測定し、必要時に発話予算を2発話確保して `v_proposal_required` プロンプトを発行
    - 質問の重複検出 (`_question_signature`)、JSON契約違反発話の取り扱い、無効発話を予算に加算しない制御を追加
    - CSV行に `role_value_assignment_id`, `value_criteria_schema_id`, `value_criteria_schema_version`, `v_alignment_required`, `v_alignment_requirement_reasons`, `v_protocol_state`, `v_protocol_transition_history`, `question_count`, `answered_question_count`, `duplicate_question_count`, `max_consecutive_duplicate_questions`, `invalid_discussion_output_count`, `v_proposal_required_prompt_issued`, `missing_v_proposal_after_required_prompt` を追加
    - `build_value_manifest` を `value-manifest-2` にし `value_criteria_schema` を含む; `append_profile_assignment` は固定プロファイルでも seed ごとの `role_value_assignment_id` を付与
  - `scripts/qwen_two_agent_experiment.py` / `scripts/qwen_parallel_worker.py`:
    - 固定プロファイルでも各 seed ごとに `append_profile_assignment` を呼び出し
  - `hivc_sim/turn_game_metrics.py`:
    - `v_process_metrics` の分母を `v_alignment_required` に変更し、後方互換フォールバックを実装
    - `missing_v_proposal_after_required_prompt_rate`, `v_schema_completeness_rate` を追加
    - `question_answer_rate`, `duplicate_question_rate`, `max_consecutive_duplicate_questions_metric`, `invalid_discussion_output_rate` を追加し `compute_summary_metrics` に統合
  - `scripts/local_preview.html`:
    - ターン詳細に `v_alignment_required`, `v_protocol_state`, `v_protocol_transition_history`, `v_proposal_required_prompt_issued`, `missing_v_proposal_after_required_prompt` を表示
    - Identity エリアに `role_value_assignment_id` と `value_criteria_schema` を表示
    - 条件比較テーブルに `v_alignment_required`, `v_protocol_state` を表示
  - `hivc_sim/tests/test_v_flow.py`:
    - 共通Vオントロジー (`oxygen`, `power`, `hull_damage`, `flooding`, `communication`) にテストデータを更新
    - プライバシーテストを persona テキストマーカー方式に変更
  - `pytest hivc_sim/tests -q` が 133 tests passed

# 2026-07-15

- GPU並列実験 orchestrator の最終検証
  - `python3 -m py_compile scripts/qwen_parallel_experiment.py` で構文を確認
  - `pytest hivc_sim/tests -q` が 75 tests passed

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

# 2026-07-14

- 自由議論の回答 JSON スキーマを修正
  - `scripts/llm_turn_game_common.py`:
    - 全自由議論出力で `addressed_to` と `reply_to_message_id` を必須キーとして明示
    - 通常発言・質問・未回答質問への回答で、状況別の有効な JSON 例を提示
    - 未回答質問がある場合は、対象の質問 ID と質問者を埋め込んだ回答専用 JSON 例に切り替え
    - 回答時に新しい質問や別話題を禁止し、`reply_to_message_id` を省略しないよう明記
  - `hivc_sim/tests/test_turn_game.py`:
    - 通常発言・質問スキーマに質問メタデータキーが含まれることを検証
    - 未回答質問への回答スキーマに正しい質問 ID が埋め込まれることを検証
  - `pytest hivc_sim/tests -q` が 56 テストで通過

# 2026-07-14

- GPU実験ワークフローを `uv run` の4コマンドへ統合
  - `pyproject.toml` / `scripts/workflow_cli.py`:
    - `uv run sync`: ローカルのGit状態・origin・ブランチを検証し、push後にGPU側で `git pull --ff-only`、HEAD一致を確認
    - GPU側に `.git` がない初回は、SSH agent forwardingでGit管理を復元。秘密鍵を複製せず、`.venv` と実験結果を保持
    - `uv run experiment`: 一意なrun IDを発行し、GPUで1エピソードをバックグラウンド起動。run単位でログ・PID・終了コード・CSVを保存
    - `uv run download`: 直前のrun IDを引き継ぎ、完了と終了コードを確認してMacへ取得
    - `uv run visualize`: `127.0.0.1` 限定のローカルGUIを起動しブラウザを自動表示
  - `.hivc-workflow.json` で直前runをローカル管理し、Git対象外に設定
  - GPU側runとローカルdownloadsをGit対象外に設定
  - `hivc_sim/tests/test_workflow_cli.py` にCLI登録・run ID検証・リモートコマンド生成のテストを追加
  - `README.md` に4コマンドの基本フローとオプションを追記
  - `pytest hivc_sim/tests -q` が 61 テストで通過
  - `uv run visualize --no-open --port 8876` でHTMLと `/api/runs` の応答、Ctrl+Cでの正常終了を確認

# 2026-07-14

- `uv run sync` にstage・commitの自動実行を追加
  - 未コミット変更がある場合は `git add -A` と `git commit` を自動実行してからpush・GPU pullへ進む
  - `--message` で自動commitメッセージを指定可能
  - `--allow-dirty` は未コミット変更を除外して現在のHEADだけを同期する互換オプションとして維持
  - dry-runではstage・commit・push・GPU pull予定を表示するだけでGit状態を変更しない
  - 自動commit後にhook等による未コミット変更が残った場合は同期を停止
  - `pytest hivc_sim/tests -q` が 62 テストで通過

# 2026-07-14

- `uv run experiment` の実行中判定による誤検出を修正
  - SSHで実行中の起動コマンド自身を `pgrep -f` が実験プロセスとして検出し、実験がない状態でもexit 23になる問題を解消
  - 全プロセスの文字列検索を廃止し、runディレクトリに記録したPID・終了コード・Linuxのプロセスコマンドラインを照合して実行中runを判定
  - 現在の起動用シェル自身と親プロセスを判定対象から除外
  - `hivc_sim/tests/test_workflow_cli.py` に自己誤検出を防ぐ回帰テストを追加
  - `pytest hivc_sim/tests -q` が 63 テストで通過

# 2026-07-14

- GPU実験CLIの利用手順を文書化
  - `document/run-experiments/GPU実験実行手順.md` を廃止済みのGPUライブビューアー手順から現行の `uv run` 4コマンド運用へ全面更新
  - 同期、実験開始、状態確認、ログ追跡、停止、ダウンロード、ローカル可視化、エピソード再実行、エラー対応を記載
  - `README.md` のuv統合CLI節から詳細手順へリンクを追加

# 2026-07-14

- ローカル実験ログGUIの改行入りCSV読込を修正
  - `scripts/local_preview.html` のCSVパーサーを、引用符内の改行・カンマ・エスケープ済み引用符を保持するレコード単位の実装へ置換
  - 最終投票時の思考テキストに含まれる改行でCSVレコードが分割され、議論JSON断片がチャットとして誤表示される問題を解消
  - `hivc_sim/tests/test_local_preview_html.py` に改行入り引用フィールドの回帰テストを追加
  - 実CSVをGUIのJavaScriptパーサーで検証し、5レコード・59列・議論26件を正しく復元することを確認
  - `pytest hivc_sim/tests -q` が64テストで通過

# 2026-07-14

- HIVC-D条件の合意形成プロンプトを詳細化
  - `scripts/llm_turn_game_common.py` の組込み `hivc_d` プロトコルを、I（情報共有）、V（共通基準 V* による優先順位整合）、A（実行可能性確認）の具体的な対話・投票手順へ拡張
  - 不可視状態を推測で断定しないこと、質問への先行回答、条件付き提案、最終投票前チェックを明記
  - 自由議論と最終投票の両プロンプトへ同じ詳細プロトコルが挿入されることを回帰テストで検証
  - `pytest hivc_sim/tests -q` が65テストで通過

# 2026-07-14

- `consulting` 条件の一般助言をHIVC-D条件と同程度に詳細化
  - 状況整理、選択肢のリスク・便益比較、実行前確認、最終投票前チェックを明記し、追加指示の分量・具体性を揃えた
  - 共通ゲーム情報や可視情報は従来どおり全条件で同一とし、`consulting` には I/V/A や共通基準 `V*` といったHIVC-D固有概念を入れない
  - 両条件の手順文量と概念分離を検証する回帰テストを追加
  - 手順文は `consulting` 778文字、`hivc_d` 859文字（91%）であることを確認
  - `pytest hivc_sim/tests -q` が66テストで通過

# 2026-07-15

- GPU並列実験の効率化要件を策定
  - `document/要件定義/GPU並列実験効率化要件.md` を追加
  - RTX A5000×2の実測値をbaselineとし、1GPU・1worker、2GPU同時実行の目標構成を定義
  - 条件ごとのseed集合を両GPUへ同じ規則で割り当て、GPU割当と条件効果の交絡を防止する要件を明記
  - master run、shard manifest、GPU負荷監視、温度時の安全停止、失敗再開、結果整合性検査を定義
  - baseline比1.4倍以上のgames/hourと、VRAM・温度・エラーの安全基準を採用条件に設定
  - 現在稼働中の `episode-20260715-210345` は中断せず、現行方式のbaselineとして完了させる導入順序とした

- GPU並列実験を実装
  - `scripts/qwen_parallel_experiment.py` (orchestrator) 新規作成
    - 1GPU・1worker、条件ごとに blocked schedule（control → consulting → hivc_d）でshardを起動
    - `nvidia-smi` によるGPU検出、VRAM・温度・他プロセスの事前検査
    - 30秒間隔の `gpu_metrics.csv` 記録、温度閾値・thermal slowdown検出、`pause_request` による安全停止
    - `master_manifest.json` / `merge_report.json` 生成、shard結果の整合性検査と `summary.csv` 再計算
  - `scripts/qwen_parallel_worker.py` (shard worker) 新規作成
    - `CUDA_VISIBLE_DEVICES` で1GPUを固定し、1 condition の連続seed範囲を実行
    - 各ゲーム後に `pause_request` を検知、`paused_thermal` 状態と `shard_manifest.json` を記録
  - `scripts/workflow_cli.py` を並列モード対応
    - `uv run experiment --parallel --gpus 0 1 --conditions ... --games ... --seed ...` などに対応
    - `--workers-per-gpu`, `--temperature-warning`, `--temperature-stop-scheduling`, `--resume` 引数追加
    - 既存の非並列実行、status/logs/stop、runディレクトリ管理を維持
  - `hivc_sim/tests/test_qwen_parallel_experiment.py` と `hivc_sim/tests/test_workflow_cli.py` に並列系テストを追加
  - `pytest hivc_sim/tests -q` が75テストで通過

# 2026-07-16

- RoleとValue分離によるV整合検証要件を策定
  - `episode-20260715-210345` の90ゲームを診断根拠とし、情報利用は増えたが、固定Roleに対応する行動選好はほぼ変化しなかったことを記録
  - `Role`、`Persona`、初期V、現在V、グループ判断用V*、交渉特性の責務分離を定義
  - 個人Vの完全一致ではなく、個人Vを残したまま受諾できる共通基準V*を合意対象として定義
  - 議論前V・行動、V提案、accept/reject/counter、受諾済みV*、最終投票の独立記録と状態永続化を要件化
  - `legacy_hard`、`soft_value`、`expertise_only` の三モードと、control / consulting / hivc_d の対応あり比較を定義
  - V*の優先順位を事前に与えない `hivc_d` version 2 を主検証とし、現行相当の固定V*手順を `hivc_d_prescribed_v1` 感度分析として分離
  - 主検証を `soft_value` の100seed以上とし、V*成立、行動整合、regret・成果への因果連鎖を分解して評価する基準を定義
  - 本フェーズでは要件定義のみを行い、実装とGPU再実験は未実施

## RoleとValue分離によるV整合検証の実装・ローカル検証

- `hivc_sim/profiles.py` に Role / Persona / Value の分離スキーマ、検証、SHA-256算出、`legacy_hard` / `soft_value` / `expertise_only` の読み込みを追加
- `scripts/llm_turn_game_common.py` に議論前の `v_before`・`action_before`、V提案、`accept` / `reject` / `counter`、明示的な両者受諾によるV*成立、投票後の `v_after`、V*状態の同一ターン内引き継ぎを追加
  - 未受諾の提案をV*として補完せず `unresolved` として記録する設計を採用
  - `v_before` は全framework条件で同じ測定契約を使い、`v_after` は最終投票確定後に取得して既存の質問・応答closureを維持
- `hivc_sim/turn_game_metrics.py` にV提案率、V*受諾率、V距離・gain、票変更率、V*行動整合率、未解決率と分子・分母を追加し、`compute_summary_metrics` に統合
- `scripts/llm_turn_game_common.py` と `scripts/qwen_two_agent_turn_game_smoke.py` のCSV行を要件 §9.1 のV列へ対応
- `scripts/qwen_two_agent_experiment.py` と `scripts/qwen_parallel_worker.py` に、実行時プロファイル・framework・設定・Git commitを保存する `value_manifest.json` 生成を追加
- `scripts/local_preview.py` / `scripts/local_preview.html` にvalue manifest配信、Vタイムライン、同一seed・turnの条件比較、旧CSVの「記録なし」表示を追加
- `hivc_sim/tests/` にプロファイル検証、V交渉・状態引き継ぎ、V指標、manifest、プレビュー互換性のテストを追加・更新
- ローカル検証結果
  - `pytest hivc_sim/tests -q`: 107 passed
  - 対象Pythonファイルの `python3 -m py_compile`: 成功
  - `git diff --check`: 成功
  - GPU実験は実行していない

## Role/Value実装レビュー7項目の修正

- `soft_value` / `expertise_only` の分離プロファイルを追加し、既定実験設定を `soft_value` へ切り替え
- 状態可視性と役割固有根拠を、固定agent名ではなく解決済みRoleの `observation_scope`、`expertise_domains`、`responsibility` から生成するよう変更
- 議論前の個人V・行動・理由をprivateとして分離し、HIVC-Dで `share_v_before=true` が明示された場合だけ共有状態へ追加
- 並列shardの `value_manifest.json` をmaster runへ結合し、ローカルプレビューでmasterおよび旧shard-only runを配信可能に変更
- 単一runnerはgame seedごと、並列runnerはseed rangeごとに条件順を決定論的にランダム化
- `legacy_hard` のプロンプトでは固定判断基準を明記し、更新可能という表現を除外
- workflow CLIへ `hivc_d_prescribed_v1` と `--role-value-mode` を追加し、通常・並列runnerへ伝播
- ローカル検証結果
  - `pytest hivc_sim/tests -q`: 115 passed
  - GPU実験は実行していない
  - pushは実行していない

## レビュー指摘5件の修正

- `scripts/qwen_parallel_experiment.py` と `scripts/local_preview.py` の value manifest 結合キーを、生成側の `game_profile_assignments` に統一
  - 結合側・プレビュー側が存在しない `game_entries` を読んでいたため、shard 2以降の割付けが失われていた
  - テストも `game_profile_assignments` を使うよう更新
- `scripts/qwen_parallel_experiment.py` の shard 生成を条件単位から worker（GPU）単位へ変更
  - 各 worker は担当 seed 範囲内の全条件を、seed ごとに `condition_order_for_seed` でシャッフルされた順序で実行
  - `counterbalanced_shard_rounds` は per-seed シャッフル済みの shard リストを1ラウンドとして返す
  - `condition_order_for_seed` を `scripts/llm_turn_game_common.py` に移動し、単一 runner と並列 worker で共有
- `--role-value-mode` 単独指定時の role_file 自動選択を追加
  - `scripts/llm_turn_game_common.py` に `resolve_role_file_path` を追加
  - `qwen_two_agent_experiment.py`、`qwen_parallel_worker.py`、`qwen_parallel_experiment.py` で config 読み込み後に解決
  - workflow CLI へ `--role-file` 引数を追加し、runner へ転送
- `pyproject.toml` の `dependencies` に `requirements.txt` と同じ実行・テスト依存を追加
  - `uv run pytest -q` で 115 passed を確認
- `scripts/llm_turn_game_common.py` の自由議論プロンプトから、HIVC-D 固有の V 共有契約を control/consulting へ提示していた部分を除去
  - `hivc_d` / `hivc_d_prescribed_v1` 条件でのみ V 提案・V*応答・`share_v_before=true` の説明を含める
- ローカル検証結果
  - `pytest -q`: 115 passed
  - `uv run pytest -q`: 115 passed
  - GPU実験は実行していない
  - pushは実行していない

## GPU事前検査の nvidia-smi クエリ互換性修正

- GPUサーバーのNVIDIAドライバで未対応の `clocks_throttle_reasons.thermal` を、対応済みの `clocks_throttle_reasons.sw_thermal_slowdown` に変更
- `nvidia-smi` が非0終了した場合、終了コード、stdout、stderrをすべてエラーに記録するよう改善
  - 従来はstderrのみ記録していたため、stdoutに出力された無効フィールドエラーが空文字として表示されていた
- `hivc_sim/tests/test_qwen_parallel_experiment.py` に、対応フィールドの使用とエラー詳細保存の回帰テストを追加
- ローカル検証結果
  - `uv run pytest hivc_sim/tests -q`: 118 passed
  - `python3 -m py_compile scripts/qwen_parallel_experiment.py`: 成功
  - `git diff --check`: 成功
- GPUサーバー上で修正後と同一の `nvidia-smi` クエリが2 GPUとも成功することを確認
- 実GPU出力の `Not Active` が部分文字列判定で `Active` と誤認される既存バグも修正
  - thermal slowdown系フィールドは前後空白と大小文字を正規化し、`Active` と完全一致した場合だけ真とする
  - `Not Active` の3フィールドをすべて偽と判定する回帰テストを追加
- GPU実験本体は実行していない

## 対応ありseed・parallel config hash・中断run結合の設計修正

- `--games N` を1条件あたりの対応ありseed数として固定
  - 3条件・100seedはseed 100個と総ゲーム300件に分けてmanifestへ記録
  - 各shardで `seed_count × condition数 == task数` かつ各seedに全条件が1回ずつあることを起動前検査
- orchestratorのrun全体 `config_hash` をworkerへ明示伝搬し、worker固有の `shard_config_hash` と分離
- `paused_thermal` などshard未完了時は、seed完全性とvalue manifest結合をskip扱いにし、部分CSV行数だけを進捗として記録
- worker再起動前に古い `pause_request` を削除し、即時再停止を防止
- `--resume` が同一runの既存manifestを上書きする前に読み、完了shardのみ再利用するよう修正
  - run ID省略時は最後runを再利用し、resume時の既存ディレクトリチェックを通常起動と分離
  - 旧 `exit_code`、`finished_at`、`run.log` は `.pre-resume` に退避し、再開中にstatusが旧終了コードを読まないようにする
- ローカル検証結果
  - `uv run pytest hivc_sim/tests -q`: 125 passed
  - 対象Pythonファイルの `py_compile`: 成功
  - `git diff --check`: 成功
- GPU実験本体は実行していない

## GPU電力上限の一時適用・自動復元

- `uv run experiment --parallel` に `--power-limit-w` を追加
  - 未指定時はGPU設定を変更せず、指定時だけ選択GPUへ同一の電力上限を適用
  - GPUごとの現在値・既定値・最小値・最大値を取得し、許容範囲外ならworker起動前に停止
  - `nvidia-smi` の設定失敗・権限不足・適用後照合不一致でもworkerを起動しない
- 実験開始前のpower limitをGPUごとに保存し、正常終了・エラー終了・SIGTERM終了時に元の値へ戻して再照合
  - 複数GPUへの適用途中で失敗した場合も、設定済みGPUをロールバック
  - `uv run experiment --stop` はworkerを先に停止し、次にorchestratorへSIGTERMを送って復元処理を実行
  - SIGKILL・ホスト停止など復元不能なケースは要件書に制約として明記
- `master_manifest.json` の `power_limit_policy` に指定値、開始前値、許容範囲、適用値、復元値・成否を保存
- GPUサーバーを読み取り確認し、GPU 0/1はいずれも現在値230W、許容範囲100W〜230W、温度35℃/39℃であることを確認
- ローカル検証結果
  - `uv run pytest hivc_sim/tests -q`: 130 passed
  - 対象Pythonファイルの `py_compile`: 成功
  - `git diff --check`: 成功
  - `uv run experiment ... --power-limit-w 180 --dry-run`: CLIからGPU runnerへの引数伝播を確認
- GPUのpower limit変更およびGPU実験本体はまだ実行していない

## sudo不要の温度連動thermal duty cycle

- GPU power limitを変更できない一般ユーザー向けに `--thermal-duty-cycle` を追加
  - 既定では78℃以上でworkerプロセスグループへ `SIGSTOP`
  - 70℃以下まで冷却後、`SIGCONT` で同じworkerを再開
  - 83℃の新規起動停止とthermal slowdown時のゲーム境界停止は従来どおり維持
- GPU監視間隔を30秒から5秒へ短縮し、230W動作時の急な温度上昇へ追従
- モデル・プロンプト・seed・生成パラメータは変更せず、wall-clock時間のみを延長する設計
- `thermal_events.jsonl` に停止・再開時刻、温度、GPU、shard、PID、累積停止時間を保存
- `gpu_metrics.csv` に停止中shardを、`master_manifest.json` にshard別の停止回数・累積停止秒数を保存
- `uv run experiment --stop` は停止中workerへSIGTERM後にSIGCONTを送り、終了シグナルを確実に処理可能に変更
- CLIで温度を変更する場合は `resume < suspend < stop-scheduling` を起動前検査
- ローカル検証結果
  - `uv run pytest hivc_sim/tests -q`: 133 passed
  - 対象Pythonファイルの `py_compile`: 成功
  - `git diff --check`: 成功
- GPU実験本体はまだ実行していない

## thermal duty cycle pilot分析に基づくV整合要件の改訂

- `episode-20260717-202452` の取得済みログを診断し、GPU運用は合格、HIVC-D効果検証は不合格と判定
  - 全4 HIVC-DターンでV提案0件、`missing_v_proposal`
  - V測定のcriteriaがRole固有の一部項目へ縮退
  - 質問の同文反復、実質未回答と指標の不一致、壊れたJSONの有効message混入
  - 固定profileの `game_profile_assignments` 欠落
- `document/要件定義/RoleとValue分離によるV整合検証要件.md` を更新
  - 共通Vオントロジーと完全次元バリデーション
  - Role×Valueの交換・直交割付けによる交絡検査
  - `v_alignment_required`、proposal機会保証、HIVC-D状態機械
  - 質問型正規化、回答優先、反復抑止、JSON契約違反の拒否
  - 固定profileを含むseed・condition別manifest割当完全性
  - 会話契約指標、受入基準、Gate A〜Cの大規模GPU実験開始ゲート
- 本更新は要件定義のみで、修正実装および追加GPU実験は実施していない

## レビュー指摘7件の修正

- `scripts/llm_turn_game_common.py`:
  - V交渉ループを再構成し、提案者は自分の提案を自動受諾、相手には accept/reject/counter を要求
  - HIVC-Dで `v_alignment_required=true` のターンは受諾済みV*ができるまでA_CHECK/FINAL_VOTEへ進まず、V予約予算（3発話）で再試行・counter応答に対応
  - V予約予算を使い果たしても未解決の場合は `forced_decision_reason=v_negotiation_budget_exhausted` で状態遷移
  - `_extract_json_object` を厳密化し、JSON以外の前後テキスト・Markdownフェンス・断片がある場合は契約違反として拒否
  - `get_discussion_message` で `speech_act` 無効または `message` 空の出力を `invalid_discussion_output` として拒否し、`information_request` への補完を停止
  - V測定（`v_before` / `v_after`）に `max_v_measurement_retries` による再試行を実装し、`v_measurement_retry_count` を記録
  - 回答すべき未回答質問に新しい質問を返した場合は強制遷移せず、同じagentに再試行させる (`invalid_response_while_answer_required`)
  - 観測不能を示す応答を `unanswerable_question` として閉じる簡易検出を追加
  - 未回答質問が残ったまま強制遷移理由のない状態で決定した場合を `silent_unanswered_question_count` として記録
  - CSV行に `unanswerable_question_count` と `silent_unanswered_question_count` を追加
  - `hivc_d_prescribed_v1` のV*を共通Vオントロジー (`oxygen`, `power`, `hull_damage`, `flooding`, `communication`) に統一
- `hivc_sim/turn_game_metrics.py`:
  - `_v_schema_complete` で weights/ordered_criteria が共通Vオントロジーの5項目と完全一致することを検査
  - `DEFAULT_VALUE_CRITERIA_SCHEMA` を `profiles.py` からインポート
- `scripts/llm_turn_game_common.py` / `scripts/qwen_two_agent_experiment.py` / `scripts/qwen_parallel_worker.py`:
  - `append_profile_assignment` を (seed, condition) 単位に変更し、`condition` フィールドを追加
  - 同一seedのframework条件間では `role_value_assignment_id` を共有
- `scripts/qwen_parallel_experiment.py`:
  - `_merge_value_manifests` で `game_profile_assignments` の期待件数 `games × conditions` と実際件数を比較
  - 欠落・重複がある場合は `assignment_completeness=false`、診断用 `missing_assignments` を記録
  - `_merge_results` で `value_manifests_mergeable` チェックに assignment 完全性を含める
- `configs/profiles_soft_value_swap.yaml` / `configs/profiles_soft_value_orthogonal.yaml`:
  - Role×Value交換・直交のsmoke用割付けを新規作成
- `hivc_sim/tests/test_turn_game.py`:
  - 「回答すべき質問に再質問を返す」テストを、強制遷移から再試行後に正しく回答するケースへ更新
- ローカル検証結果
  - `uv run pytest hivc_sim/tests -q`: 133 passed
  - 対象Pythonファイルの `py_compile`: 成功
  - `git diff --check`: 成功
  - GPU実験本体は実行していない
  - pushは実行していない

## 2026-07-17: V-negotiation・JSON契約・質問ルーティングのblocker修正

### 目的
GPU smoke実行前に指摘された6件のblocker/high/medium項目を修正する。

### 実装内容

- `scripts/llm_turn_game_common.py`:
  - **[Blocker 1] 必須V提案のself-accept保存**: `v_proposal_required_prompt` が提案とself-acceptを同じJSONで要求しているにもかかわらず、実行側が `proposal, _ = parse_v_negotiation(...)` でself-acceptを破棄していた問題を修正。`proposal, self_response` の両方を取得し、`v_responses[agent]` とtranscriptへ記録するように変更。
  - **[Blocker 2] counter提案者の自動accept廃止**: 自由議論・必須交渉の両パスで、counterを受け取るとシステムが自動的にacceptを追加していた仕様を廃止。両agentが同一提案を明示的にacceptすることがV*受諾の要件であり、counter提案者も後続応答で明示的にacceptする必要がある。
  - **[High 3] JSON型検証の抜け修正**: `_is_valid_discussion_payload` で `reply_to_message_id: {}` (辞書) が正規化後にNoneとなり型検査を迂回する問題と、`addressed_to: 123` (整数) が文字列として受理される問題を修正。正規化前の値について、`addressed_to` は `str|null`、`reply_to_message_id` は `str|int|null` (bool除外) のみ厳密に受理するように変更。
  - **[High 4] invalid JSONの再試行実装**: invalid出力を監査経路へ分離しただけでしたが、そのまま発話数・予算を消費して次話者へ進んでいた問題を修正。`max_discussion_retries` (既定1) による同一agentへの修復プロンプト再送と、`discussion_retry_count` の記録を実装。リトライ上限後もinvalidの場合は監査経路へ確定保存する。
  - **[Medium 5] 質問signatureと回答不能ルーティング**:
    - `_question_signature` を `normalized_requested_fields` ベースに変更。`requested_fields` が明示されている場合はそれを正規化してsignatureに使い、未明示の場合は action+reason+message にフォールバックする。
    - `_normalize_requested_fields` ヘルパーを追加（str/list/tuple をソート済み小文字セットへ正規化）。
    - `observation_scope` に基づく回答不能判定を実装。`requested_fields` が両agentとも観測できない場合は `unanswerable_question` として閉じ、再送しない。
    - `closed_questions` リストを追加し、回答済み質問を移動。closed questionの再質問は `reask_reason` がある場合のみ許可する。
    - `get_discussion_message` で `requested_fields` と `reask_reason` を raw_payload から抽出し、レスポンスdictに含める。
    - 質問JSON契約に `requested_fields` と `reask_reason` の任意キー説明を追加。
- `hivc_sim/turn_game_metrics.py`:
  - **[Medium 6] schema completeness の全Vレコード検査**: `_v_schema_complete` が `alpha_v_before` と `beta_v_before` の項目集合しか確認していなかった問題を修正。`v_after` (存在時)、weightsの全値の有限性・非負性、weights合計の1.0一致（1e-6許容誤差）を直接検査するように変更。
- `hivc_sim/tests/test_v_flow.py`:
  - `test_counter_proposal_is_auto_accepted_by_counter_proposer` を `test_counter_proposal_requires_explicit_self_accept` に改名し、counter提案者が明示的にacceptしない場合はunresolved、明示的にacceptすればacceptedになる両ケースを検証。
- `hivc_sim/tests/test_turn_game.py`:
  - `addressed_to: 123` (整数)、`reply_to_message_id: {}` (辞書)、`reply_to_message_id: true` (bool) の型検証拒否テストを追加。
  - `_normalize_requested_fields` と `_question_signature` の requested_fields ベース動作テストを追加。
  - `test_invalid_discussion_output_is_not_added_to_transcript` を `test_invalid_discussion_output_triggers_retry_and_recovers` と `test_invalid_discussion_output_retry_exhaustion_records_audit` に分割し、リトライ成功時とリトライ枯渇時の両方を検証。
- `hivc_sim/tests/test_turn_game_metrics.py`:
  - `_v_schema_complete` の v_after検査、weight合計検査、NaN/負値検査テストを追加。
- ローカル検証結果
  - `python -m pytest hivc_sim/tests -q`: 153 passed (148 → 153)
  - 対象Pythonファイルの `py_compile`: 成功
  - `git diff --check`: 成功
  - GPU実験本体は実行していない
  - pushは実行していない

## 2026-07-17: counter経路・scopeルーティング・token計数の追加修正

### 目的
前回修正後のレビューで指摘された3件のblocker/high項目を修正する。

### 実装内容

- `scripts/llm_turn_game_common.py`:
  - **[Blocker] counter経路の4発話予算確保**: `reserved_v_messages` を3→4に変更。自動accept廃止後のcounter経路は最大4発話が必要 (alpha提案+self-accept → beta counter → alpha accept → beta self-accept)。
  - **[Blocker] counter出力でself-accept同時表現スキーマ**: `parse_v_negotiation` で `v_star_response.self_accept` フラグを解析し、`self_accept_for_counter_id` としてresponseに含める。両パス（自由議論・必須交渉）で、counter出力時に `self_accept_for_counter_id` があればcounter提案への明示的acceptを別途 `v_responses` へ記録する。これによりcounter経路が3発話で合意に到達可能になる。
  - **[Blocker] v_proposal_response_prompt へ self_accept 説明追加**: counter出力時に `self_accept: true` で自分のcounter提案にも同意できることをプロンプトに明記し、JSONテンプレートに `self_accept` フィールドを追加。
  - **[High] observation_scopeによる質問ルーティング完成**:
    - 各requested_fieldについて、alpha/betaどちらが観測可能かを3分類 (only_alpha, only_beta, both, neither) で判定。
    - 全fieldを両者とも観測できない場合は `unanswerable_question` として閉じる。
    - 一部fieldだけ観測不能な場合は `unanswerable_partial_fields` としてtranscriptに記録。
    - 現在のaddressed_toが観測できず、もう一方が観測できる場合は宛先を切り替える (scope_routed)。
    - scope回答不能質問を `closed_questions` へ保存し、reask_reasonなしの再送を抑止。
    - requested_fields なしの場合は従来通りもう一方のagentへ正規化。
  - **[High] JSONリトライ分のtokenを各attemptで加算**: リトライループ内で各attemptの生成tokenを `token_count` に累積加算し、リトライ分の推論負荷が `discussion_token_budget_used` に正しく反映されるように修正。
- `hivc_sim/tests/test_v_flow.py`:
  - `test_counter_happy_path_reaches_accepted_in_run_one_game`: counter経路でself_accept=trueを使い3発話で合意に到達することをrun_one_game統合テストで検証。
  - `test_counter_without_self_accept_needs_extra_message`: self_accept=falseの場合4発話で合意に到達することを検証。
- `hivc_sim/tests/test_turn_game.py`:
  - `test_scope_unanswerable_question_is_closed_and_not_resent`: 両agentとも観測できないfieldへの質問がunanswerableとして閉じられ、closed_questionsに保存されることを検証。
  - `test_scope_routed_question_goes_to_observing_agent`: requested_fieldsを観測できるagentへ質問がルーティングされることを検証。
- ローカル検証結果
  - `python -m pytest hivc_sim/tests -q`: 157 passed (153 → 157)
  - 対象Pythonファイルの `py_compile`: 成功
  - `git diff --check`: 成功
  - GPU実験本体は実行していない
  - pushは実行していない

## 2026-07-17: observation_scopeルーティングの追加修正 (self_observable・partial_unanswerable・重複検出順序)

### 目的
前回のscopeルーティング実装に対するレビューで指摘された3件を修正する。

### 実装内容

- `scripts/llm_turn_game_common.py`:
  - **[High] 質問者だけが観測できるfieldの質問を self_observable_question として閉じる**:
    - 従来は `fields_for_other` から質問者自身が観測できるfieldを除外し、空になると宛先を変更しなかった。その結果、alphaだけが観測できる `hull_damage` への質問が、観測できないbetaへ残っていた。
    - 各fieldを「質問者のみ観測可能 (only_speaker)」「相手のみ観測可能 (only_other)」「両者観測可能 (both)」「両者とも観測不可 (neither)」の4分類で判定するよう変更。
    - 全fieldが質問者のみ観測可能（または質問者＋両者観測不可）で、相手が観測できない場合は `self_observable_by_scope=True` として質問を閉じる (`closed_as_self_observable=True`, `requires_response=False`)。
    - `self_observable_question_count` を新設し、CSV行にも出力する。
    - 自己観測可能質問も `closed_questions` へ保存し、reask_reasonなしの再送を抑止する。
  - **[High] 一部回答可能な質問を全体回答不能と誤判定する問題を修正**:
    - 従来の判定条件が `only_alpha_fields` と `only_beta_fields` しか見ておらず、`both_observe_fields` を考慮していなかった。その結果、`requested_fields=["oxygen", "unknown_field"]` で oxygen を両者が観測できる場合でも、質問全体が unanswerable になっていた。
    - 判定条件に `both_observe_fields` を追加し、両者観測可能なfieldが1つでもあれば全体を unanswerable にしない。観測不能なfieldは `unanswerable_partial_fields` に記録する。
  - **[Medium] 重複検出をscope判定の前に移動**:
    - 従来はscope判定で `closed_questions` に保存した質問と同じsignatureの質問が、次回のscope判定で再び unanswerable/self_observable として閉じられ、重複検出に到達しなかった。
    - 重複検出ブロックをscope判定ブロックの前に移動し、closed_questionsにある質問と同じsignatureの質問は scope判定に入る前に `duplicate_question=True` として処理する。
    - 重複判定用に `addressed_to` の正規化を重複検出ブロック内で先に行う。
- `hivc_sim/tests/test_turn_game.py`:
  - `test_scope_unanswerable_question_is_closed_and_not_resent`: 発話上限を6に増加し、alphaが常にmorale質問を送り続けるシナリオで、1回目がunanswerableとして閉じられ、2回目以降がduplicate_questionとして拒否されることをassert。`duplicate_question_count >= 1` を検証。
  - `test_scope_self_observable_question_is_closed` (新規): alphaがhull_damage (alphaのみ観測可能) への質問を出した場合、`closed_as_self_observable=True` として閉じられ、`self_observable_question_count >= 1` になることを検証。
  - `test_scope_partial_unanswerable_with_both_observe_not_full_unanswerable` (新規): `requested_fields=["oxygen","morale"]` で oxygen が両者観測可能、morale が両者観測不可の場合、質問全体が unanswerable にならず、morale が `unanswerable_partial_fields` に記録されることを検証。
- ローカル検証結果
  - `python -m pytest hivc_sim/tests -q`: 158 passed (157 → 158)
  - 対象Pythonファイルの `py_compile`: 成功
  - `git diff --check`: 成功
  - GPU実験本体は実行していない
  - pushは実行していない

## 2026-07-17: question_count分母修正とunanswerable requires_response修正

### 目的
scope分岐後の `question_count` が unanswerable/self_observable/duplicate 質問を含まず、`duplicate_question_rate` の分母が0になりNaNになる問題を修正する。

### 実装内容

- `scripts/llm_turn_game_common.py`:
  - **[High] question_count を分岐前に加算**: `question_count += 1` を重複検出ブロックの先頭（`is_question` 判定直後）に移動し、unanswerable/self_observable/duplicate いずれの分岐でも `continue` する前に加算されるようにした。通常の質問パス（`open_questions.append`）の `question_count += 1` は削除し二重加算を防止。
  - **[High] unanswerable質問の requires_response を False に修正**: 回答を求めない質問が `requires_response=True` のままにならないよう `False` に変更。
- `hivc_sim/tests/test_turn_game.py`:
  - `test_scope_unanswerable_question_is_closed_and_not_resent`: `question_count >= 2`、`question_count >= unanswerable + duplicate`、`requires_response is False`、`duplicate_question_rate` がNaNにならないことをassert。
  - `test_scope_self_observable_question_is_closed`: `question_count >= 1`、`question_count >= self_observable_question_count` をassert。
- ローカル検証結果
  - `python -m pytest hivc_sim/tests -q`: 158 passed
  - 対象Pythonファイルの `py_compile`: 成功
  - `git diff --check`: 成功
  - GPU実験本体は実行していない
  - pushは実行していない

## 2026-07-20: GPU実験完了 (100ゲーム x 3条件)

### 実行環境
- GPUサーバー: RTX A5000 x2
- モデル: Qwen3-14B (4bit量子化)
- config: configs/experiment.yaml
  - games=100, seed=42, max_new_tokens=256, max_discussion_turns=12
  - discussion_token_budget=3072, evaluator_rollouts=64
  - role_value_mode=soft_value, enable_thinking=false
- 実行時間: 約20時間 (Jul17 22:59 → Jul20 10:05)

### 結果サマリ (summary.csv)

| メトリクス | control | consulting | hivc_d |
|---|---|---|---|
| win_rate | 0.36 | 0.30 | 0.36 |
| survival_rate | 0.49 | 0.44 | 0.42 |
| mean_return | 423.25 | 369.65 | 419.35 |
| mean_regret | 152.25 | 198.18 | 174.89 |
| expert_match_rate | 0.355 | 0.345 | 0.381 |
| route_choice_accuracy | 0.721 | 0.768 | 0.778 |
| cross_role_evidence_use | 0.463 | 0.529 | 0.537 |
| unanswered_question_rate | 0.473 | 0.261 | 0.030 |
| duplicate_question_rate | 0.588 | 0.092 | 0.200 |
| invalid_discussion_output_rate | 0.774 | 0.922 | 0.932 |
| vote_revision_rate | 0.967 | 0.984 | 0.991 |
| v_schema_completeness_rate | 1.0 | 1.0 | 1.0 |
| v_proposal_rate | NaN (0/0) | NaN (0/0) | NaN (0/0) |
| v_star_acceptance_rate | NaN | NaN | 0.0 (0/6) |
| v_alignment_distance_before | 0.0 | 0.0 | 0.0 |
| v_alignment_distance_after | 0.0 | 0.0 | 0.0 |
| v_alignment_gain | 0.0 | 0.0 | 0.0 |

### 観察

**hivc_d条件の改善点:**
- expert_match_rate が最高 (0.381) — 専門家選択との一致率が良い
- route_choice_accuracy が最高 (0.778) — 経路選択精度が良い
- cross_role_evidence_use が最高 (0.537) — 他役割の証拠活用が活発
- unanswered_question_rate が最低 (0.030) — 質問未回答率が大幅改善
- vote_revision_rate が最高 (0.991) — 投票改訂が活発

**懸念点 (次期改善対象):**
- v_proposal_rate が全条件で NaN (0/0) — V提案プロンプトがトリガーされていない
- v_star_acceptance_rate が hivc_d で 0.0 (0/6) — V*受容が発生していない
- v_alignment_distance が全条件で 0.0 — V整列距離が測定されていない
- invalid_discussion_output_rate が hivc_d で最高 (0.932) — JSON解析失敗率が高い
- question_answer_rate が全条件でほぼ0 — 質問への回答率が極めて低い

### 結果ファイル
- ローカル: hivc_sim/results/turn_game/downloads/2026-07-20-experiment/
  - all_games.csv, control_games.csv, consulting_games.csv, hivc_d_games.csv
  - summary.csv, value_manifest.json, stream.jsonl

## 2026-07-20: 100x3分析に基づく測定・対話・実験preflight是正

### 実装内容

- `scripts/llm_turn_game_common.py`
  - before/afterのV測定テンプレートから、完成済み一様0.2重みと固定 `action_before=A` を除去した。
  - 5基準オントロジーとJSON-only契約を維持しつつ、ROLE/PERSONA初期値と現在観測から現在の優先度を導出し、プレースホルダーや例示値をコピーしないよう明示した。
  - 情報要求の質問に限り `action=null` を許可した。非質問は引き続きA-F必須とした。
  - 質問の `addressed_to` がnull/省略の場合だけ、二者ゲームで一意な相手へ補完する。未知名、辞書、数値などは補正せずJSON契約違反として拒否する。既存の `reply_to_message_id` 型検査は維持した。
  - 回答用JSON例も非質問契約に合わせて `addressed_to=null` とし、回答先は `reply_to_message_id` だけで表すよう統一した。V測定テンプレートは共通5基準をすべて列挙し、単一の抽象キーを実在キーと誤認しない形にした。
- `scripts/validate_experiment_preflight.py` / `scripts/workflow_cli.py`
  - smoke CSVの実測値に対して、一様Vコピーなし、V整合必須ターン、V提案分母、受諾V*、invalid discussion rate 10%未満を個別診断する再利用可能validatorと `uv run validate-smoke` を追加した。
  - 2ゲーム以上の新規GPU runは合格する `--smoke-run-id` を要求し、GPU上でvalidatorが終了コード0を返すまで開始しない。
  - V固有gateを使わないlegacy実験は `--scientific-gate not-applicable` / `--applicability not-applicable` の明示時だけ対象外にできる。
- `scripts/qwen_two_agent_experiment.py`
  - 直列runは既存成果物を含む出力先と既存streamを拒否し、streamを排他的に新規作成するよう変更した。
  - `run_metadata.json` にrun ID、git commit、開始/完了状態、CSV、stream、value manifestのハッシュとサイズを記録するようにした。
  - workflowのrun bootstrapとの互換性、および既存の並列worker/shard merge/resume動作は維持した。
- `hivc_sim/tests/`
  - Vプロンプトの非コピー性とbefore/after対称性、質問action/宛先契約、run_one_gameの安全な相手ルーティング、preflight pass/fail/legacy対象外、run分離・stream再利用拒否・metadataを追加した。

### 検証

- `uv run pytest hivc_sim/tests -q`: 169 passed
- 対象Pythonファイルの `python3 -m py_compile`: 成功
- `git diff --check`: 成功
- GPU実験は実行していない。
- commit/pushは実行していない。

## 2026-07-21: Devin API v3 二者実験バックエンド

### 実装内容

- `scripts/devin_api.py`
  - 標準ライブラリだけで Devin API v3 organization session のcreate/get/message pagination/send/archive/terminateを扱うclientを追加した。
  - `DEVIN_API_KEY` と任意の `DEVIN_ORG_ID` だけを認証源とし、org未指定時は `GET /v3/self` で安全に解決する。
  - 429/5xx backoff、timeout、request correlation ID、sanitized error、session status lifecycleを実装した。
- `scripts/llm_turn_game_common.py`
  - 既存Qwen経路を既定のまま保ち、agent名付きprompt runnerを `run_one_game` に局所的に依存注入できるようにした。
- `scripts/devin_two_agent_experiment.py`
  - 各 `(seed, condition)` でalpha/betaの独立Devin sessionを必ず2つ作り、ゲーム・条件間で再利用しないrunnerを追加した。
  - strict JSON + request correlation、cursor以後の新規message poll、malformed/stale retry、ACU上限、terminate+archive cleanupを実装した。
  - 既存CSV、metrics、condition randomization、Role/Persona/Value、`value_manifest`を再利用し、private promptを含まない `devin_provenance.json` を分離した。
  - network callなしのdry-runと、`GET /v3/self` だけを行うenvironment validationを追加した。
- `configs/devin_experiment.yaml` / `pyproject.toml`
  - 3条件×1ゲーム、seed=42、soft_value、保守的timeout/poll/retry/ACU上限、ローカル出力先を追加した。
  - `devin-experiment` と `devin-validate-env` entry pointを追加した。
- `README.md` / `document/run-experiments/Devin_API実験ガイド.md`
  - 既定smokeが6sessionを新規作成する有料のDevin-only実験であり、過去のQwen/GPU runと科学的に比較できないことを明記した。
  - `scripts/local_preview.py --downloads-dir hivc_sim/results/turn_game/devin/runs` によるローカル閲覧手順を追加した。
- `hivc_sim/tests/test_devin_experiment.py`
  - secret非露出、org discovery、HTTP retry、message pagination、session独立性と非再利用、private routing、strict JSON/correlation、dry-run、validation、manifest/preview互換をmockで検証した。

### 検証

- `uv run pytest hivc_sim/tests -q`: 181 passed
- 変更Pythonファイルの `python3 -m py_compile`: 成功
- `git diff --check`: 成功
- 提供されたcredentialを永続化せず、一時環境変数から `uv run devin-validate-env` を実行: `GET /v3/self` 認証成功、organization API readyを確認
- 有料Devin API実験は実行していない。
- GPU/Qwen実行経路は実行していない。
- Devin run成果物ディレクトリを `.gitignore` に追加し、API keyをシェル履歴へ残しにくい対話入力手順へ文書を更新した。
- 実API smokeで、session作成直後の初期プロンプト実行中にmessageを送るとHTTP 400になることを確認したため、相関ID付きready応答をpollしてから次のmessageを送る初期化barrierを追加した。
- Devin API smoke `devin-api-smoke-20260721-retry1` を実行し、`control / consulting / hivc_d` 各1ゲーム、alpha/beta計6sessionを完了・archiveした。
  - `hivc_d`: 3ターン、`loss_power`。V提案率/受諾率1.0、未解決V率0.0、V*行動整合率0.333。
  - `control`: 3ターン、`loss_power`。
  - `consulting`: 4ターン、turn 2で最適行動Bを採用後、`loss_hull`。平均regretは3条件中最小。
  - 98 exchanges、837 HTTP requests。完了直後の再GETでも全6sessionの `acus_consumed` は0.0とAPI報告された（無料保証ではなく、請求反映仕様は別途確認が必要）。
  - `uv run pytest hivc_sim/tests -q`: 181 passed。
- commit/pushは実行していない。

## 2026-07-21: Devin API 3条件・合計90ゲームrun

- `configs/devin_experiment_90games.yaml` を追加した。
- `control / consulting / hivc_d` を各30ゲーム（seed 42〜71）、計90ゲーム・180独立sessionとして実行する。
- 1session上限1 ACU、run理論上限180 ACUとする。
- 直近3条件smokeの実測から、直列所要時間は約59時間を見込む。
- run ID `devin-api-90games-20260721` をバックグラウンド起動した。
  - experiment PID: `10262`
  - `caffeinate -dimsu -w` PID: `10263`
  - ログ: `/tmp/hivc-devin-90games-20260721.log`
  - 起動時manifest: `status=running`, `planned_session_count=180`

## 2026-07-21: Z.ai GLM-4.7-Flash直接APIバックエンド

### 実装内容

- `scripts/zai_api.py`
  - Z.ai一般推論endpoint `/api/paas/v4/chat/completions` を標準ライブラリだけで呼ぶclientを追加した。
  - 認証は `ZAI_API_KEY` 環境変数だけから取得し、clientのserializeを拒否する。
  - `glm-4.7-flash`、非stream、JSON mode、`do_sample=false` を利用できるようにした。
  - HTTP 429/5xxと一時的な高頻度・rate limit・混雑API codeのbackoff retryを実装した。
  - API responseの `prompt_tokens / completion_tokens / cached_tokens / total_tokens` を型検証して取得する。
- `scripts/zai_two_agent_experiment.py`
  - Devin sessionやGPUを使わず、既存 `run_one_game` へspeaker-aware Z.ai prompt runnerを注入する実験runnerを追加した。
  - モデルを `glm-4.7-flash` に固定し、seedごとの条件順、Role/Persona/Value、既存CSV/metrics/value manifestを再利用する。
  - private promptを保存せず、条件・seed・agent・用途別token usage、finish reason、HTTP status/retryを `zai_provenance.json` に記録する。
  - APIの実測completion token数を共通ゲームの議論token budgetへ渡す `UsageAwareTokenizer` を追加した。
  - 各 `(seed, condition)` 完了後にCSV、manifest、provenance、value manifestをcheckpointし、途中失敗でも完了済み結果を保持する。
  - run全体の `max_total_tokens` safety limit、networkなしdry-run、実model/JSON modeを1 callで検証する `zai-validate-env` を追加した。
- `configs/zai_experiment.yaml` / `configs/zai_experiment_90games.yaml`
  - 3条件×各1ゲームのsmokeと、3条件×30 seedの合計90ゲーム設定を追加した。
- `pyproject.toml` / `.gitignore`
  - `zai-experiment` と `zai-validate-env` entry pointを追加し、Z.ai run成果物をGit対象外にした。
- `README.md` / `document/run-experiments/ZAI_API実験ガイド.md`
  - API keyの対話入力、環境検証、smoke、90ゲーム、rate limit、checkpoint、ローカルpreview手順を追加した。
  - Z.ai結果はモデルと推論serviceが異なるため、過去Qwen/GPU結果と直接比較しないことを明記した。
- `hivc_sim/tests/test_zai_experiment.py` / `test_workflow_cli.py`
  - secret非露出、一般API payload、JSON mode、retry、usage集計、議論budget連携、dry-run、実model validation、manifest/checkpoint/preview互換、CLI登録をmockで検証した。

### 検証

- `uv run pytest hivc_sim/tests -q`: 189 passed
- `uv run python -m py_compile scripts/zai_api.py scripts/zai_two_agent_experiment.py`: 成功
- `git diff --check`: 成功
- 90ゲームdry-run: `planned_games=90`, seed 42〜71, `model=glm-4.7-flash`, `network_calls=0` を確認
- Z.ai API keyが未提供のため実API callとGPU実験は実行していない。
- commit/pushは実行していない。

### 追記: 条件並列実行

- `--parallel-conditions` を追加し、同一seedの `control / consulting / hivc_d` を最大3workerで並列実行できるようにした。
- provenance recorderをlockで保護し、並列completionとHTTP traceのtoken集計を安全にした。
- 並列中に一条件が失敗しても、成功した条件をCSV/manifest/value manifest/provenanceへcheckpointしてから失敗を報告する。
- 3workerがbarrierへ同時到達するmockテストを追加し、見かけ上ではなく実際に条件が並列実行されることを検証した。
- 初回実API並列smokeは57 completion後に `HTTP 429 / API code 1302` で終了した。全57応答が `finish_reason=length` で、GLM-4.7系列の既定thinkingが256 tokenを使い切っていた。
- Z.ai公式仕様に従い `thinking.type=disabled` を追加し、無料tier向けに3条件worker間で共有する `api_concurrency: 1` semaphoreを追加した。条件処理は並行のまま、同時HTTP requestを1本に制限する。

### 実API並列smoke結果

- run ID: `zai-flash-parallel-smoke-20260721-retry1`
- 実行時間: 2026-07-21 09:53:36〜10:20:43 JST（約27分7秒）
- 構成: seed 42、`control / consulting / hivc_d` 各1ゲーム、3 condition worker、API concurrency 1、thinking disabled
- API: 180 calls、HTTP 200が180件、transport retry 2回、API error codeなし、全180 responseが `finish_reason=stop`
- token: input 354,117、output 25,621、cache 51,079、total 379,738
  - control: 63 calls / 115,402 total tokens
  - consulting: 64 calls / 144,942 total tokens
  - hivc_d: 53 calls / 119,394 total tokens
- control: 5 turns、`loss_hull`、terminal score -445、mean regret 195.125
- consulting: 5 turns、`survived_timeout`、terminal score 15、mean regret 122.75
- hivc_d: 4 turns、`loss_power`、terminal score -130、mean regret 90.521
  - accepted V*: 3/4 turn。turn 1は `v_negotiation_budget_exhausted`。
  - accepted turnのV*/action consistencyはturn 0のみtrue、turn 2/3はfalse。
- invalid discussion output rateはcontrol 7.4%、consulting 11.1%、hivc_d 10.5%。consultingとhivc_dは10% gateをわずかに超えるため、90ゲーム本実験前に修復promptまたはschema調整が必要。
- 成果物: `hivc_sim/results/turn_game/zai/runs/zai-flash-parallel-smoke-20260721-retry1/`
- API keyは実行processの環境変数だけで使用し、終了時にunsetした。成果物・設定・ログには保存していない。

### 追記: Z.ai条件スケジューリングの公平化

- 同一seedの条件開始順を、実験seedから作る基準順序の循環Latin squareで決定するようにした。3条件×30 seedでは各条件が各開始位置をちょうど10回ずつ占める。
- `api_concurrency: 1` かつ条件並列時は、共有semaphoreの非決定的な再獲得を廃止し、active条件間でAPI requestを1件ずつ決定的round-robinするschedulerを追加した。短い条件は完了時にschedulerから外れるため残り条件を妨げない。
- manifest/dry-runへ `effective_api_concurrency`、実行schedule mode、条件順戦略、seed別開始順、request interleave有無を記録し、「condition worker並列」と「API同時実行数」を区別できるようにした。
- 90ゲーム設定は3 condition worker + 1 API slotのrequest-level round-robinを使うようにした。
- `uv run pytest hivc_sim/tests/test_zai_experiment.py -q`: 11 passed。
- commit/pushは実行していない。

### 追記: Z.ai smokeログから判明したV・発話監査不整合の修正

- V*が `accepted` へ遷移した全経路で、古い `v_star_failure_reason` と `v_star_unresolved_reason` を同時に消去するよう修正した。
- `verify_vote_v_star_consistency` を最上位criteriaの単語包含判定から、受諾V weights、現在状態、ゲーム内の即時行動効果を使う決定的判定へ変更した。
  - reason内で採用すると述べたactionとJSON actionが矛盾する場合は不整合とする。
  - pod準備（Action E）は主効果が共通V ontology外のため `None`（評価不能）とする。
  - 自力脱出（Action F）は現在状態が脱出条件を満たす場合に限り整合性を決定評価する。
- discussion promptへcanonicalなAction A-Fの効果を追加し、質問以外のspeech actではactionを必須と明記した。
- `invalid_discussion_outputs` に `agent`、`attempt`、`validation_reason`、`raw_output` を追加し、既存の監査フィールドも後方互換のため維持した。
- 回帰テストとして、counter合意後の古い未解決理由消去、行動効果ベースのV整合判定、理由内action矛盾、ontology外Action E、invalid JSON監査フィールド、promptのaction契約を追加した。
- `uv run pytest hivc_sim/tests -q`: 194 passed。
- `python3 -m py_compile scripts/llm_turn_game_common.py scripts/zai_api.py scripts/zai_two_agent_experiment.py`: 成功。
- `git diff --check`: 成功。
- 90ゲームZ.ai dry-runで `planned_games=90`、各条件が各開始位置を10回ずつ占めること、`effective_api_concurrency=1`、request-level interleave有効を確認した。
- 実API実験、GPU実験、commit、pushは実行していない。

### 追記: 3独立Z.aiアカウントによる条件並列とsleep交絡の検証

- credential値を設定・成果物へ保存せず、`api_key_envs` に列挙した環境変数名だけを扱うmulti-key client poolを追加した。
- `control / consulting / hivc_d` を3つの独立API key slotへ割り当て、seedごとに循環させることで、30 seedでは各条件が各アカウントを10回ずつ使うよう反平衡化した。
- manifestへkey数、環境変数名、seed別条件割当、key別adaptive pacing状態を追加した。API key本文は保存しない。
- global concurrencyとkey単位concurrencyを分離し、`api_concurrency=3 / api_concurrency_per_key=1` で3条件を真に並列化できるようにした。
- 初回3-key run `zai-flash-multikey-smoke-20260721-1` は失敗したが、macOS power logとの照合でrun中の12:13:55にClamshell Sleepへ入り、通常復帰が13:04:29だったことを確認した。通信断、DarkWake、復帰時retryが混在するため、このrunから共有IP制限を断定しない。
- `caffeinate -dimsu` 配下でawake再試験 `zai-flash-awake-multikey-smoke-20260721-1` を実施した。
  - 13:19:07〜13:53:47 JST、34分40秒、3条件すべて完了。
  - 181 completionすべてHTTP 200で取得。追加attemptとして429が8件、transport errorが4件発生したが、adaptive retryで全て回復した。
  - 実行中のSleep/Wakeは0件。実効API concurrencyは3。
  - control / consultingは5ターンで`survived_timeout`、hivc_dは4ターンで`loss_power`。
  - V schema completenessは全条件1.0、accepted V*に古いfailure/unresolved reasonが残る行は0件。
  - invalid attempt監査20件は全てagent、turn、attempt、validation reason、raw、token、recovered/final exhaustedを保持した。
- `uv run pytest hivc_sim/tests -q`: 199 passed。
- commit/pushは実行していない。

### 追記: invalid discussion retryのattempt単位監査

- JSON契約違反が後続retryで修復された場合も、途中のinvalid attemptを `invalid_discussion_outputs` へ保存するようにした。
- 各attemptへ `agent` / `speaker`、`turn`、`opportunity`、1始まりの `attempt`、`max_attempts`、`validation_reason`、`raw` / `raw_output`、`raw_payload`、`token_count`、`recovered`、`final_exhausted` を記録する。
- `invalid_discussion_output_count` は従来どおり「全retry後も無効だった論理発話数」とし、後方互換性を維持した。
- ターン行へ `invalid_attempt_count` と `repaired_invalid_output_count` を追加した。
- summaryへattempt基準の `invalid_discussion_attempt_rate` と、論理発話基準の `discussion_repair_success_rate`、各分子・分母countを追加した。
- 修復成功経路とretry全失敗経路の回帰テスト、およびsummary集計テストを追加した。

### 追記: Z.ai 90ゲーム実行時間短縮

- 固定8秒間隔を廃止し、全condition workerで共有するadaptive rate limiterを追加した。
  - 通常は2.5秒から開始し、成功12回ごとに0.25秒短縮して下限2秒へ近づける。
  - `1302 / 1303 / 1305` またはHTTP 429時は全workerを60秒cooldownし、要求間隔を最大12秒まで増やす。
  - 一時的なtransport errorと5xxも最大4回まで外側で再試行する。
- consulting / HIVC-Dの固定手順を意味を保ったcompact版へ置換する `zai-compact-v1` prompt profileを追加した。
  Role、観測状態、会話履歴、V/JSON契約は削除しない。
- Z.aiの `max_new_tokens` を256から192へ変更した。
- manifestをv3へ更新し、adaptive設定・実行時rate state・prompt profile・max tokenを記録する。
- compact-v1は従来のfull prompt Z.ai runとは別実験系列として扱う。
- `uv run pytest hivc_sim/tests -q`: 198 passed。
- `python3 -m py_compile` と `git diff --check`: 成功。
- 90ゲームdry-runでadaptive設定、`zai-compact-v1`、`max_new_tokens=192`、90 planned gamesを確認した。
- 親スレッドで開始済みのsmoke processには触れず、新設定での実API実験、commit、pushは実行していない。

### 追記: Z.ai max_new_tokens=256 再smoke

- `configs/zai_experiment.yaml` と `configs/zai_experiment_90games.yaml` の `max_new_tokens` を192から256へ戻した。
- `caffeinate -dimsu` 配下で3独立API key・3条件並列smoke `zai-flash-awake-multikey-256-smoke-20260721-1` を実行した。
  - 2026-07-21 14:02:39〜14:47:25 JST、44分46.553秒、3条件すべて完了。
  - 167 completionを取得し、HTTP 200は167件。追加attemptとしてHTTP 429 / API code 1302が9件、transport errorが9件発生したが、adaptive retryで全て回復した。
  - `finish_reason=length` は2/167（1.20%）。192-token smokeの31/181（17.13%）から大幅に低下した。
  - invalid discussion attemptは1/77（1.30%）。最終的なinvalid discussion outputは3条件とも0件。
  - V schema completenessは3条件とも1.0。accepted V*に古いfailure/unresolved reasonが残る行は0件。invalid attempt監査1件の必須フィールド欠落も0件。
  - controlは4ターンで`loss_power`、consultingは5ターンで`survived_timeout`、hivc_dは4ターンで`loss_power`。1 seedだけのため条件間の成績差は評価しない。
  - tokenはinput 330,731、output 25,294、cache 67,700、total 356,025。
  - 実行時間帯のmacOS power logにSleep/Wakeはなく、スリープ交絡は確認されなかった。
- 192-token smokeより約10分7秒長かった主因は、consulting割当キーを中心とするrate-limit cooldownであり、256-token化だけによる遅延とは切り分けられない。
- commit/pushは実行していない。

### 追記: 貸出サーバー向けZ.ai 3条件x10ゲーム設定

- `configs/zai_experiment_server_10games.yaml` を追加した。
- `control / consulting / hivc_d` 各10ゲーム（seed 42〜51）、計30ゲームをGLM-4.7-Flashで実行する。
- 3つの独立アカウントキーはseedごとに条件へ反平衡割当する。
- 貸出サーバーIPから連続した環境検証を行うと、1キー成功後に別キーでもHTTP 429 / code 1305が発生した。このため送信元IP単位の制限も想定し、condition workerは並行のままHTTP requestを全体で1本に直列化した。
- provider throttle時は終了せず長時間待機できるよう、adaptive retry上限を1000回、cooldownを120秒とした。
- API key本文は設定・ログ・SSHコマンドラインへ保存しない。
- 貸出サーバー `student222@172.16.51.202:2222` の `/home/student222/hivc-d-verification-zai` へ作業ツリーを同期し、uv 0.11.26とlock済み依存環境を構築した。
- run ID `zai-server-3x10-20260721-1` をtmux session `hivc-zai-30` で2026-07-21 15:12:05 JSTに起動した。
  - manifest: `status=running`, `planned_games=30`, `games_per_condition=10`, `api_key_count=3`, `effective_api_concurrency=1`, `adaptive_max_retries=1000`, `max_new_tokens=256`
  - remote log: `/home/student222/hivc-zai-30-20260721.log`
  - API keyは標準入力から起動shellの環境へ渡し、起動後にtmux global environmentから削除した。process command line、設定、ログにはkey本文を含めていない。

### 追記: GLM-4.7-FlashX 1条件x1ゲーム時間測定

- 無料Flash run `zai-server-3x10-20260721-1` はユーザー指示により最初のcheckpoint前に停止し、manifestを `failed / user_requested_stop_before_first_checkpoint` として確定した。
- `scripts/zai_two_agent_experiment.py` の科学的一貫性allowlistへ `glm-4.7-flashx` を追加し、既定の `glm-4.7-flash` は維持した。
- 有料General API用の `configs/zai_flashx_smoke.yaml` を追加した。
  - `hivc_d` 1ゲーム、seed 42、max_new_tokens 256、thinking disabled、API concurrency 1。
  - `glm-4.7-flashx` 以外の未対応モデルは引き続き拒否する。
- 貸出サーバーでrun `zai-flashx-hivcd-timing-20260721-1` を実行した。
  - 実測時間: 748.31秒（12分28.31秒）。manifest差分は748.071秒。
  - 4ターン、47 completion、全47件がHTTP 200かつ `finish_reason=stop`。
  - HTTP 429 / Z.ai business errorは0件。60秒transport timeoutが6件あり、adaptive retryですべて回復した。
  - token: prompt 92,516、completion 6,799、cached 8,629、total 99,315。
  - 公式単価での推定費用は約$0.009（1セント未満）。
  - invalid discussion output/attemptは0、V schema completenessは1.0。
  - 結果は `loss_power`、terminal score 190。時間測定smokeのため条件効果の評価には使わない。
- `uv run pytest hivc_sim/tests/test_zai_experiment.py -q`: 16 passed。
- `python -m py_compile`、dry-run、`git diff --check`: 成功。
- commit/pushは実行していない。

### 追記: FlashX対話分析に基づく意味的安全ゲート

- `hivc_sim/turn_game.py` に `preview_action_safety` を追加した。ターン開始時消費、現在イベント、行動の固定効果、脱出条件を `step` と同じ順序で反映し、確率分岐を漏らさず確実な即時敗北だけを事前検出する。
- 最終投票をシステム側で検証し、確実な敗北行動または受諾V*との不整合を検出した場合は同じagentへ修復プロンプトを再送するようにした。再試行後も無効な投票は採用せず、安全かつV*整合なsystem fallbackを使用する。
- `v_star_consistent` のモデル自己申告を最終投票JSONから削除した。旧出力に含まれる場合も監査互換として解析するが、整合性判定には使用しない。
- V*行動整合性へ資源危険閾値のurgencyと、communication=3到達による救助経路開始価値を追加した。確実な即時敗北行動は常に不整合とする。
- Roleの非対称観測について、現在イベント・危険閾値・救助/脱出経路から判断関連fieldを抽出するようにした。相手だけが観測でき、まだ明示共有されていない場合は `information_request` と `requested_fields` を要求し、通常発話で無視した場合はJSON修復経路で再試行する。
- 同一 `proposal_id` が異なる内容で再利用された場合、speaker・message index・semantic hashから決定論的な一意IDへ修復する。対応するself-acceptも新IDへ付け替え、交渉予算をID衝突だけで失わないようにした。
- ターンCSVへ `final_vote_retry_count`、`rejected_final_votes`、`safety_override_used`、`v_proposal_id_repairs`、`required_information_question_count` などの監査列を追加した。
- 安全preview、自己申告非依存V判定、投票再試行、V整合fallback、proposal ID修復、非公開重要情報への質問強制と回答閉包の回帰テストを追加した。
- `pytest hivc_sim/tests -q`: 208 passed。
- `python3 -m py_compile hivc_sim/turn_game.py scripts/llm_turn_game_common.py`: 成功。
- `git diff --check`: 成功。
- GPU/API実験、commit、pushは実行していない。

### 追記: GLM-4.7 flagship比較smoke

- Z.ai runnerのmodel allowlistへ `glm-4.7` を追加し、FlashX smokeとの差分がmodel名だけになる `configs/zai_glm47_smoke.yaml` を追加した。
- dry-runで `glm-4.7 / hivc_d / seed 42 / 1 game / thinking disabled / max_new_tokens 256` を確認し、YAML差分が `model: glm-4.7-flashx -> glm-4.7` だけであることを検証した。
- run `zai-glm47-hivcd-smoke-20260721-1` を実行した。
  - 2026-07-21 17:30:38〜17:46:37 JST、959.73秒（15分59.73秒）。
  - 4ターン、58 completion、全58件がHTTP 200かつ `finish_reason=stop`。transport timeout相当12件ではなく4件で、全てretry回復した。rate-limitは0件。
  - tokenはprompt 115,807、completion 7,589、cached 26,880、total 123,396。公式単価による概算費用は約$0.073。
  - turn 1はsignal_windowでCを選び、accepted V*と行動が整合し、best action一致・regret 0だった。
  - turn 2は危険なAを10回拒否し、system fallbackがbest action Bを選んだ。
  - V*は1/4ターンaccepted、3/4ターンunresolved。最終的にturn 3のhull_fractureで `loss_hull`。
- 同一seedでもAction B/Cの確率分岐とevent samplingが同一RNGを共有するため、FlashXとGLM-4.7でevent列が分岐した。このrunの勝敗・平均regretをモデル性能差として直接比較してはならず、モデル比較にはevent RNGとaction-effect RNGの分離または固定event列が必要と判明した。
- 会話品質ではFlashXより正確なI/V/A理由付けが1ターン確認できた一方、Action記号の意味混同、oxygen=2を「余裕」とする誤認、同じ危険行動の反復、V交渉budget内での未合意が残った。
- `pytest hivc_sim/tests -q`: 209 passed。
- `python3 -m py_compile scripts/zai_two_agent_experiment.py`: 成功。
- `git diff --check`: 成功。
- commit、pushは実行していない。

### 追記: Action理解・予測状態・V交渉・回避不能イベントの修正

- `hivc_sim/turn_game.py` にAction固定効果、ターン開始消費、イベント効果のcanonical説明を追加し、プロンプト側の説明を実際の `step` と共有するようにした。
- V測定、自由議論、最終投票、必須V提案、V応答の全プロンプトへ、A〜FのAction一覧、`oxygen -1 / power -1` のターン開始消費、現在イベント効果、全Actionの確定的な予測 `state_after` を追加した。25%事故は未確定リスクとしてAction一覧に残し、予測値には混入しない。
- 予測状態はRoleの `observation_scope` を維持し、非可視fieldを `hidden` として表示する。イベントが非可視の場合も現在イベントの固有効果は開示しない。
- 最終投票の安全性修復フィードバックへ、拒否Action、`state_before -> turn_start -> event -> action` の計算順序、予測state_after、安全候補Actionを追加した。同じ危険Actionを理由不明のまま反復しにくくした。
- HIVC-Dのcounterを既定で1ラウンドに制限した。counter発生時は必須V交渉予算を4発話から6発話へ動的延長し、2回目以降のcounterは監査上 `counter_round_limit_reached` として拒否する。CSVへcounter数、拒否counter数、実効交渉予算、使用発話数を追加した。
- イベント抽選に `sample_viable_event` を追加した。予定イベントが全Actionを確実な即時敗北にする場合だけ、安全Actionが1つ以上残るイベントへ元の確率比で再抽選する。イベント以前から回避不能なら追加損傷のない `none` を採用する。Action体系A〜Fは変更していない。
- 全プロンプトのAction支援ブロック、修復計算内訳、counter上限制御・動的予算、船体亀裂の回避不能状態除外について回帰テストを追加した。
- `pytest hivc_sim/tests -q`: 215 passed。
- `python3 -m py_compile`、`git diff --check`: 成功。
- GPU/API実験、commit、pushは実行していない。

### 追記: Action予測・V交渉修正後のFlashX 1ゲームsmoke

- `glm-4.7-flashx / hivc_d / seed 42` の1ゲームをrun `zai-flashx-system-fixes-smoke-20260721-1` として実行した。
- 2026-07-21 18:10:38〜18:19:35 JST、536.59秒（8分56.59秒）。4ターンで最終結果は `win`、mean regretは55.72925だった。
- tokenはprompt 140,327、completion 7,742、cached 19,460、total 148,069。57 completionのうち55件が `finish_reason=stop`、2件が `length`。HTTP 200は57件、60秒transport timeout相当が2件あり、runは回復して完了した。rate-limit eventは0件。
- counterが出たturn 0/2では交渉予算が4から6へ動的延長され、いずれも3発話でV*を受諾した。turn 1はcounterなしの4発話を使い切り未解決だった。
- turn 3では予測表を踏まえ、モデルがAction A後のoxygen=3と、B/C/D/Eの即時敗北を認識した。Aで救助到着まで生存し勝利したため、ターン開始消費・イベント効果・予測state_after提示の有効性を確認できた。
- 新たなblockerとして、`verify_vote_v_star_consistency` が安全候補だけでなくunsafe Actionもutility最大値の比較対象に含め、かつturn開始消費・イベント反映前のstateでAction効果を評価していることが判明した。turn 2ではBを12回、turn 3では唯一安全かつbestのAを4回 `v_star_action_inconsistent` として拒否し、system fallbackでD/Aを採用した。90ゲーム前にV整合検証を予測state_after基準へ修正する必要がある。
- invalid discussion attemptは1件で、required field付き質問への修復retryにより回復した。全4ターンで質問は回答済みだった。
- commit、pushは実行していない。

### 追記: V整合検証のunsafe比較・turn-start基準バグ修正

- `hivc_sim/turn_game.py` に `preview_turn_start_state` を追加し、ターン開始消費と現在イベントを反映したAction比較の共通基準状態を公開した。
- `verify_vote_v_star_consistency` のAction utilityを、raw `state_before`への手書き効果ではなく、`preview_turn_start_state` と各Actionの `preview_action_safety` による予測 `state_after` の差分から算出するように変更した。
- V整合性の最大utility比較対象を、安全性検証に通ったActionだけへ限定した。unsafe Actionが高いutilityを持つことで唯一安全なActionを拒否する問題を解消した。
- 共通V ontology外の主効果を持つAction Eは従来どおり三値判定の `None` とし、utility比較対象から除外した。B/Cの25%事故は予測stateへ混入させず、従来どおり期待損失として加味する。
- smoke turn 2の `relay_short` 状態でBが整合と判定されること、turn 3の `backup_power_found` 状態で唯一安全なAが整合、unsafeなBが不整合になることを回帰テスト化した。
- `pytest hivc_sim/tests -q`: 217 passed。
- `python3 -m py_compile`、`git diff --check`: 成功。
- API/GPU実験、commit、pushは実行していない。

### 追記: V整合検証修正後のFlashX再smoke

- `glm-4.7-flashx / hivc_d / seed 42` をrun `zai-flashx-v-consistency-fix-smoke-20260721-1` として1ゲーム実行した。
- 2026-07-21 18:28:28〜18:56:33 JST、1684.52秒（28分04.52秒）。5ターン、最終結果は `survived_timeout`、mean regretは109.7082だった。
- tokenはprompt 191,358、completion 9,638、cached 24,562、total 200,996。78 completion中77件が `stop`、1件が `length`。HTTP 200は78件、60秒transport timeoutが10件、HTTP 500が2件あり、adaptive retryで完了した。rate-limit eventは0件で、速度評価には不適なrunだった。
- 修正対象の「unsafe Actionをutility最大値に含め、安全Actionを誤拒否する」症状は再発しなかった。turn 3ではunsafeなA/Cを詳細な計算内訳付きで拒否し、唯一安全なBへ修復して通常合意、best action一致・regret 0となった。
- turn 0/1/3は最終的に通常合意、turn 2/4はV整合検証によるfallbackとなった。turn 2では受諾V*のoxygen 0.4 / communication 0.3に対してモデルがCを反復したが、検証器はoxygenを増やすAを最大utilityとして要求した。turn 4ではV*順位上powerがfloodingより高く、D/Cを拒否してBへfallbackした。
- 上記turn 2/4はunsafe比較バグではなく、V整合を「受諾Vと矛盾しない行動」ではなく「V重み付き即時utilityが最大または僅差の行動」に限定する現設計による。Vが合意後のActionを決めすぎるため、許容集合・Pareto改善・資源飽和を含む整合性定義への再設計が必要と判断した。
- Action Eは共通V ontology外のため `None` 判定で許容され、turn 0ではCの拒否後にEで合意した。これは誤拒否回避になる一方、EだけV制約を迂回できるため、ontology coverage上の残課題である。
- commit、pushは実行していない。

### 追記: V整合判定から数値リスク・utility強制を撤廃

- `verify_vote_v_star_consistency` から、V weights、危険度倍率、Action効果の正規化値、最大utilityとの差 `0.025` によるAction順位付けを削除した。
- V*を唯一の正解Actionを決める採点表ではなく、安全な複数案を比較・説明するための共通の熟議観点として扱うように変更した。
- システム側のV整合検証は、受諾V* IDの参照、共通V schema、reasonとactionの明示的矛盾、既知の即時安全性に限定した。安全なAction間のtrade-offはエージェントの観測・議論・合意に委ねる。
- 共通V ontology外だったAction Eの `None` 特例を廃止し、他の安全なActionと同じ手続き判定に統一した。
- 最終投票プロンプトへ「V*は数値スコアや唯一の正解Actionを定めない」と明記した。
- FlashX smokeで問題になったturn 2の通信窓Cとturn 4の浸水対処Dを、安全なtrade-offとして許容する回帰テストを追加した。
- モデル向けのV表現を数値 `weights` から `priority_levels`（`high / mid / low`）へ変更した。初期Value、V before/after測定、V提案、counter、prescribed V*に適用した。
- 既存profileの数値設定はモデルへ直接表示せず離散化する。confidence、risk tolerance、譲歩傾向、根拠要求度もモデル表示時は `high / mid / low` に変換する。
- 既存run・legacy backendの数値 `weights` は読み取り互換を維持し、新しい測定・提案プロンプトでは小数・百分率による重み付けを禁止した。
- V距離とschema completeness集計は `priority_levels` に対応し、モデルへ数値スコアを戻さず離散レベルの不一致として扱う。
- `pytest hivc_sim/tests -q`: 221 passed。
- `python3 -m py_compile`、`git diff --check`: 成功。
- GPU/API実験、commit、pushは実行していない。

### 追記: GLM-4.7 qualitative V smoke

- `glm-4.7 / hivc_d / seed 42` をrun `zai-glm47-qualitative-v-smoke-20260721-1` として1ゲーム実行した。
- 2026-07-21 20:10:59〜20:22:47 JST、約11分48秒。5ターンで `survived_timeout`、win rate 0、survival rate 1、mean regret 94.8334だった。
- 75 completion、prompt 177,097 / completion 7,999 / total 185,096 tokens。HTTP 200は75件、60秒transport timeoutが2件、HTTP 429 / API code 1302が1件あり、adaptive retryで完了した。
- `v_before`、`v_after`、V提案、counter、受諾V*の全生成記録を検査し、数値 `weights` 出力は0件だった。すべて `priority_levels` の `high / mid / low` と `confidence_level` を使用し、V schema completenessは1.0だった。
- V*成立はturn 3の1/5ターンだけで、残り4ターンはcounter後に同じagentがrejectを繰り返して交渉予算を使い切った。unresolved V rate 0.8、fallback rate 0.8だった。
- turn 3はqualitative V*を両者が受諾し、Bで通常合意、best action一致、regret 0、V*/Action consistency 1.0だった。全ターンを通じ、従来の数値utilityによる `v_star_action_inconsistent` 拒否は0件だった。
- 離散V表現への移行自体は成功したが、counter後の同一reject反復とV*成立率の低さは別の交渉状態機械課題として残った。
- commit、pushは実行していない。

### 追記: reject終端化とValue非一致下のAction調整

- mandatory V negotiationで、最新proposal IDへの有効な `reject` を `v_proposal_rejected` 終端イベントとして扱うようにした。同じagentへ同じproposalのaccept/rejectを再要求しない。
- `reject` 終端後はV*を暗黙受諾せず、`unresolved` のまま `A_CHECK` へ遷移する。予算枯渇とは区別して `v_star_failure_reason` / `v_star_unresolved_reason` / transition historyへ記録する。
- 後続の意思決定機会でも終端済みV交渉を再開せず、重複した `A_CHECK -> A_CHECK` transitionも記録しない。
- V整合交渉の発火条件を `action_before` の不一致に限定した。Role由来のpriority level差は距離指標として監査するが、それだけでV*合意を強制しない。
- V*未成立かつ2回目以降の最終投票では `ACTION_RECONCILIATION` を提示し、Value自体を一致させず、相手の根拠を取り入れて同じ安全なActionへ合意できることを明記した。
- 回帰テストで、Valueが大きく異なってもAction一致ならV交渉不要であること、Action不一致なら必要であること、有効reject後の応答が1回で終了すること、次の機会でAction通常合意へ到達することを検証した。
- `pytest hivc_sim/tests -q`: 224 passed。
- `python3 -m py_compile`、`git diff --check`: 成功。
- API/GPU実験、commit、pushは実行していない。

### 追記: reject終端化後のGLM-4.7 smoke

- `glm-4.7 / hivc_d / seed 42` をrun `zai-glm47-reject-terminal-smoke-20260721-1` として1ゲーム実行した。
- 2026-07-21 20:44:04〜20:53:58 JST、594.65秒（約9分55秒）。5ターンで `survived_timeout`、mean regret 97.25だった。
- 66 completion、prompt 160,288 / completion 7,733 / total 168,021 tokens。HTTP 200は66件、60秒transport timeoutが1件あり回復した。数値 `weights` 出力は0件だった。
- 修正前runとの同一seed比較で、completionは75→66、total tokenは185,096→168,021、実行時間は707.88→594.65秒へ減少した。
- 同一proposal IDへのreject反復は0件。turn 0/1/2の有効rejectはいずれも `v_proposal_rejected` で終端し、重複応答を要求しなかった。
- fallback rateは0.8→0.2、opportunity単位agreement rateは0.111→0.444へ改善した。turn 1は2回目にE、turn 2は3回目にC、turn 4は2回目にBへ、Valueを一致させずAction consensusへ到達した。turn 3は事前Action一致のためV交渉を発火せずBで即時合意した。
- V* acceptanceは0だったが、5ターン中4ターンでAction consensusとなった。V*成立率をAction合意品質の代理にしない設計意図どおりの結果だった。
- 残課題として、`v_alignment_required=false` のturn 3/4も `v_star_status=unresolved / missing_v_proposal` と記録され、集計上のunresolved V rateが1.0になった。交渉不要ターンは `not_required` として分母から除外する必要がある。
- commit、pushは実行していない。

### 追記: GPUサーバー上でGLM-4.7 3条件×30ゲームを開始

- commit `3ed0a94` の実験実装と、単一キーfallback設定を追加したcommit `c155d49` をGitHubへpushした。
- GPUサーバー `student222@172.16.51.202:2222` の既存dirty checkoutには触れず、専用ディレクトリ `/home/student222/hivc-d-verification-glm47-90-3ed0a94` にcommit済みtreeを配置した。
- 3つのZ.ai API keyをGLM-4.7の最小completionで個別検査した。key slot 1は成功し、slot 2/3はHTTP 429 / API code 1113で利用不可だった。このため、失敗する3アカウント並列ではなく `configs/zai_glm47_90games_single_key.yaml` により利用可能な1キーで直列実行する構成へ切り替えた。
- 2026-07-21 21:06 JST、run `zai-glm47-3x30-20260721-1` をtmux session `hivc-glm47-90` で開始した。条件は `control / consulting / hivc_d`、seedは42〜71、計90ゲーム、モデルは`glm-4.7`、各seedの条件順はLatin squareで反平衡化する。
- 起動後にmanifestの `status=running`、`planned_games=90`、`api_key_count=1`、`execution_scheduling_mode=seed_counterbalanced_sequential_conditions` と実行processの生存を確認した。API key受け渡し用の一時credential fileはworker起動直後に削除した。
- 直近smokeの約10分/ゲームから所要時間は約15時間を見込む。adaptive retryとゲーム単位checkpointを有効にし、短時間のrate limitや通信断でrun全体を終了しない設定とした。

### 追記: GLM-4.7 90ゲームを単一キー3条件並列へ切替

- 直列run `zai-glm47-3x30-20260721-1` は2/90ゲーム完了時点でSIGINTにより停止し、checkpointを保存した。旧runは削除せず、manifestは`status=failed / completed_game_conditions=2`として監査可能な状態で保持した。
- `configs/zai_glm47_90games_single_key_parallel.yaml` を追加し、1つの有料`ZAI_API_KEY`を共有する3 condition worker、global concurrency 3、per-key concurrency 3へ変更した。
- dry-runで`planned_games=90`、`condition_workers=3`、`effective_api_concurrency=3`、`seed_counterbalanced_condition_workers_bounded_parallel_api`を確認した。`pytest hivc_sim/tests -q`は224 passed。
- commit `340b13e` をpush・GPUサーバーへ同期し、2026-07-21 21:47 JSTにrun `zai-glm47-3x30-singlekey-parallel-20260721-1`をtmux session `hivc-glm47-90-p3`で開始した。
- 起動後にmanifestの`status=running / planned_games=90 / condition_workers=3 / effective_api_concurrency=3`、3つの`zai-seed-42` worker thread、credential一時ファイルの削除を確認した。
