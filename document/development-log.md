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
