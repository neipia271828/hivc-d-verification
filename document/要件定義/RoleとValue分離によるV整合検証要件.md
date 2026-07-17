# RoleとValue分離によるV整合検証要件

- 更新日: 2026-07-17
- 状態: GPU pilot診断反映済み・修正実装前
- 対象読者: 実験設計者、実装者、レビュー担当者

## 1. 目的

エージェントの専門領域・可視情報・伝え方を定める `Role` と、行動案を評価する判断基準 `Value`（以下 `V`）を分離する。

現行の `role.json` は、専門性だけでなく、`priority_weights`、`goal_focus`、`risk_tolerance`、`concession_tendency`、優先行動を指定する `notes` を同時に与える。これらが変更不可能な人格命令として働くと、HIVC-Dが想定する `V` の明示・整合・共通基準 `V*` の形成が成立しない。

本要件は、次の二点を切り分けて検証できる実験基盤を定義する。

1. 合意形成フレームワークが、個人の初期 `V` から共通基準 `V*` を形成できるか。
2. `V*` の形成が、投票、行動、regret、生存、勝利へつながるか。

本文書は要件定義である。既存実装とGPU pilotの診断結果を根拠に修正要求を定めるが、本文書の更新自体は修正実装と本実験を含まない。

## 2. 背景と現行実験の診断

### 2.1 対象run

診断の基準は `episode-20260715-210345` とする。

- Qwen3-14B
- `control` / `consulting` / `hivc_d` 各30ゲーム
- seed 42〜71を各条件で共有
- alpha=`agent_01`、beta=`agent_02` を固定
- モデル生成は `do_sample=false`

主な観測結果は次の通りである。

| 指標 | control | consulting | hivc_d |
|---|---:|---:|---:|
| 勝率 | 33.3% | 30.0% | 26.7% |
| 生存率 | 46.7% | 33.3% | 33.3% |
| 平均regret | 128.6 | 162.9 | 154.1 |
| 相手の役割情報利用率 | 47.4% | 83.6% | 84.2% |
| 意思決定機会での合意 | 9/239 | 10/236 | 20/232 |
| fallback率 | 93.2% | 91.8% | 83.3% |

HIVC-D条件は情報利用と合意を増やしたが、行動品質と成果を改善しなかった。さらに、HIVC-D条件の120ターンで次の固定化が見られた。

- alphaは安全系行動 A/D を105回（87.5%）選択した。
- betaは通信修理 C を107回（89.2%）選択した。
- 複数の意思決定機会があった76ターンで、機会間に票を変更したのは alpha 4回、beta 5回だった。
- alpha と beta の両方が電力を主要価値としないまま、電力切れは control 11ゲームに対し、consulting 16ゲーム、hivc_d 17ゲームとなった。

これは「情報 `I` は共有されたが、判断基準 `V` はほぼ変化しなかった」という仮説と整合する。ただし、現行ログは `V` そのものを記録しないため、現時点で因果は確定できない。

### 2.2 現行設計の問題

1. **専門性と規範的価値の混合**
   - `role` に、可視情報と専門性だけでなく、固定優先順位と行動命令が入っている。
2. **低い譲歩傾向の同時操作**
   - `agent_01=0.20`、`agent_02=0.25` の `concession_tendency` が、HIVC-Dの整合可能性と交絡する。
3. **静的ペルソナの再注入**
   - 各発言・投票で元の `priority_weights` と `notes` が再提示され、議論中の変更より強いアンカーになる。
4. **個人Vと共通V*の未分離**
   - 個人の初期選好を変えるのか、グループ判断だけに適用する `V*` を作るのかが明示されていない。
5. **V整合の状態と測定の欠如**
   - `v_before`、`v_proposal`、`v_star`、受諾、`v_after` が実行状態にもCSVにも存在しない。
6. **投票前選好の欠如**
   - 現行の `individual_actions` は最終投票の複製であり、議論前から議論後への変化を測れない。
7. **プロトコル側V*の固定**
   - 現行HIVC-D手順は「破局リスク → 勝利条件 → 次ターンの選択肢」を事前に与える。これ自体を共通価値とする場合は、自由なV交渉ではなく「外部から与えたV*への適応」として別にラベル付けする必要がある。

### 2.3 thermal duty cycle GPU pilotの診断

修正要求の追加根拠として `episode-20260717-202452` を用いる。

- Git commit: `f445dcab1f0e511b4a44ace7b95c51228b8e1b03`
- Qwen3-14B、`soft_value`
- seed 42、`control` / `consulting` / `hivc_d` 各1ゲーム
- GPU 0、thermal duty cycle 78℃停止 / 70℃再開
- 全shard完了、merge検査全項目合格、最高79℃、thermal slowdown 0回
- 3条件合計11ターン、実行時間約20分50秒

GPU運用と成果物結合は成立した一方、実験妥当性について次の阻害要因が確認された。

| ID | 観測 | 影響 | 判定 |
|---|---|---|---|
| P-V01 | HIVC-Dの全4ターンで `v_proposals=[]`、`v_star_failure_reason=missing_v_proposal` | H-V1/H-V2を計算できず、HIVC-D固有処理が実質未実行 | blocker |
| P-V02 | V測定結果が5項目からRole固有の2項目程度へ縮退 | RoleとValueの分離を検証できず、V距離も同一空間で比較不能 | blocker |
| P-Q01 | HIVC-D 42発話中32発話、consulting 26発話中20発話が同文反復 | 議論予算を質問再送が消費し、V提案機会を圧迫 | blocker |
| P-Q02 | 実質的な未回答質問がある一方、`unanswered_question_rate=0.0` | 質問閉包指標が会話状態を表していない | high |
| P-J01 | 壊れたJSON断片が有効な `message` として1件保存 | 発話契約違反が正常発話に混入 | high |
| P-M01 | 固定プロファイル使用時、rootの `game_profile_assignments=[]`、shardではキー欠落 | seed・条件ごとの割当追跡が不完全 | high |

本pilotはthermal duty cycleの受入証拠として使用できるが、HIVC-Dの効果推定データには含めない。

## 3. 検証する因果連鎖

本検証は次の因果連鎖を分解して測定する。

```text
Role / 可視情報
        ↓
個人の初期Vと事前行動案
        ↓
フレームワークによる I共有・V比較
        ↓
共通基準V*の提案・受諾
        ↓
V*に基づく投票・行動
        ↓
regret・生存・勝利・return
```

次を別々に判定する。

- `V*` が形成されない場合は、整合プロセスの不成立である。
- `V*` は形成されたが投票に反映されない場合は、Vと行動の接続失敗である。
- 投票は `V*` に従ったが成果が改善しない場合は、`V*` の品質または環境との適合の問題である。
- 投票も成果も改善した場合に限り、一連の改善効果を支持できる。

## 4. 用語と責務の分離

| 概念 | 定義 | 主比較での扱い |
|---|---|---|
| Role | 専門領域、責任名、可視情報、役割固有診断 | 条件間で固定 |
| Persona | 話し方、根拠要求の強さなど、価値順位と独立な行動様式 | 条件間で固定 |
| 初期V | 議論前に個人が持つ暫定的な優先基準 | 同一seed・同一agentで固定 |
| 現在V | 議論と受諾により更新され得る個人の判断基準 | 変更可能な状態 |
| V* | 個人Vの一致を要求せず、グループ行動を比較するために両者が受諾した共通基準 | 成立、受諾、行動反映を測定 |
| Framework | I共有、V整合、A確認を進める共有手順 | 主たる操作変数 |
| 交渉特性 | 譲歩傾向、支配性、合意志向など | 初期主比較で中立化。将来の独立因子 |

Roleは「何を知るか」を主に定め、Vは「知った事実をどの基準で比較するか」を定める。両者を同じフィールドや自然文に混在させない。

## 5. プロファイル要件

### 5.1 論理分離

実装上の保存ファイル数にかかわらず、解決後の内部表現は次の三領域を持つ。

```yaml
role:
  id: safety_operator
  label: 安全管理担当
  expertise_domains: [oxygen, hull_damage, flooding]
  observation_scope: [oxygen, hull_damage, flooding]
  responsibility: 担当領域の観測事実とリスクをチームに正確に伝える

persona:
  communication_style: skeptical
  evidence_demand: 0.85

value:
  id: safety_first_soft
  version: "1.0"
  initial_priority_weights:
    oxygen: 0.35
    power: 0.10
    hull_damage: 0.25
    flooding: 0.25
    communication: 0.05
  confidence: 0.60
  negotiable: true
```

### 5.2 Roleに記載できる内容

- 専門領域と役割名
- 可視な状態・診断の範囲
- 担当情報を正確に共有する責任
- 行動の実行可能性を評価する専門的制約
- 勝敗条件を改変しない説明

### 5.3 Roleに記載しない内容

- 「必ず」「最優先する」「反対する」などの固定行動命令
- 定量的な優先重み
- 特定の勝ち筋を常に優先する `goal_focus`
- 優先行動を規定する `notes`
- 譲歩しないことを役割上の義務とする記述

### 5.4 Valueの要件

- `initial_priority_weights` は行動命令ではなく、議論開始時の暫定的事前分布とする。
- プロンプトには「観測事実、相手の根拠、共通基準V*により更新可能」と明記する。
- `confidence` は初期Vへの確信度であり、初期実験では全agentで同一値に固定する。
- `negotiable=false` は選択可能にする場合も主比較と混在させず、`hard_value` の感度分析として扱う。
- 重みは必ず同一尺度・同一項目集合で正規化し、合計0や負値を拒否する。
- 数値重みが過度のアンカーになるかを調べるため、順位のみを与える表現を感度分析として持てるようにする。

#### 5.4.1 共通Vオントロジー

主比較で用いる数値Vは、Roleや観測範囲にかかわらず、実験設定で固定した同一の `value_criteria_schema` 上に表現する。

```yaml
value_criteria_schema:
  id: submarine-survival-v1
  version: "1.0"
  criteria: [oxygen, power, hull_damage, flooding, communication]
```

- `v_before`、`v_after`、`v_proposal`、`v_star` の `weights` は、`criteria` と完全に同じキー集合を持つ。
- `ordered_criteria` は全criteriaを重複なく1回ずつ含む完全順列とする。
- Roleの `expertise_domains` や `observation_scope` に含まれないcriterionも削除してはならない。根拠が弱い場合は低い重みまたは低いconfidenceで表現し、次元自体は維持する。
- モデル出力で項目欠落、未知項目、重複、負値、合計0が発生した場合、残った項目だけを再正規化して受理してはならない。契約違反として再試行し、上限後は測定失敗として記録する。
- 重みは許容誤差 `1e-6` 以内で合計1へ正規化する。実装側の丸め補正を行った場合は補正前後を監査ログへ残す。
- `value_criteria_schema` のID、version、本文、SHA-256をturn CSVまたはmanifestから追跡可能にする。

#### 5.4.2 RoleとValueの交絡検査

RoleがValueを決定していないことを検査するため、GPU本実験前に次の割付けを行う。

- 同一Roleへ2種類以上の初期Valueを割り当てる。
- 同一Valueを2種類以上のRoleへ割り当てる。
- 少なくとも1つのsmokeセルでalpha/betaのValue割当を交換し、Roleを固定する。
- `role_value_assignment_id` を事前に固定し、各seed・conditionへ同じ割当を適用する。
- Role専門領域と高重みcriterionの一致は記述的診断として報告するが、一致自体を成功または失敗とみなさない。

主比較でRoleとValueを常に意味的に整合した一組だけに固定し、フレームワーク効果とRole-Value結合を識別不能にしてはならない。

### 5.5 Personaと交渉特性

- `communication_style` と `evidence_demand` は、価値順位を直接命令しない範囲でPersonaに残せる。
- `concession_tendency`、`consensus_orientation`、`dominance` は交渉成否に直接影響するため、初期主比較では全agent同一の中立値に固定する。
- 交渉特性を操作する実験は、フレームワーク効果と分けた二次実験とする。

## 6. V整合プロセス要件

### 6.1 議論前の独立記録

相手の議論を見せる前に、各agentから次を独立に取得する。

```json
{
  "v_before": {
    "ordered_criteria": ["oxygen", "power", "hull_damage", "flooding", "communication"],
    "weights": {"oxygen": 0.30, "power": 0.25, "hull_damage": 0.20, "flooding": 0.15, "communication": 0.10},
    "confidence": 0.6
  },
  "action_before": "B",
  "reason_before": "電力切れが直近の破局リスクだから"
}
```

- 議論前取得は、議論トランスクリプトと合意形成手順の影響を受けない。
- `action_before` は個人の初期行動案であり、最終投票と別列にする。
- `v_before` を取得せず、最終投票から逆算してはならない。

### 6.2 V*の提案と受諾

HIVC-D条件は、自由議論中に次の論理状態を形成する。

1. 両agentが相手の `v_before` と根拠を確認する。
2. 少なくとも1agentが `v_proposal` を提示する。
3. 各agentが提案に `accept` / `reject` / `counter` のいずれかを返す。
4. 両agentが同一の基準と順序を `accept` した場合に限り `v_star_status=accepted` とする。
5. 受諾されない場合は `v_star_status=unresolved` とし、合意したものとして補完しない。

`V*` は「個人Vを永久に変える」ことを必須としない。個人Vが異なるままでも、そのターンのグループ行動を比較する共通規則を受け入れれば成立とする。

#### 6.2.1 V整合必要性の判定

各ターンの議論開始前に、システムはモデルの自己申告とは独立して `v_alignment_required` を判定する。

次のいずれかを満たす場合は `true` とする。

- `alpha_action_before != beta_action_before`
- 共通Vオントロジー上のL1距離が初期閾値 `0.20` 以上
- 両agentの `ordered_criteria` の先頭criterionが異なる

いずれも満たさない場合は `false` とし、`v_star_status=not_required_already_aligned` を使用できる。`not_required` を `unresolved` や `accepted` に含めてはならない。

#### 6.2.2 V提案機会の保証

- HIVC-Dで `v_alignment_required=true` の場合、最終投票へ進む前に最低1件の有効な `v_proposal` が必要である。
- 自由議論だけで提案が出なかった場合、残り予算内で `v_proposal_required` の構造化プロンプトを提示する。
- システムはV提案の内容を生成・補完してはならない。agentが有効な提案を返さない場合は `unresolved` とし、`missing_v_proposal_after_required_prompt` を記録する。
- proposal、相手の `accept|reject|counter`、counter後の応答に必要な発話予算を、通常の情報要求より先に予約する。
- controlとconsultingには `v_proposal_required` を提示しない。条件対称性のためのV測定と、HIVC-D固有のV交渉を混同しない。

#### 6.2.3 V整合状態機械

HIVC-Dの論理状態は次の順序を持つ。

```text
I_SHARE
  -> V_COMPARE
  -> V_PROPOSE | V_NOT_REQUIRED
  -> V_RESPOND
  -> A_CHECK
  -> FINAL_VOTE
```

- 未回答質問が残る間は `I_SHARE` を完了扱いにしない。
- `v_alignment_required=true` かつ有効proposalがない状態で `A_CHECK` または `FINAL_VOTE` へ遷移してはならない。
- 各遷移、遷移理由、失敗理由、消費発話数・token数をturn状態とCSVへ保存する。
- 絶対予算上限に達した場合のみ強制遷移を許し、`forced_decision_reason` とV失敗理由を別々に記録する。

### 6.3 V*のスコープと永続化

- `scope=turn` は当該ターンのみに適用する。
- `scope=game` は後続ターンにも適用し、新しい事実やイベントにより再交渉可能とする。
- 初期実装は `scope=turn` を標準とし、ターンをまたぐ効果は別実験とする。
- 同一ターン内の複数の意思決定機会では、受諾済み `V*` を構造化状態として引き継ぐ。
- 次の発言プロンプトで静的な初期Vだけを再注入し、受諾済み `V*` を落としてはならない。

### 6.4 投票とV*の接続

各最終投票は次を返す。

```json
{
  "action": "B",
  "reason": "V*で第一とした直近の破局回避に合致する",
  "ready": true,
  "v_star_id": "seed42-turn1-v2",
  "v_star_consistent": true
}
```

- `v_star_status=accepted` の場合、各agentは `v_star_id` を必ず参照する。
- `v_star_consistent` はモデルの自己申告だけで確定せず、後処理でも実際の基準と行動根拠の整合を検査する。
- 個人Vと `V*` が異なること自体は不整合としない。受諾後のグループ投票に `V*` を適用できたかを評価する。

### 6.5 測定操作の条件間対称性

Vを測るための追加処理そのものを、HIVC-Dだけの効果にしてはならない。

- `v_before` と `action_before` の取得プロンプト、出力スキーマ、生成設定、取得タイミングは全フレームワーク条件で同一にする。
- 議論前測定でHIVC-D、V整合、V*受諾を指示しない。その時点の判断基準と行動案の記述だけを要求する。
- 議論前測定結果を自分の議論プロンプトに戻すかどうかは全条件で統一し、manifestへ記録する。
- `v_after` の取得が最終投票に影響しないよう、原則として最終投票確定後に測定する。
- 追加の測定呼び出し数、トークン数、失敗・再試行数を条件ごとに記録する。
- controlとconsultingにV交渉を強制してはならない。自然に提案された判断基準は記録してよいが、未提案をV*として補完しない。

### 6.6 質問・回答閉包と反復防止

質問閉包は全framework条件で共通の基盤契約とし、HIVC-D固有効果に含めない。

#### 6.6.1 質問行為の正規化

- `information_request`、`question_objection`、および質問として定義した別名は、内部表現 `question` へ正規化する。
- questionはモデル出力の `requires_response` にかかわらず常に `requires_response=true` とする。
- questionは `message_id` と有効な `addressed_to` を必須とする。
- 宛先は、要求されたstate fieldを `observation_scope` に持つagentを優先する。両者とも観測できないfieldへの質問は `unanswerable_question` として閉じ、再送しない。

#### 6.6.2 回答優先と閉包

- 自分宛の未回答質問があるagentの次発話は、対象 `message_id` を指定した `reply_to_message_id` 付き回答に限定する。
- 回答中の新規質問、別話題への遷移、同じ質問の再送を禁止する。
- 回答として質問を返した場合は質問を閉じず、`invalid_response_while_answer_required` として再試行する。
- 回答不能な場合も無回答にせず、観測不能である事実と、回答可能なRoleまたは不足情報を構造化して返す。
- 未回答質問が残ったまま通常の意思決定へ進む場合は、絶対予算上限到達など明示的な失敗理由を必須にする。

#### 6.6.3 反復質問の抑止

- `(speaker, addressed_to, normalized_requested_fields)` をquestion signatureとして管理する。
- 同じsignatureのopen questionを同じspeakerが再送してはならない。
- 同文または同一signatureの再送を検出した場合、発話予算へ加算せず、宛先agentの回答専用プロンプトへ切り替える。
- closed questionの再質問は、新しいstate変化または新しい根拠を `reask_reason` で示した場合のみ許可する。
- `duplicate_question_count`、`duplicate_question_rate`、最大連続反復数を条件別に保存する。

#### 6.6.4 JSON契約違反

- JSON構文不正、必須キー欠落、型不一致、壊れたJSON断片を `message` に埋め込んだ出力を有効発話として扱わない。
- raw出力は監査用の別フィールドへ保存し、有効transcriptには検証済み構造だけを追加する。
- 再試行上限後は `invalid_discussion_output` として記録し、空文字や別speech actへ暗黙変換しない。

## 7. プロンプト合成と優先順位

プロンプトは次の論理優先順位に従う。

```text
ゲームの不変ルール・可視情報境界・JSON契約
  > 現在の構造化状態（未回答質問、受諾済みV*）
  > 共有フレームワーク
  > Roleの専門性・観測責務
  > 変更可能な個人の初期V
  > Personaの伝え方
  > 過去の議論
```

次を必須とする。

- 各ブロックは見出しと機械可読なIDで区切る。
- 静的プロファイルと現在状態が異なる場合は、現在状態を優先する。
- Roleは担当事実の提供を要求できるが、受諾済み `V*` と矛盾する固定行動を強制できない。
- `V*` が未成立の場合、一致したものとしてプロンプトへ補完しない。
- 数値重みを「必ず守る命令」と解釈させる説明を禁止する。

## 8. 実験条件

### 8.1 Role/Value表現の感度分析

次の三レジームを定義する。

| `role_value_mode` | Role | Value | 位置付け |
|---|---|---|---|
| `legacy_hard` | 現行 `role.json` の複合定義 | 固定重みと規範的notes | 現行runとの接続用。感度分析のみ |
| `soft_value` | 専門性・可視情報を分離 | 変更可能な初期重み | 新しい主比較 |
| `expertise_only` | 専門性・可視情報のみ | 明示的な事前重みなし | 数値Vアンカーの感度分析 |

### 8.2 フレームワーク条件

各 `role_value_mode` で次を比較できるようにする。

| `framework_id` | 手順 | 位置付け |
|---|---|---|
| `control` | 追加合意形成手順なし | 負の対照 |
| `consulting` | 一般的な情報整理・リスク比較・確認 | 非特異的助言対照 |
| `hivc_d` version 2 | I共有・事実に基づくV提案・両者の明示受諾・A確認 | 新しい主検証 |
| `hivc_d_prescribed_v1` | 「破局リスク → 勝利条件 → 次ターンの選択肢」を外部からV*として与える現行相当手順 | 感度分析。自由なV整合として報告しない |

`hivc_d` version 2 はV*の具体的な優先順位を事前に与えない。ゲームの勝敗条件と不変の安全制約は共有するが、何をどの順で比較するかは当該ターンの観測事実と両agentの提案から形成する。

最初の主比較は `soft_value` 内の三条件とする。`legacy_hard` は現行結果との比較、`expertise_only` はアンカー影響の感度分析とし、主仮説と混合しない。

### 8.3 条件固定と割付け

- 同一のモデル版、量子化設定、生成設定、Role、Persona、初期V、seed、scenario、議論予算、評価器を条件間で一致させる。
- 各seedは全フレームワーク条件に対応付ける。
- 条件の実行順はseedごとにランダム化または反平衡化し、`control → consulting → hivc_d` の固定ブロック順と条件効果を交絡させない。
- 固定Role組み合わせ一組だけを一般化しない。複数の対立強度と専門領域の組み合わせを事前に定め、各条件へ同一に割り当てる。
- 交渉特性は主比較で中立化し、操作する場合は独立因子として記録する。
- Role、Persona、Valueの割当は、固定プロファイルの場合もseed・conditionごとの明示的な割当レコードとして生成する。
- 同一seedではframework条件間で同じ `role_value_assignment_id` を使用する。

### 8.4 比較規模

- 実装確認は各条件・各modeで5seed以上のsmokeとする。
- 探索的比較は各セル30seed以上とする。
- 主検証は事前に固定した `soft_value` の対応ありseed各100以上とする。
- `legacy_hard` と `expertise_only` まで同時に完全要因実験を行う場合は計算量が増えるため、主検証と感度分析の実行順を事前登録する。

## 9. 記録スキーマ

### 9.1 ターンCSV

現行スキーマに次を追加する。JSON子構造はUTF-8 JSON文字列として保存する。

```text
role_value_mode,
alpha_role_id, beta_role_id,
alpha_value_profile_id, beta_value_profile_id,
alpha_value_profile_sha256, beta_value_profile_sha256,
alpha_v_before, beta_v_before,
alpha_action_before, beta_action_before,
v_proposals, v_star_id, v_star, v_star_scope, v_star_status,
alpha_v_star_response, beta_v_star_response,
alpha_v_after, beta_v_after,
alpha_vote_changed, beta_vote_changed,
alpha_v_star_consistent, beta_v_star_consistent,
v_alignment_distance_before, v_alignment_distance_after,
v_star_action_consistency
```

加えて、修正後スキーマは次を記録する。

```text
value_criteria_schema_id, value_criteria_schema_version,
v_alignment_required, v_alignment_requirement_reasons,
v_protocol_state, v_protocol_transition_history,
question_count, answered_question_count, unanswered_question_count,
duplicate_question_count, max_consecutive_duplicate_questions,
invalid_discussion_output_count
```

### 9.2 `value_manifest.json`

各runに次を保存する。

- 実行時に解決したRole、Persona、Valueプロファイルの完全本文
- 各プロファイルのID、version、SHA-256、入力パス
- `role_value_mode`
- 交渉特性の解決後値
- フレームワークID、version、SHA-256
- モデルパスまたはモデルID、生成設定、seed範囲、scenario範囲
- 実験設定の完全スナップショットとSHA-256
- Git commit、開始時刻、ランナー版
- `value_criteria_schema` のID、version、完全本文、SHA-256
- `game_profile_assignments` を固定・ランダムプロファイルの別なく、全 `(seed, condition)` について保存する
- 各assignmentに `role_value_assignment_id`、alpha/betaのRole・Persona・Value IDとSHA-256を保存する
- `seed_range`、`scenario_range` を空欄のままにせず、対象範囲または明示的な `not_applicable_reason` を保存する

parallel mergeは、期待assignment数を `games_per_condition × condition数` として検査する。欠落、重複、masterとshardの不一致がある場合、mergeを成功扱いにしない。

manifestに正本を保存せず、実験後の現在ファイルからRoleやValueを推定してはならない。

### 9.3 ローカルプレビュー

同一seed・turnについて、次を条件間で比較できるようにする。

- 初期Vと議論前行動
- V提案とaccept/reject/counterの時系列
- 受諾されたV*と適用scope
- 最終投票と議論前からの変化
- V*と最終行動の整合判定
- regret、best action、outcome
- Role・Value・FrameworkのID、version、ハッシュ

## 10. 評価指標

### 10.1 Vプロセス指標

| 指標 | 定義 | 望ましい方向 |
|---|---|---:|
| `v_proposal_rate` | `v_alignment_required=true` のターンで有効提案が出た割合 | 高 |
| `v_star_acceptance_rate` | V提案のうち両者が同一V*を受諾した割合 | 高 |
| `v_alignment_distance_before` | 議論前の両V間距離 | 条件間で同一 |
| `v_alignment_distance_after` | 議論後の両V間距離 | 低 |
| `v_alignment_gain` | `distance_before - distance_after` | 高 |
| `vote_revision_rate` | 議論前行動と最終投票が異なる割合 | 記述的 |
| `v_star_action_consistency` | 受諾V*と最終投票が整合する割合 | 高 |
| `unresolved_v_rate` | `v_alignment_required=true` のままV*未成立で最終決定へ進んだ割合 | 低 |
| `v_schema_completeness_rate` | 全Vレコードのうち共通criteriaを完全に持つ割合 | 1.0 |
| `missing_v_proposal_after_required_prompt_rate` | 提案必須prompt後も有効提案がない割合 | 低 |

重みベクトル間距離は、同一項目の正規化後L1距離を標準とする。順位のみを使うプロファイルではKendallの順位相関を併記する。数値形式と順位形式を同じ指標に混在させない。

`v_alignment_required=false` のターンをV提案率や未解決率の分母へ含めない。分母0は `NaN` とするが、HIVC-DのGPU smoke全体で分母0の場合は「V整合経路未検証」として受入失敗にする。

`v_alignment_distance_before` と `v_alignment_distance_after` はV*の提案・受諾有無から独立して、共通criteriaを満たす両agentのVが存在すれば計算する。V*がないことだけを理由に距離を `NaN` にしてはならない。V測定契約違反で計算不能な場合は、`NaN` と具体的な失敗理由を記録する。

### 10.2 会話契約指標

| 指標 | 定義 | 望ましい方向 |
|---|---|---:|
| `question_answer_rate` | answerable questionのうち有効回答で閉じた割合 | 1.0 |
| `duplicate_question_rate` | 全questionに対する禁止された再送の割合 | 0 |
| `max_consecutive_duplicate_questions` | 同一signatureが連続した最大回数 | 1以下 |
| `invalid_discussion_output_rate` | 全生成発話に対する契約違反出力の割合 | 低 |
| `silent_unanswered_question_count` | 失敗理由なしで意思決定へ持ち越した未回答数 | 0 |

質問指標は、モデルが返した `requires_response` の値ではなく、正規化後の質問状態機械から計算する。

### 10.3 意思決定品質と成果

現行の次の指標を維持する。

- `expert_match_rate`
- `mean_regret`
- `conflict_resolution_quality`
- `plan_revision_quality`
- `agreement_rate_by_opportunity`
- `fallback_rate`
- `route_choice_accuracy`
- `survival_rate`
- `win_rate`
- `mean_return`

合意率の上昇だけを成功としない。`V*` 形成、V*と行動の整合、regretまたは成果改善の連鎖として判定する。

### 10.4 指標の有効性

- 分子と分母を必ず併記する。
- 該当機会がない指標は0ではなく `NaN` とする。
- 2agentの同数対立で「少数派」が定義できない場合、`minority_adoption_rate` を主要根拠に使わない。
- 全条件で上限または下限に飽和した指標は、成果の根拠ではなく実装チェックとして扱う。
- `premature_launch_rate` のように対象行動が少ない指標は、率だけでなく実数を示す。

## 11. 仮説と統計要件

### 11.1 主仮説

`soft_value` モードで次を事前登録する。

- **H-V1**: HIVC-Dはcontrolおよびconsultingより `v_star_acceptance_rate` を上げる。
- **H-V2**: HIVC-Dはcontrolおよびconsultingより `v_star_action_consistency` を上げる。
- **H-V3**: HIVC-Dはcontrolおよびconsultingより `mean_regret` を下げる。

`win_rate`、`survival_rate`、`mean_return`、個別の行動選択、失敗種別は副次指標とする。まずV整合と行動接続を確認し、終端成果の分散だけに依存しない。

### 11.2 統計解析

- 同一seed間の対応を保った対応あり比較を行う。
- ゲーム単位でブートストラップ95%信頼区間を報告する。
- 二値の勝敗は対応ありの不一致数を示し、必要に応じて正確McNemar検定を使う。
- 複数の主指標を同時検定する場合は、有意水準または多重性補正を事前に定める。
- 実験結果を見た後の仮説・サブグループは「探索的」と明記し、確認的結果と混在させない。
- Role/Value分離の効果は `framework × role_value_mode` 交互作用として感度分析し、HIVC-Dの主効果と混合しない。

## 12. 後方互換と移行

### 12.1 現行 `role.json`

- 現行エントリは削除せず、`schema_version=legacy-1`、`role_value_mode=legacy_hard` として読み込めるようにする。
- 旧ファイルを暗黙に `soft_value` へ変換しない。同じデータの意味が変わるためである。
- `legacy_hard` を使った場合は実験開始時とmanifestに警告を記録する。
- `legacy_hard` の結果を、新しい `soft_value` 主比較にプールしない。

### 12.2 新プロファイル

新しいプロファイルは、例えば次のように分離する。

```text
roles/<role-id>.yaml
personas/<persona-id>.yaml
values/<value-id>.yaml
```

実際のフイル分割数は実装時に決めてよいが、manifestと解決後内部表現では責務を分離する。

## 13. テスト要件

### 13.1 スキーマ検証

- Roleへの禁止フィールド混入を検出できる。
- Valueの重み欠落、非数値、負値、合計0、未知項目を明確なエラーにする。
- `negotiable=false` を主比較に含めようとした場合は停止または明示的な感度分析ラベルを必須にする。
- 共通criteriaから1項目でも欠けたV、未知項目を含むV、不完全な `ordered_criteria` を拒否する。
- Roleの観測範囲が狭くても、V出力のcriteria集合が縮退しないことを検証する。

### 13.2 プロンプトと状態

- Role、Persona、初期V、現在V、受諾済みV*が別ブロックで出力される。
- 同一ターンの次の意思決定機会で、受諾済みV*が欠落しない。
- 静的初期Vと現在Vが異なるテストで、現在Vが優先される。
- `v_star_status=unresolved` の状態で、受諾済みとして投票プロンプトに挿入されない。
- `v_alignment_required=true` でproposalがない場合、最終投票より前に `v_proposal_required` が提示される。
- controlとconsultingでは `v_proposal_required` が提示されない。
- V状態機械の全正常遷移と、予算上限による失敗遷移を検証する。

### 13.3 会話契約

- `v_proposal`、`accept`、`reject`、`counter` を正常にパースできる。
- 対象V提案IDのない受諾、両者で内容が異なる受諾、モデル出力の欠落を「合意」として補完しない。
- 上限に達した場合は `unresolved` と失敗理由を記録し、fallbackの行動選択とV*成立を分ける。
- `information_request` と `question_objection` の双方がquestionへ正規化され、`requires_response=true` になる。
- 自分宛の未回答質問があるagentが新しい質問を返した場合、元質問を閉じず再試行する。
- open questionと同一signatureの再送を発話予算へ加算せず、回答者へ制御を移す。
- 観測不能な質問を無限再送せず、`unanswerable_question` として閉じる。
- 壊れたJSONやJSON断片を有効なmessageへ降格せず、raw出力と契約違反を記録する。

### 13.4 CSV・manifest・プレビュー

- ターンCSVの新列が連続複数ゲームとparallel mergeで保存される。
- `value_manifest.json` のハッシュと実行時本文が一致する。
- ローカルプレビューは、同一seedの条件比較で異なる `V*` を混在させない。
- 旧CSVは新列がなくても開け、V指標を「記録なし」と表示する。
- 固定プロファイルでも全 `(seed, condition)` の `game_profile_assignments` が生成される。
- parallel mergeでassignment欠落・重複・ハッシュ不一致を検出し、成功扱いにしない。
- unanswered、duplicate、invalid outputの集計値がtranscriptから再計算した値と一致する。

## 14. 受け入れ基準

1. Role、Persona、Value、Frameworkの各責務が解決後データとプロンプトで分離されている。
2. `soft_value` の初期Vが変更可能な事前基準として明記され、固定行動命令として再注入されない。
3. 議論前のVと行動、V提案、両者の受諾、受諾済みV*、最終投票を別々に記録できる。
4. 受諾されていないV*をシステムが合意済みとして補完しない。
5. 受諾済みV*が同一ターン内の後続発言・投票へ引き継がれ、Roleの固定優先指示により消失しない。
6. 個人Vが不一致のままでも、グループ判断用V*の明示的な受諾を表現できる。
7. `legacy_hard`、`soft_value`、`expertise_only` を同一ランナーで別ラベルとして実行し、混在させず集計できる。
8. 同一seed、Role、Persona、初期V、生成設定を各framework条件で共有できる。
9. 主比較は `soft_value` で行い、主仮説・指標・サンプル数・多重性の扱いをGPU実験前に固定できる。
10. 実行時のRole、Value、Framework、設定、Git commitをmanifestから復元できる。
11. 旧runは後方互換で閲覧でき、V測定のない旧結果を新主比較に混入しない。
12. 全テストが通過し、GPU smokeでV状態からCSV・manifest・プレビューまで追跡できる。
13. 全Vレコードが同一の `value_criteria_schema` を完全に保持し、Roleに応じた項目脱落がない。
14. HIVC-Dの `v_alignment_required=true` ターンで、V提案機会を経ずに最終投票へ進まない。
15. question型の別名が共通状態へ正規化され、回答、回答不能、または明示的予算失敗のいずれかで閉じる。
16. 同一open questionの再送ループがなく、壊れたJSONを有効発話として保存しない。
17. 固定プロファイルを含む全seed・conditionのRole/Persona/Value割当をmanifestから復元できる。

### 14.1 大規模GPU実験の開始ゲート

100seed主検証を開始する前に、次を順番に満たす。

#### Gate A: ローカル契約テスト

- 共通Vオントロジー、V状態機械、質問閉包、反復抑止、JSON拒否、manifest完全性の自動テストがすべて成功する。
- 人工的に `action_before` とV順位を対立させた統合fixtureで、proposal → response → accepted/unresolved → voteの全経路を再現できる。

#### Gate B: GPU smoke

- `soft_value`、3framework、対応あり5seed以上を実行する。
- merge検査、config hash、seed集合、assignment完全性がすべて合格する。
- `v_schema_completeness_rate=1.0`。
- HIVC-Dの `v_alignment_required=true` ターンが1件以上存在し、`v_proposal_rate` の分母が0でない。
- HIVC-Dの対象ターンに対する `v_proposal_rate >= 0.8`。
- smoke全体でaccepted V*が1件以上あり、V*から投票まで追跡できる。これは効果量の合格基準ではなく経路実装の到達確認とする。
- `silent_unanswered_question_count=0`。
- `duplicate_question_rate <= 0.05` かつ `max_consecutive_duplicate_questions <= 1`。
- 壊れたJSONが有効transcriptへ入った件数が0。
- thermal slowdown、OOM、shard欠落が0。

#### Gate C: 探索的30seed

- Gate B後に対応あり30seedを実行し、指標分母、失敗理由分布、実行時間を確認する。
- V提案・質問契約の失敗が特定Role、condition、turnへ偏っていないことを確認する。
- Gate Cの結果を見て主仮説や成功基準を有利に変更しない。変更が必要なら別versionとして再事前登録する。

Gate A〜Cのいずれかが不合格なら100seedへ進まない。`episode-20260717-202452` はGate B不合格の診断runとして扱う。

### 14.2 pilot findingと要求の追跡

| Finding | 修正要求 | 主な検証 |
|---|---|---|
| P-V01 | §6.2.1〜6.2.3 V必要性判定・提案機会・状態機械 | Gate Aの人工対立fixture、Gate Bのproposal分母と到達率 |
| P-V02 | §5.4.1 共通Vオントロジー | criteria欠落拒否テスト、`v_schema_completeness_rate=1.0` |
| P-Q01 | §6.6.2〜6.6.3 回答優先・反復抑止 | duplicate率、最大連続反復数、質問→回答統合テスト |
| P-Q02 | §6.6.1〜6.6.2 質問正規化・閉包 | transcript再集計との一致、silent unanswered 0 |
| P-J01 | §6.6.4 JSON契約違反 | malformed JSON拒否・raw監査テスト |
| P-M01 | §9.2 assignment完全性 | 固定profile mergeテスト、期待件数検査 |

## 15. 修正実装順序（未実施）

1. `value_criteria_schema` と完全次元バリデータを追加し、部分Vの暗黙受理を停止する。
2. question型の正規化、回答優先、反復signature、unanswerable処理を共通会話基盤へ追加する。
3. JSON契約違反を有効messageへ降格しない再試行・監査経路を追加する。
4. `v_alignment_required` とHIVC-DのV状態機械を追加し、必要時のproposal予算を予約する。
5. ターンCSVと指標へV状態、質問閉包、反復、invalid outputの監査列を追加する。
6. 固定プロファイルを含む `game_profile_assignments` をseed・condition単位で生成し、parallel mergeの完全性検査へ追加する。
7. Role×Valueを直交または交換したsmoke用割付けを作成する。
8. Gate Aのユニット・統合テストを実施する。
9. Gate Bの対応あり5seed GPU smokeを実施し、ログとローカルプレビューを目視確認する。
10. Gate Cの30seedを完了後、事前登録を固定して100seed主検証へ進む。

## 16. 対象外

- 現行のゲーム規則、勝敗条件、Q値評価器のスコア重みの変更
- 非公開情報をエージェントに追加開示すること
- 結果を有意にするためのRole、V、シナリオ、評価指標の事後調整
- 個人Vの完全一致を合意の必須条件にすること
- HIVC-D本体の理論定義の書き換え
- GPU実験の即時実行
