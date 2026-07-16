# RoleとValue分離によるV整合検証要件

## 1. 目的

エージェントの専門領域・可視情報・伝え方を定める `Role` と、行動案を評価する判断基準 `Value`（以下 `V`）を分離する。

現行の `role.json` は、専門性だけでなく、`priority_weights`、`goal_focus`、`risk_tolerance`、`concession_tendency`、優先行動を指定する `notes` を同時に与える。これらが変更不可能な人格命令として働くと、HIVC-Dが想定する `V` の明示・整合・共通基準 `V*` の形成が成立しない。

本要件は、次の二点を切り分けて検証できる実験基盤を定義する。

1. 合意形成フレームワークが、個人の初期 `V` から共通基準 `V*` を形成できるか。
2. `V*` の形成が、投票、行動、regret、生存、勝利へつながるか。

本文書は要件定義であり、現時点で実装と再実験は行わない。

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
    "ordered_criteria": ["avoid_immediate_loss", "preserve_power", "advance_rescue"],
    "weights": {"avoid_immediate_loss": 0.5, "preserve_power": 0.3, "advance_rescue": 0.2},
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
| `v_proposal_rate` | V不一致ターンで提案が出た割合 | 高 |
| `v_star_acceptance_rate` | V提案のうち両者が同一V*を受諾した割合 | 高 |
| `v_alignment_distance_before` | 議論前の両V間距離 | 条件間で同一 |
| `v_alignment_distance_after` | 議論後の両V間距離 | 低 |
| `v_alignment_gain` | `distance_before - distance_after` | 高 |
| `vote_revision_rate` | 議論前行動と最終投票が異なる割合 | 記述的 |
| `v_star_action_consistency` | 受諾V*と最終投票が整合する割合 | 高 |
| `unresolved_v_rate` | V不一致のまま最終決定へ進んだ割合 | 低 |

重みベクトル間距離は、同一項目の正規化後L1距離を標準とする。順位のみを使うプロファイルではKendallの順位相関を併記する。数値形式と順位形式を同じ指標に混在させない。

### 10.2 意思決定品質と成果

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

### 10.3 指標の有効性

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

### 13.2 プロンプトと状態

- Role、Persona、初期V、現在V、受諾済みV*が別ブロックで出力される。
- 同一ターンの次の意思決定機会で、受諾済みV*が欠落しない。
- 静的初期Vと現在Vが異なるテストで、現在Vが優先される。
- `v_star_status=unresolved` の状態で、受諾済みとして投票プロンプトに挿入されない。

### 13.3 会話契約

- `v_proposal`、`accept`、`reject`、`counter` を正常にパースできる。
- 対象V提案IDのない受諾、両者で内容が異なる受諾、モデル出力の欠落を「合意」として補完しない。
- 上限に達した場合は `unresolved` と失敗理由を記録し、fallbackの行動選択とV*成立を分ける。

### 13.4 CSV・manifest・プレビュー

- ターンCSVの新列が連続複数ゲームとparallel mergeで保存される。
- `value_manifest.json` のハッシュと実行時本文が一致する。
- ローカルプレビューは、同一seedの条件比較で異なる `V*` を混在させない。
- 旧CSVは新列がなくても開け、V指標を「記録なし」と表示する。

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

## 15. 実装順序（未実施）

1. Role、Persona、Valueの解決後スキーマとバリデータを追加する。
2. 現行 `role.json` を `legacy_hard` として読む互換層と、新しい `soft_value` / `expertise_only` プロファイルを追加する。
3. 議論前V・行動取得、V提案、受諾、状態永続化の会話契約を追加する。
4. 自由議論と意思決定プロンプトを、新しい優先順位とV状態に対応させる。
5. ターンCSV、集計、`value_manifest.json`、parallel mergeを新スキーマに対応させる。
6. ローカルプレビューにVの時系列と条件比較を追加する。
7. ユニット・統合テストを追加し、ローカルでスキーマと集計を検証する。
8. 各mode・各frameworkでGPU smokeを行い、プロンプト、V状態、CSV、manifest、プレビューを目視確認する。
9. 主検証条件を事前登録した後、`soft_value` の対応あり100seed以上をGPUで実行する。

## 16. 対象外

- 現行のゲーム規則、勝敗条件、Q値評価器のスコア重みの変更
- 非公開情報をエージェントに追加開示すること
- 結果を有意にするためのRole、V、シナリオ、評価指標の事後調整
- 個人Vの完全一致を合意の必須条件にすること
- HIVC-D本体の理論定義の書き換え
- GPU実験の即時実行
