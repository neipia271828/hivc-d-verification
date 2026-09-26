"""Build the editable, unified oral presentation from the existing slide CSS."""

from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE_CSS = (HERE / "unified_presentation_base.css").read_text(encoding="utf-8")
FIT_SCRIPT = (HERE / "unified_presentation_fit.js").read_text(encoding="utf-8")
TITLE = "2026年度情報処理学会 関西支部 支部大会 口頭発表資料"
OUTPUT = HERE / "ipsj_kansai_2026_unified_presentation.html"


SLIDES = [
    ("表紙", """
      <div class="eyebrow">2026年度情報処理学会 関西支部 支部大会</div>
      <h1 class="cover-title">複数人の意思決定を支える<br><span class="blue">HIVC-D</span>の提案と検証</h1>
      <p class="subtitle">口頭発表資料｜GLM-4.7による二者協力ゲーム・90ゲームの探索的評価</p>
      <div class="cover-rule"></div>
      <p class="speaker">神山まるごと高専 本科2年<br><strong>鈴木陽向</strong></p>
    """),
    ("本実験の主旨", """
      <h2>本実験の主旨</h2>
      <div class="panel"><p class="big">複数人での意思決定において、対立の理由を整理し、合意形成を支える枠組み <span class="blue">HIVC-D</span> を提案する。</p></div>
      <div class="cols"><div class="panel"><h3>対象</h3><p>同じ目的を持ちながら、見えている情報や優先基準が異なる二者の議論。</p></div><div class="panel"><h3>問い</h3><p>情報・価値・実行可能性を順に確認すると、成果と合意品質は改善するか。</p></div></div>
      <p class="muted">本発表は現行実装を90ゲームで評価した結果を報告する。</p>
    """),
    ("本実験の意義", """
      <h2>本実験の意義</h2>
      <div class="panel"><p class="big">人同士の会議をすぐに再現できなくても、LLMの役割演技を使えば、<span class="blue">異なる情報と判断基準を持つ二者</span>の合意手続きを反復評価できる。</p></div>
      <div class="cols"><div class="panel"><h3>実験で扱えること</h3><p>同じゲームとseedを使い、追加する議論手続きだけを変えて比較する。</p></div><div class="panel red"><h3>解釈の範囲</h3><p>LLMが実際の人間の感情・心理的安全性を代替できるかは、この実験だけでは判断できない。</p></div></div>
    """),
    ("流れ", """
      <h2>本日の流れ</h2>
      <div class="agenda">
        <div><b>01</b><span>背景</span></div><div><b>02</b><span>仮説</span></div><div><b>03</b><span>実験設計</span></div>
        <div><b>04</b><span>実験結果</span></div><div><b>05</b><span>考察</span></div><div><b>06</b><span>まとめ</span></div>
      </div>
    """),
    ("背景", """<div class="section-no">01 / 背景</div><h1>日常のMTGから生まれた問い</h1><p class="section-desc">自主制作の場で、意見の対立をどう建設的な議論につなげるか。</p>"""),
    ("背景", """
      <h2>神山まるごと高専のMTG</h2>
      <div class="cols"><div class="panel"><h3>自主制作・活動</h3><p>学生が主体的に制作や活動を進める。複数人で方針を決める機会が多い。</p></div><div class="panel"><h3>日常的なMTG</h3><p>案を持ち寄り、理由を伝え、作業の進め方を決める。</p></div></div>
      <div class="panel red"><p class="big">同じ制作目標に向かっていても、<span class="red">意見の対立</span>は起こる。</p></div>
    """),
    ("背景", """
      <h2>問題意識：安心して議論できるか</h2>
      <div class="flow"><div>意見が分かれる</div><span>→</span><div>理由が伝わらない</div><span>→</span><div class="red-text">感情的な議論</div></div>
      <div class="panel"><p class="big">「相手が間違っている」で止めず、<span class="blue">何が違うのか</span>を整理して話せないか。</p></div>
      <p class="muted">心理的安全性を直接測定した実験ではなく、そのための議論手続きを設計する動機である。</p>
    """),
    ("仮説", """<div class="section-no">02 / 仮説</div><h1>対立の原因を見分ける</h1><p class="section-desc">目標・情報・能力の違いを分け、必要な介入を考える。</p>"""),
    ("仮説", """
      <h2>合意のための三つの条件</h2>
      <div class="cols three"><div class="panel"><h3>同じゴール</h3><p>何を達成したいかを共有している。</p></div><div class="panel"><h3>同等の情報</h3><p>判断に必要な事実を共有している。</p></div><div class="panel"><h3>同等の能力</h3><p>選択した行動を実行できる。</p></div></div>
      <div class="panel"><p>原発表の出発仮説。これらは厳密な数学的「十分条件」として検証済みではなく、対立を整理するための作業仮説である。</p></div>
    """),
    ("仮説", """
      <h2>山登りの例：対立はどこで始まるか</h2>
      <div class="mountain"><div class="summit">▲ 山頂へ行く</div><div class="route-line"><span class="route-a">A：斜度の小さい赤ルート</span><span class="route-b">B：青ルートがよい</span></div></div>
      <div class="cols"><div class="panel"><h3>見えている根拠</h3><p>Aは地形を見て「赤の方が安全」と言う。</p></div><div class="panel red"><h3>まだ共有されていないこと</h3><p>Bの目的や情報、体力上の事情は分からない。</p></div></div>
      <p class="muted">元資料p21–31の山登りルートの絵コンテを再構成。</p>
    """),
    ("仮説", """
      <h2>議論の破綻と原因分類</h2>
      <div class="panel red"><p class="big">「赤の方が斜度が小さい」対「青ルートの気分」だけでは、<span class="red">比較する基準が共有されない</span>。</p></div>
      <table><tr><th>確認する違い</th><th>山登りでの問い</th><th>介入</th></tr><tr><td>ゴール</td><td>速く登る？ 安全に登る？</td><td>目的を共有する</td></tr><tr><td>情報</td><td>地形・天候を同じように見ている？</td><td>根拠を共有する</td></tr><tr><td>能力</td><td>その道を全員が歩ける？</td><td>役割やルートを調整する</td></tr></table>
    """),
    ("仮説", """
      <h2>HIVC-D：対立原因を順に扱う</h2>
      <div class="flow"><div><b>I</b><small>情報を共有</small></div><span>→</span><div><b>V</b><small>優先基準を調整</small></div><span>→</span><div><b>A</b><small>実行可能性を確認</small></div><span>→</span><div><b>D</b><small>決定</small></div></div>
      <div class="panel"><p>目的は共通という実験条件の下で、各役割の観測情報（I）、価値判断（V）、実行可能性（A）を対話で確認する。初期案が衝突した場合、V*は両者の明示的受諾で成立する。</p></div>
      <p class="muted">山登りの「ゴールの違い」は枠組みの発想に含まれる。今回のゲームでは共通ゴールを前提にした。</p>
    """),
    ("実験設計", """<div class="section-no">03 / 実験設計</div><h1>手続きだけを変えて比較する</h1><p class="section-desc">同じゲーム、同じモデル、対応するseedで三条件を評価。</p>"""),
    ("実験設計", """
      <h2>比較する三条件</h2>
      <div class="panel"><h3>control｜追加の協議手続きなし</h3><p>共通の役割・状態・ゲーム規則・出力形式のみを提示する。</p></div>
      <div class="panel"><h3>consulting｜一般的な協議</h3><p>情報確認、リスクと勝利寄与の比較、根拠に応じた譲歩、実行前確認を促す。</p></div>
      <div class="panel red"><h3>hivc_d｜I → V → A</h3><p>情報共有、明示的なV*提案・受諾、実行可能性確認を行う。</p></div>
      <p class="muted">元資料のA/B/Cは順に hivc_d / consulting / control に対応する。</p>
    """),
    ("実験設計", """
      <h2>使用したAI</h2>
      <div class="panel"><p class="big">Z.ai APIの <span class="blue">GLM-4.7</span> を二つの役割に使用。</p></div>
      <div class="cols"><div class="panel"><h3>生成設定</h3><ul><li>temperature 0.2</li><li>sampling無効</li><li>1応答256 token上限</li></ul></div><div class="panel"><h3>比較条件</h3><ul><li>両役割とも同じモデル</li><li>規則・観測・出力形式・対話予算を共通化</li><li>追加の手続き文面を変更</li></ul></div></div>
      <p class="muted">モデル名は元PDFの表記から実際の実験記録に合わせて修正。</p>
    """),
    ("実験設計", """
      <h2>タスク選定：独自ゲームを開発</h2>
      <div class="flow"><div>NASAゲームを検討</div><span>→</span><div class="red-text">学習データ混入の懸念</div><span>→</span><div>深海脱出ゲーム</div></div>
      <div class="panel"><p>NASAゲームは既知の正解が広く流通しており、モデルが記憶した答えで解く可能性がある。独自の資源管理・部分観測ゲームで、対話による意思決定を評価した。</p></div>
    """),
    ("実験設計", """
      <h2>独自ゲーム：深海脱出</h2>
      <div class="hero-grid"><div><b>5</b><span>最大ターン</span></div><div><b>6</b><span>行動</span></div><div><b>10</b><span>イベント</span></div><div><b>2</b><span>勝ち筋</span></div></div>
      <div class="cols"><div class="panel"><h3>共通目標</h3><p>2人の乗員が協力し、5ターン以内に生還する。</p></div><div class="panel"><h3>勝ち筋</h3><p>通信を復旧して救助を待つ／脱出艇を整備して自力で脱出する。</p></div></div>
      <p class="muted">安全重視と勝利へ向けた迅速な行動が対立しうる設計。</p>
    """),
    ("実験設計", """
      <h2>資源とターン進行</h2>
      <div class="cols"><div class="panel"><h3>主な状態</h3><table><tr><th>資源</th><th>役割</th></tr><tr><td>酸素・電力</td><td>0以下で敗北</td></tr><tr><td>船体損傷・浸水</td><td>5以上で敗北</td></tr><tr><td>通信・脱出艇</td><td>勝利条件の進捗</td></tr><tr><td>士気</td><td>終端スコアに反映</td></tr></table></div><div class="panel"><h3>毎ターンの流れ</h3><ol><li>酸素と電力が各1減る</li><li>確率イベントが起こる</li><li>二者が対話して行動を一つ選ぶ</li><li>終端条件を判定する</li></ol><p class="muted">「異常なし」22%。全行動が確実に敗北となるイベントは再抽選。</p></div></div>
    """),
    ("実験設計", """
      <h2>選べる六つの行動</h2>
      <table><tr><th>記号</th><th>行動</th><th>主な効果</th></tr><tr><td>A</td><td>酸素供給を安定化</td><td>酸素 +3、電力 −1</td></tr><tr><td>B</td><td>発電系を修理</td><td>電力 +3、25%で船体損傷 +1</td></tr><tr><td>C</td><td>通信アンテナを修理</td><td>通信 +1、電力 −1、25%で浸水 +1</td></tr><tr><td>D</td><td>浸水区画を封鎖</td><td>浸水 −2、酸素 −1</td></tr><tr><td>E</td><td>脱出艇を整備</td><td>整備度または健全性 +1、電力 −1</td></tr><tr><td>F</td><td>自力脱出を実行</td><td class="red">条件不足なら敗北</td></tr></table>
      <p class="muted">通信窓・浸水イベントではC/Dの効果が変わる。各行動の詳細は元のゲームルール7枚に記載。</p>
    """),
    ("実験設計", """
      <h2>勝利・敗北条件</h2>
      <div class="cols"><div class="panel"><h3>勝利：二つのルート</h3><ul><li><b>通信救助</b>：通信 ≥3 の後、2ターン生存</li><li><b>自力脱出</b>：脱出艇の整備度・健全性が各 ≥2、酸素 ≥3、電力 ≥2、浸水 ≤3でFを実行</li></ul></div><div class="panel red"><h3 class="red">敗北</h3><ul><li>酸素または電力が0以下</li><li>船体損傷または浸水が5以上</li><li>条件不足のままFを実行</li></ul></div></div>
      <div class="panel"><p>5ターン終了まで生存しても勝利条件を満たさなければ「生存タイムアウト」。勝率と生存率は別に数える。</p></div>
    """),
    ("実験設計", """
      <h2>部分観測と四つのシナリオ</h2>
      <div class="cols"><div class="panel"><h3>役割ごとの情報</h3><p><b>alpha：</b>船体損傷・浸水など安全側の診断。</p><p><b>beta：</b>通信・脱出艇・酸素・電力など脱出側の診断。</p><p class="blue">相手の診断を直接見られないため、情報共有が必要。</p></div><div class="panel"><h3>シナリオ</h3><table><tr><td>ambiguous</td><td>進路が不明確</td></tr><tr><td>comms_favored</td><td>通信が有利</td></tr><tr><td>escape_favored</td><td>脱出が有利</td></tr><tr><td>route_reversal</td><td>途中で有利な進路が逆転</td></tr></table></div></div>
    """),
    ("実験設計", """
      <h2>実験デザインと評価指標</h2>
      <div class="panel"><p class="big">3条件 × 30ゲーム ＝ <span class="blue">90ゲーム</span>。seed 42–71を各条件で共有し、実行順はLatin squareで入れ替える。</p></div>
      <div class="cols"><div class="panel"><h3>成果</h3><p>勝率、生存率、終端スコア（勝利 +1000、敗北 −200等の重み付き和）。</p></div><div class="panel"><h3>意思決定の過程</h3><p>行動合意率、fallback率、平均regret、無効出力・再試行、V受諾。</p></div></div>
      <p class="muted">regretは24回のrolloutで推定した最善行動とのQ値差。真の長期最適値ではない。</p>
    """),
    ("実験結果", """<div class="section-no">04 / 実験結果</div><h1>90ゲームで何が起きたか</h1><p class="section-desc">成果、合意、手続き品質を分けて読む。</p>"""),
    ("実験結果", """
      <h2>主要結果：条件別サマリー（各30ゲーム）</h2>
      <table class="compact"><tr><th>条件</th><th class="num">勝率</th><th class="num">生存率</th><th class="num">終端スコア</th><th class="num">平均regret</th><th class="num">合意率</th><th class="num">fallback率</th></tr><tr><td>control</td><td class="num">46.7%</td><td class="num">93.3%</td><td class="num">744.8</td><td class="num">100.8</td><td class="num">94.0%</td><td class="num">6.0%</td></tr><tr><td class="blue">consulting</td><td class="num blue">60.0%</td><td class="num">100%</td><td class="num blue">935.3</td><td class="num">119.0</td><td class="num">95.3%</td><td class="num">4.7%</td></tr><tr><td class="red">hivc_d</td><td class="num">46.7%</td><td class="num">100%</td><td class="num">740.2</td><td class="num red">127.3</td><td class="num red">78.4%</td><td class="num red">21.6%</td></tr></table>
      <div class="cols"><div class="panel"><p><span class="blue">consulting</span>が勝率・終端スコアで最高。</p></div><div class="panel red"><p>hivc_dは行動合意率が低く、fallbackが多い。</p></div></div>
      <p class="muted">勝利数は14 / 18 / 14。生存率100%の二条件は敗北0、controlは敗北2。</p>
    """),
    ("実験結果", """
      <h2>対応するseed間の比較</h2>
      <table class="compact"><tr><th>比較</th><th>指標</th><th class="num">平均差</th><th class="num">95% CI</th><th class="num">p値</th></tr><tr><td>hivc_d − control</td><td>終端スコア</td><td class="num">−4.7</td><td class="num">[−172.3, 158.5]</td><td class="num">0.93</td></tr><tr><td>hivc_d − control</td><td>合意率</td><td class="num red">−15.3pt</td><td class="num">[−21.5, −9.3]</td><td class="num red">0.0003</td></tr><tr><td>hivc_d − consulting</td><td>終端スコア</td><td class="num red">−195.2</td><td class="num">[−390.2, −4.5]</td><td class="num red">0.036</td></tr><tr><td>hivc_d − consulting</td><td>fallback率</td><td class="num red">+17.0pt</td><td class="num">[11.7, 22.2]</td><td class="num red">&lt;0.001</td></tr><tr><td>consulting − control</td><td>終端スコア</td><td class="num blue">+190.5</td><td class="num">[56.2, 345.2]</td><td class="num blue">0.026</td></tr></table>
      <div class="panel"><p>勝率の二値比較はいずれも有意ではない。検出力設計と多重比較補正のない<span class="red">探索的分析</span>として読む。</p></div>
    """),
    ("実験結果", """
      <h2>シナリオ別の勝率</h2>
      <table><tr><th>シナリオ</th><th class="num">control</th><th class="num">consulting</th><th class="num">hivc_d</th></tr><tr><td>ambiguous（各8）</td><td class="num red">0%</td><td class="num blue">37.5%</td><td class="num">12.5%</td></tr><tr><td>comms_favored（各7）</td><td class="num">57.1%</td><td class="num">57.1%</td><td class="num">57.1%</td></tr><tr><td>escape_favored（各7）</td><td class="num">57.1%</td><td class="num blue">71.4%</td><td class="num red">42.9%</td></tr><tr><td>route_reversal（各8）</td><td class="num">75.0%</td><td class="num">75.0%</td><td class="num">75.0%</td></tr></table>
      <div class="cols"><div class="panel"><p>差が見えるのは<span class="blue">不明確な進路</span>と<span class="blue">脱出有利</span>の局面。</p></div><div class="panel red"><p>各セル7–8ゲームの事後分析。別seedと別環境での追試が必要。</p></div></div>
    """),
    ("実験結果", """
      <h2>V受諾と手続き品質</h2>
      <div class="cols"><div class="panel"><h3>hivc_dのV整合</h3><p class="big">必要 <span class="blue">87ターン</span> → 受諾 <span class="blue">52ターン</span>（59.8%）</p><p>counter 52件、V交渉メッセージ234件。</p><p>受諾あり27ゲームの終端スコア775.2、なし3ゲームは425.0。後者はn=3で因果は判断できない。</p></div><div class="panel"><h3>機械処理の品質</h3><table><tr><th>条件</th><th class="num">無効出力</th><th class="num">再試行</th></tr><tr><td>control</td><td class="num">11</td><td class="num">24</td></tr><tr><td>consulting</td><td class="num">14</td><td class="num">25</td></tr><tr><td>hivc_d</td><td class="num blue">4</td><td class="num blue">18</td></tr></table></div></div>
      <div class="panel"><p>hivc_dのV測定tokenはcontrolより<span class="red">+138.4</span>（p=0.02）。明示的手続きは出力を安定させたが、成果優位には結びつかなかった。</p></div>
    """),
    ("考察", """<div class="section-no">05 / 考察</div><h1>合意率はなぜ下がったか</h1><p class="section-desc">成立条件、価値から行動への接続、議論予算を検討する。</p>"""),
    ("考察", """
      <h2>合意率低下の二つの経路</h2>
      <div class="cols"><div class="panel"><h3>① 成立条件が厳格</h3><p>対立時はV*への明示的acceptが必要。87ターン中52ターンで受諾。形式を通過しない決定はfallbackに分類される。</p></div><div class="panel red"><h3>② V*から行動が決まらない</h3><p>「安全重視」に合意しても酸素・電力・浸水への対処が競合する。投票が振動し、行動が決まらない。</p></div></div>
      <div class="panel"><p><b>代表例：</b>seed 63・turn 1で酸素優先のV*に合意後、AとBで投票が振動してfallback。採用Aに対し推定最善はC、regret 966.25。</p></div>
      <p class="muted">形式的受諾と実質的な譲歩が一致しない例もある。V交渉は限られた対話予算を使う。</p>
    """),
    ("考察", """
      <h2>安全性と決断性の両立</h2>
      <div class="cols"><div class="panel"><h3>安全側の結果</h3><p>hivc_dは敗北0・生存率100%。合意しない場合に安全候補へ置き換えるfallbackが、致命的失敗を避けた可能性がある。</p></div><div class="panel red"><h3>勝利への進捗</h3><p>勝率はcontrolと同じ46.7%。安全優先が通信・脱出の機会を逃す例がある。合意率だけでは判断の正しさを測れない。</p></div></div>
      <div class="panel"><h3>consultingが示すこと</h3><p>形式的なV*を設けず、事実・敗北リスク・勝利寄与・資源・次ターンの選択肢を比較した条件が、最高の勝率60.0%と終端スコア935.3を示した。</p></div>
    """),
    ("まとめ", """
      <h2>まとめ：成果と次の改訂</h2>
      <div class="panel red"><p class="big">現行HIVC-Dは無効出力・再試行を減らした一方、勝率・終端スコア・regretの優位は確認できず、行動合意率は下がった。</p></div>
      <div class="cols"><div class="panel"><h3>改訂案：C比較</h3><p>V整合とA確認の間で、各候補の予測状態差分・即時失敗条件・勝利進捗・機会費用を表で比較する。</p></div><div class="panel"><h3>限界</h3><p>単一モデル・単一ゲーム・単一役割ペア。検出力設計と多重比較補正なし。人間の心理的安全性への外挿は未検証。</p></div></div>
      <p class="muted">検証対象は今回のI→V→A実装。HIVC-Dの概念一般を否定する結論ではない。</p>
    """),
    ("結び", """
      <div class="section-no">06 / 結び</div><h1>ご清聴ありがとうございました</h1>
      <div class="panel"><p class="big">問い：価値の合意を、どう具体的な行動比較へつなげるか。</p></div>
      <p class="speaker">神山まるごと高専 本科2年　鈴木陽向</p>
      <p class="muted">出典：HIVCD_GLM47_INTERIM.md、analysis/glm47-final-90/、既存のゲームルール・実験結果・合意率分析スライド</p>
    """),
]


EXTRA_CSS = """
  /* The base CSS above is copied verbatim from experiment_results_slides.html. */
  .eyebrow, .section-no { color: var(--accent); font-size: 22px; font-weight: 700; margin-bottom: 22px; }
  .cover-title { font-size: 52px; line-height: 1.25; }
  .subtitle { font-size: 22px; color: var(--muted); }
  .cover-rule { width: 150px; border-top: 5px solid var(--accent); margin: 34px 0; }
  .speaker { font-size: 24px; line-height: 1.7; }
  .section-desc { font-size: 25px; line-height: 1.6; margin-top: 12px; }
  .agenda { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }
  .agenda div { border: 1.5px solid #ccc; border-top: 4px solid var(--accent); padding: 30px 25px; border-radius: 6px; }
  .agenda b { display: block; color: var(--accent); font-size: 30px; }
  .agenda span { font-size: 26px; }
  .three > .panel { min-width: 0; }
  .flow { display: flex; gap: 13px; align-items: center; margin: 12px 0 24px; }
  .flow > div { flex: 1; border: 1.5px solid #ccc; border-top: 4px solid var(--accent); padding: 22px 15px; text-align: center; font-size: 22px; font-weight: 700; min-height: 100px; }
  .flow > span { font-size: 28px; color: var(--accent); }
  .flow small { display: block; font-size: 16px; font-weight: 400; margin-top: 8px; }
  .flow .red-text { border-top-color: var(--accent2); color: var(--accent2); }
  .mountain { border: 1.5px solid #ccc; padding: 22px 30px; margin-bottom: 18px; }
  .summit { text-align: center; font-size: 28px; font-weight: 700; margin-bottom: 25px; }
  .route-line { display: flex; justify-content: space-between; gap: 28px; }
  .route-line span { flex: 1; border: 3px solid var(--accent); padding: 20px; text-align: center; font-size: 21px; font-weight: 700; }
  .route-line .route-a { border-color: var(--accent2); color: var(--accent2); }
  .route-line .route-b { color: var(--accent); }
  .hero-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 18px; }
  .hero-grid div { border: 1.5px solid #ccc; border-top: 4px solid var(--accent); padding: 20px; text-align: center; }
  .hero-grid b { display: block; font-size: 55px; color: var(--accent); }
  .hero-grid span { font-size: 17px; }
  .compact th, .compact td { padding: 5px 7px; font-size: 15px; }
  ol { padding-left: 28px; }
  .slide > .muted:last-child { margin-top: 8px; font-size: 15px; }
"""


def build():
    total = len(SLIDES)
    sections = []
    for i, (section, body) in enumerate(SLIDES, 1):
        footer = f"2026年度情報処理学会 関西支部 支部大会｜{section}‖{i} / {total}"
        sections.append(f'<section class="slide" data-footer="{footer}">{body}</section>')
    html = (f'<!DOCTYPE html>\n<html lang="ja">\n<head>\n<meta charset="UTF-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f'<title>{TITLE}</title>\n<style>{BASE_CSS}\n{EXTRA_CSS}</style>\n</head>\n'
            f'<body><div class="deck">\n' + "\n".join(sections) +
            f'\n</div><script>{FIT_SCRIPT}</script></body></html>\n')
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT} ({total} slides)")


if __name__ == "__main__":
    build()
