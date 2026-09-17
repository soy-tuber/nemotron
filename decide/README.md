# decide — 閉じた選択肢だけを返す推論 / Closed-set inference

文章を書かせない。**決められた選択肢の中から1つ選ばせ、その確率分布を返す。**

The model writes nothing. It **picks one of a fixed option set, and what comes
back is the probability distribution over that set.**

```console
$ python -m decide.cli route_inquiry --var input="先月の請求が二重に引き落とされています"
route_inquiry: billing
  confidence 0.912   coverage 0.981   118ms   [logprob]
  billing       0.912 ███████████████████████████
  technical     0.041 █
  account       0.026 █
  sales         0.014
  other         0.007
```

## なぜローカルで作るのか / Why build this locally

この種の「判断専用API」はクラウド側にいくつかあるが、重みが公開されていない
ものは**ローカルでは動かせない**。動かせないのは実装であって、発想ではない。
同じ保証は、vLLMが返す `logprobs` の上に自分で組める。

Hosted "decision-only" APIs exist, but the ones that do not publish weights
**cannot be run locally at all**. What cannot be moved is the implementation,
not the idea — the same guarantee is buildable on top of the `logprobs` vLLM
already returns.

## 何が保証されて、何が保証されないか / What is and is not guaranteed

| | |
|---|---|
| **形式 / Format** | 保証される。返り値は分布から導出するので、カタログ外の値は原理的に出ない / Guaranteed. The answer is derived from the distribution, so an off-catalogue value cannot occur |
| **中身 / Substance** | 保証されない。自信を持って間違えることはある / Not guaranteed. It can be confidently wrong |

だから `confidence` と `coverage` と棄権(abstain)がある。「幻覚がない」とは
**形式の誤りがない**という意味であって、判断が常に正しいという意味ではない。
ここを混同しないことが、この方式を実運用に載せる前提になる。

Hence `confidence`, `coverage`, and the abstain path. "No hallucination" means
**no malformed output** — not that the judgement is right. Keeping those two
apart is the precondition for putting this in production.

## 仕組み / How it works

1. 各選択肢に**1トークンのラベル**(`A`, `B`, … / 尺度なら `1`…`5`)を割り当てる。
   Each option gets a **single-token label**.
2. 数トークンだけ生成させ、`top_logprobs`(既定20件)を受け取る。
   Generate a handful of tokens and read `top_logprobs` (20 by default).
3. ラベル以外のトークンを捨て、**ラベル集合の上で再正規化**する。
   Drop every non-label token and **renormalise over the label set**.
4. 分布の最大値を選択肢として返す。生成テキストは読まない。
   Return the argmax. The generated text is never parsed.

生成テキストを読まないので、`<think>` が混ざろうが引用符が付こうが壊れない。
先頭に無関係なトークンが来た場合に備えて、数トークン分を前方に走査する。

Because the text is never parsed, a stray `<think>`, quote, or newline cannot
break it; the scan walks forward a few positions to find the informative one.

### coverage — 「ちゃんと答えようとしたか」の指標 / did it even try

再正規化**前**のラベル総質量。低い = モデルはラベル形式で答える気がなかった、
つまり質問が伝わっていない。`min_coverage` を下回れば棄権する。

The label mass **before** renormalisation. Low coverage means the model was not
answering in label form at all — the question did not land. Below
`min_coverage` the call abstains.

### 制約デコードは保険であって本命ではない / constrained decoding is the fallback

top-k にラベルが1つも現れなかった場合だけ、vLLMのguided decodingで
**形式だけは**保証して答えを取る(`method: "constrained"`)。ただしこの経路の
`confidence` は**マスク後の分布**から出るため常に高く出る。閾値判定に使っては
いけない。結果の `warnings` と `calibrated: false` がそれを示す。

Only when no label appears in the top-k do we re-ask with guided decoding, which
guarantees **the format alone** (`method: "constrained"`). The `confidence` from
that path is computed over a *masked* distribution, so it is inflated by
construction and must not be thresholded on — `calibrated: false` and an entry
in `warnings` mark it.

vLLMは新しい版で `guided_choice` を `structured_outputs` に改名したので、
最初のリクエストが400を返したら自動で旧形式に切り替える。

Newer vLLM renamed `guided_choice` to `structured_outputs`; a 400 on the first
attempt switches to the legacy field automatically.

## 使い方 / Usage

### カタログ / The catalogue

判断は `decisions.yaml` に宣言する。ここに無いものは返ってこない。
Decisions are declared in `decisions.yaml`. Nothing outside it can come back.

```yaml
decisions:
  - name: diff_risk
    question: |
      次の変更をレビューなしでマージした場合のリスクはどれか。
      ---
      {input}
      ---
    options:
      - {key: safe,     text: "ドキュメント・整形のみ"}
      - {key: low,      text: "局所的な変更、テスト済み"}
      - {key: high,     text: "挙動・APIの変更"}
      - {key: critical, text: "認証・課金・データ削除に触れる"}
    min_confidence: 0.70
```

`mode: scale` にすると `1`–`5` の尺度になり、勝者に加えて**期待値**
(`expected_value`)が返る。分布そのものが答えになるケース向け。

`mode: scale` turns it into a 1–5 rating and adds an **expected value** on top of
the argmax — for the cases where the distribution itself is the answer.

### CLI

```bash
python -m decide.cli --list
python -m decide.cli diff_risk --var input="$(git diff --cached)"
python -m decide.cli urgency --var input=- < ticket.txt        # 標準入力 / stdin
python -m decide.cli --inline "これは苦情か" -o yes -o no --var input=...
python -m decide.cli route_inquiry --var input=... --json      # 機械向け / for pipes
```

### HTTP (port 8200)

```bash
uvicorn decide.server:app --host 127.0.0.1 --port 8200
```

```bash
curl -s localhost:8200/v1/decide -H 'content-type: application/json' -d '{
  "decision": "route_inquiry",
  "variables": {"input": "ログインできません"}
}'
```

```json
{
  "decision": "route_inquiry",
  "choice": "account",
  "confidence": 0.873,
  "coverage": 0.964,
  "distribution": {"account": 0.873, "technical": 0.094, "other": 0.021, "billing": 0.008, "sales": 0.004},
  "abstain": false,
  "method": "logprob",
  "calibrated": true,
  "latency_ms": 121.4,
  "warnings": []
}
```

| Endpoint | 用途 / Purpose |
|---|---|
| `POST /v1/decide` | 1件。`decision` 名か、その場定義の `inline` / One decision, by catalogue name or `inline` |
| `POST /v1/decide/batch` | 複数を並列実行。1件の失敗は他に波及しない / Parallel; one failure does not sink the rest |
| `GET /v1/decisions` | カタログ一覧と必要変数 / Catalogue with required variables |
| `GET /healthz` | 接続先ゲートウェイとモデル / Gateway and model in use |

### Python

```python
from decide import Decider, DecisionCatalog

catalog = DecisionCatalog.from_yaml("decide/decisions.yaml")

async with Decider() as decider:
    result = await decider.decide(catalog.get("needs_human"), {"input": text})
    if result.abstain:
        queue_for_human(text, result.reason)   # 迷ったら人に渡す / hand it over
    else:
        route(result.choice)
```

## 速度 / Speed

速いのはモデルが速いからではなく、**生成が数トークンで終わるから**と、
**独立した判断をvLLMがまとめて捌けるから**。9B級・RTX 5090で1判断あたり
100–300ms、同時実行で秒間の処理数はさらに伸びる。

The speed does not come from a faster model. It comes from generating a handful
of tokens instead of a paragraph, and from vLLM batching independent decisions.
On a 9B-class model and an RTX 5090, expect 100–300 ms per decision, with
throughput scaling well beyond that under concurrency.

```bash
python -m decide.bench --decision route_inquiry --repeat 50 --concurrency 1 8 32
```

## 制限 / Limits

- 選択肢は**26個まで**(`top_logprobs` の上限20件を超えると分布が切れる)。
  At most **26 options** — beyond the 20-entry `top_logprobs` ceiling the tail is lost.
- 尺度は**1桁**(0–9)。2桁は1トークンに収まらない。
  Scales are **single-digit** (0–9); two digits do not fit in one token.
- `confidence` は較正済みの確率ではなく**モデルの自信**。閾値は実データで決める。
  `confidence` is the model's own confidence, not a calibrated probability — set thresholds from real data.
- 推論モデルは `/no_think` で思考を止めている。`thinking=True` にすると
  先頭トークンが `<think>` になり、判断の質は上がらずレイテンシだけ増える。
  Reasoning is switched off with `/no_think`; turning it on buys latency, not accuracy, for a one-token decision.

## テスト / Tests

GPUもゲートウェイも要らない。`httpx.MockTransport` で偽の `logprobs` を流す。

No GPU and no gateway required — `httpx.MockTransport` feeds in fake `logprobs`.

```bash
pip install -r requirements-decide.txt pytest
pytest
```
