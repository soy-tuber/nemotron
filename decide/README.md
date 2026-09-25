# decide — 閉じた選択肢だけを返す推論 / Closed-set inference

文章を書かせない。**決められた選択肢の中から1つ選ばせ、その確率分布を返す。**

The model writes nothing. It **picks one of a fixed option set, and what comes
back is the probability distribution over that set.**

```console
$ decide/.venv/bin/python -m decide.cli route_inquiry --var input="先月の請求が二重に引き落とされています"
route_inquiry: billing
  confidence 0.766   coverage 0.950   198ms   [logprob]
  billing       0.766 ███████████████████████
  technical     0.232 ███████
  other         0.001
  sales         0.001
  account       0.000
```

2026-09-25 実測 / measured: RTX 5090, vLLM 0.15.1, `nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese`.

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
2. 思考をオフにして数トークンだけ生成させ、`top_logprobs`(既定20件)を受け取る。
   Generate a handful of tokens with thinking off and read `top_logprobs` (20 by default).
3. ラベル以外のトークンを捨て、**ラベル集合の上で再正規化**する。
   Drop every non-label token and **renormalise over the label set**.
4. 分布の最大値を選択肢として返す。生成テキストは読まない。
   Return the argmax. The generated text is never parsed.

先頭に引用符や改行が来た場合に備えて、数トークン分を前方に走査する。
ただし**思考(reasoning)は必ず切る**。このモデルのチャットテンプレートが受け付けるのは
`chat_template_kwargs: {"enable_thinking": false}` だけで、プロンプトに `/no_think` と
書いても効かない。思考が入ると最初の数トークンが英語の推論になり、ラベルの確率が
ほとんど残らない(実測で coverage 0.002、棄権)。

The scan walks forward a few positions past a stray quote or newline. **Thinking
must be off**, though: this model's chat template only honours
`chat_template_kwargs: {"enable_thinking": false}`; a `/no_think` in the prompt is
ignored. With thinking on, the first tokens are English reasoning and almost no
label mass is left (measured coverage 0.002, abstained).

### coverage — 「ちゃんと答えようとしたか」の指標 / did it even try

再正規化**前**のラベル総質量。低い = モデルはラベル形式で答える気がなかった、
つまり質問が伝わっていない。`min_coverage` を下回れば棄権する。

The label mass **before** renormalisation. Low coverage means the model was not
answering in label form at all — the question did not land. Below
`min_coverage` the call abstains.

### 制約デコードは保険であって本命ではない / constrained decoding is the fallback

top-k にラベルが1つも現れなかった場合だけ、vLLM の `structured_outputs`(`choice`)で
**形式だけは**保証して答えを取る(`method: "constrained"`、`masked: true`)。
この経路の `confidence` は閾値判定に使わない。結果の `warnings` がそれを示す。

Only when no label appears in the top-k do we re-ask with vLLM's
`structured_outputs` (`choice`), which guarantees **the format alone**
(`method: "constrained"`, `masked: true`). Do not threshold on its
`confidence`; an entry in `warnings` marks it.

vLLM 0.15.1 が受け付けるのは `structured_outputs` だけ。旧名の `guided_choice` は
エラーにならず**黙って無視される**ので、エラーを見て旧形式に切り替えることはしない。
なお 0.15.1 で実測すると、制約付きの経路でも確率は制約なしとほぼ同じだった
(billing 0.775 と 0.766)。それでも保守的に `masked: true` として扱う。

vLLM 0.15.1 accepts only `structured_outputs`; the old `guided_choice` is
**silently ignored**, not rejected, so there is no error to fall back on. On
0.15.1 the constrained path measured almost the same probabilities as the
unconstrained one (billing 0.775 vs 0.766); it is still flagged `masked: true`.

## 使い方 / Usage

コマンドは `~/nemotron` で実行する / Run the commands from `~/nemotron`.

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
PY=decide/.venv/bin/python
$PY -m decide.cli --list
$PY -m decide.cli diff_risk --var input="$(git diff --cached)"
$PY -m decide.cli urgency --var input=- < ticket.txt        # 標準入力 / stdin
$PY -m decide.cli --inline "これは苦情か" -o yes -o no --var input=...
$PY -m decide.cli route_inquiry --var input=... --json      # 機械向け / for pipes
```

モデルは既定で `nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese`(`--model` か
`NEMOTRON_DECIDE_MODEL` で変更)。vLLM は `--served-model-name` なしで起動しているので、
短い名前 `nemotron-9b-japanese` は使えない(ゲートウェイがモデルを起動したうえで vLLM が 404)。

The default model is the full id `nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese`.
vLLM runs without `--served-model-name`, so the short alias does not work: the
gateway loads the model and vLLM then answers 404.

### HTTP (port 9200)

`nemotron-decide.service` として常駐し、`127.0.0.1:9200` だけで待ち受ける。
Runs as `nemotron-decide.service`, listening on `127.0.0.1:9200` only.

```bash
curl -s localhost:9200/v1/decide -H 'content-type: application/json' -d '{
  "decision": "route_inquiry",
  "variables": {"input": "ログインできません"}
}'
```

```json
{
  "decision": "route_inquiry",
  "choice": "account",
  "confidence": 0.999811,
  "coverage": 0.997916,
  "distribution": {"account": 0.999811, "technical": 0.000137, "sales": 3.5e-05, "other": 1.7e-05, "billing": 0.0},
  "abstain": false,
  "method": "logprob",
  "masked": false,
  "model": "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese",
  "latency_ms": 84.5,
  "reason": null,
  "expected_value": null,
  "warnings": []
}
```

| Endpoint | 用途 / Purpose |
|---|---|
| `POST /v1/decide` | 1件。`decision` 名か、その場定義の `inline` / One decision, by catalogue name or `inline` |
| `POST /v1/decide/batch` | 複数を並列実行。1件の失敗は他に波及しない / Parallel; one failure does not sink the rest |
| `GET /v1/decisions` | カタログ一覧と必要変数 / Catalogue with required variables |
| `GET /healthz` | 接続先ゲートウェイとモデル / Gateway and model in use |

### 常駐 / systemd

ユニットは `/etc/systemd/system/` に置く(`~/.config/systemd/user/` は使わない)。
依存は `decide/.venv` に別建てで、vLLM の venv とは独立している。

The unit lives in `/etc/systemd/system/`; dependencies sit in their own
`decide/.venv`, independent of vLLM's.

```bash
uv venv decide/.venv --python 3.12
uv pip install --python decide/.venv/bin/python -r requirements-decide.txt
sudo cp decide/nemotron-decide.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now nemotron-decide
```

### 他のPCから / From other machines

判断のロジックはクライアント側にあるので、tailnet 上の別のPCでもライブラリや CLI を
そのまま使える。接続先をゲートウェイの Tailscale Serve にする(実測 104ms)。

The decision logic is client-side, so the library and CLI run on any machine in
the tailnet; point them at the gateway's Tailscale Serve URL (measured 104 ms).

```bash
NEMOTRON_GATEWAY_URL=https://desktop-tgi1k49.tailf198cb.ts.net:8000/v1 \
  python -m decide.cli urgency --var input="本番サーバーが全停止しています"
```

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
**独立した判断をvLLMがまとめて捌けるから**。

The speed does not come from a faster model. It comes from generating a handful
of tokens instead of a paragraph, and from vLLM batching independent decisions.

| 同時実行 / Concurrency | 件/秒 / rps | p50 | p95 |
|---|---|---|---|
| 1 | 11.6 | 86 ms | 89 ms |
| 8 | 28 | 285 ms | 330 ms |
| 32 | 36 | 752 ms | 1.2 s |

- モデルが止まっているときの最初の1件は約40秒(ゲートウェイがモデルを起動する)。
  The first decision after the model was unloaded takes ~40 s while the gateway loads it.
- 同時32件は、モデル起動後の最初の1回だけ遅い(19件/秒)。
  Concurrency 32 is slower once, right after a model load (19 rps).

```bash
decide/.venv/bin/python -m decide.bench --decision route_inquiry --repeat 64 --concurrency 1 8 32
```

## 精度の目安 / A small sanity check

想定答えを付けた11件で、9件が一致(2026-09-25)。外れた2件:

On 11 hand-labelled samples, 9 matched (2026-09-25). The two misses:

| 判断 / Decision | 入力 / Input | 想定 / Expected | 結果 / Got |
|---|---|---|---|
| urgency | 来月以降の請求書の宛名を変更したい | 1–2 | 3 (0.97) |
| diff_risk | README の誤字修正だけの差分 | safe | high (0.80) |

`diff_risk` の外れは確信度 0.80 で、下限 0.70 を超えるので棄権もしない。
pre-commit に組み込むのは、実データで閾値を決めてからにする。

The `diff_risk` miss came with confidence 0.80, above its 0.70 floor, so it did
not abstain. Set thresholds from real data before wiring it into pre-commit.

## 制限 / Limits

- 選択肢は**26個まで**(`top_logprobs` の上限20件を超えると分布が切れる)。
  At most **26 options** — beyond the 20-entry `top_logprobs` ceiling the tail is lost.
- 尺度は**1桁**(0–9)。2桁は1トークンに収まらない。
  Scales are **single-digit** (0–9); two digits do not fit in one token.
- `confidence` は較正済みの確率ではなく**モデルの自信**。閾値は実データで決める。
  `confidence` is the model's own confidence, not a calibrated probability — set thresholds from real data.
- 思考は `chat_template_kwargs` で切る。`thinking=True` にすると先頭が推論になり、
  ほぼ必ず棄権する。
  Thinking is switched off through `chat_template_kwargs`; with `thinking=True` the
  first tokens are reasoning and the call almost always abstains.

## テスト / Tests

GPUもゲートウェイも要らない。`httpx.MockTransport` で偽の `logprobs` を流す。
モックなので、実機との食い違い(モデル名やテンプレートの仕様)は検出できない。
変更したら上の CLI を実機で一度流して確かめる。

No GPU and no gateway required — `httpx.MockTransport` feeds in fake `logprobs`.
Being mocks, they cannot catch a mismatch with the real stack (model name,
template behaviour); run the CLI above against the gateway after changes.

```bash
uv pip install --python decide/.venv/bin/python pytest
decide/.venv/bin/python -m pytest
```
