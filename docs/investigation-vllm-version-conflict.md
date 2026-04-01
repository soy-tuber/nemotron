# vLLM バージョン互換性問題 調査レポート

## 問題

| vLLM バージョン | Nemotron 9B Japanese | Nemotron 12B VL |
|---|---|---|
| 最新版 (0.16+) | NG: `<think>` 内で無意味な文字列をループ | OK |
| 0.15 (安定版) | OK | NG |

## 根本原因の分析

### 最有力: カスタムパーサーの import path が 0.16+ で壊れる

HuggingFace 配布の `nemotron_nano_v2_reasoning_parser.py` は vLLM 内部モジュールを直接 import している:
```python
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest, DeltaMessage, ResponsesRequest
)
```
0.16 以降で vLLM がモジュール構造を変更した場合、**パーサーがサイレントにロード失敗**する。
パーサーが動かない → `</think>` を検出できない → 無限ループ。

### 背景: `</think>` トークンが EOS セットに含まれていない

Nemotron Nano モデルの特殊トークン構成:
- `</think>` = token ID 13
- EOS token set = {2, 11}

**reasoning parser が `</think>` を正しく検出できない限り、モデルに停止条件がなく無限ループする。**
これは vLLM 外のフレームワーク (mlx-lm) でも同じ問題が報告されている ([mlx-lm#1050](https://github.com/ml-explore/mlx-lm/issues/1050))。

### 0.16 で `nemotron_v3` パーサーが built-in 化

vLLM 0.16 で `nemotron_v3` パーサーが組み込みになったが、**Nemotron 3 Super 用であり Nano は対象外**。
built-in パーサーの自動検出ロジック変更が、カスタムパーサーの読み込み順序やフォールバックに影響している可能性がある。

### 0.17 で structured output + reasoning parser の衝突

vLLM 0.17 で、guidance 構造化出力の FSM が `<think>` 中のトークンにも制約をかけ始める問題が報告 ([#37362](https://github.com/vllm-project/vllm/issues/37362))。
`<think>` 内のテキストが JSON 形式に強制され、ゴミ出力になる。

### `<think>` タグの二重付与の可能性

新しい vLLM が chat template 経由で `<think>` を自動付与 + モデル側のテンプレートでも付与 → 二重 `<think>`。
Qwen3 で同じバグが報告済み ([#27118](https://github.com/vllm-project/vllm/issues/27118))。

## Nemotron Nano の思考制御方式 (参考)

**重要**: Nemotron-Nano-8B-v1 は vLLM の `enable_thinking` パラメータを使わない。
system prompt の内容で制御する:

- `"detailed thinking on"` → `<think>...</think>` を出力
- `"detailed thinking off"` → 推論なし (greedy decoding 推奨)

推奨サンプリング: reasoning ON → `temperature=0.6, top_p=0.95` / reasoning OFF → `temperature=0`

## 0.15 → 最新で関連する変更

| バージョン | 変更点 |
|---|---|
| 0.15.0 | 大きな reasoning parser 変更なし (安定動作の基準) |
| 0.16.0 | `nemotron_v3` パーサー built-in 化 (Super 用、Nano 対象外); 内部モジュール再構成の可能性 |
| 0.17.0 | Qwen3 reasoning parser 修正; structured output + reasoning 衝突; Nemotron NVFP4 精度修正 |

## 確認手順 (優先順)

### Step 1: カスタムパーサーの import エラーを確認
```bash
# 最新 vLLM 環境で実行
python -c "from nemotron_nano_v2_reasoning_parser import *"
```
import エラーが出たら、パーサーを現行 vLLM の API に合わせて修正する。
**これが最も有力な原因。**

### Step 2: verbose prompt で二重 `<think>` を確認
```bash
vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese --verbose-prompt
```
プロンプトに `<think>` が2回含まれていないか確認。

### Step 3: パーサーなしで `stop_token_ids` を使って起動
```bash
vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese \
  --override-pooling-config '{}' 
```
API リクエストで `stop_token_ids: [13]` を指定し、`</think>` で強制停止させる。

### Step 4: thinking token budget でループ防止
```python
extra_body={"thinking_token_budget": 2048}
```

### Step 5: repetition_penalty を明示指定
```python
repetition_penalty=1.1
```

## 長期的な解決策の選択肢

### A. カスタムパーサーを最新 vLLM API に移植 (推奨)
- import path を 0.16/0.17 の API に合わせて修正
- 一度直せば単一 vLLM インスタンスで全モデル動作
- vLLM メジャーアップデート時にメンテが必要

### B. パーサーを使わず `stop_token_ids` + ゲートウェイ後処理で制御
- `</think>` (token 13) を stop token に追加
- ゲートウェイ側で `<think>...</think>` を後処理で分離
- vLLM のパーサー機構に依存しなくなる (最も堅牢)

### C. vLLM インスタンス 2 つ構成
- port 8100: vLLM 0.15 → Nemotron Japanese
- port 8101: vLLM 最新版 → Nemotron VL
- ゲートウェイが models.yaml のモデル定義に基づいてポートを振り分け
- VRAM 排他制御は既存のまま
- 2つの vLLM バージョンを管理する運用コスト

## 関連 Issue / PR

### 直接関連 (0.15→最新の変更)
- [vllm#37362](https://github.com/vllm-project/vllm/issues/37362) — nemotron_v3 パーサー + Guidance 構造化出力の衝突 (0.17)
- [vllm#34476](https://github.com/vllm-project/vllm/issues/34476) — Nemotron NVFP4 精度修正 (0.17)
- [vllm#27118](https://github.com/vllm-project/vllm/issues/27118) — `<think>` 二重付与 (Qwen3)

### 背景情報
- [vllm#23430](https://github.com/vllm-project/vllm/issues/23430) — reasoning parser 自動有効化、無効化不可
- [vllm#30139](https://github.com/vllm-project/vllm/issues/30139) — reasoning models が completion を reasoning_content に誤配置
- [vllm#15068](https://github.com/vllm-project/vllm/issues/15068) — Nemotron-Nano-8B サポートリクエスト
- [vllm#13025](https://github.com/vllm-project/vllm/pull/13025) — reasoning parser 書き直し
- [vllm#33402](https://github.com/vllm-project/vllm/pull/33402) — `reasoning_content` → `reasoning` 移行
- [vllm#32713](https://github.com/vllm-project/vllm/issues/32713) — RFC: 統一パーサー提案
- [mlx-lm#1050](https://github.com/ml-explore/mlx-lm/issues/1050) — Nemotron thinking 無限ループ (他フレームワーク)
- [HuggingFace Discussion](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/discussions/3) — Tool calling + reasoning parsing broken
