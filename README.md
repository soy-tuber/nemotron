# Nemotron Local Multimodal Gateway

ローカルのNVIDIA Nemotron 9Bを起点に、Vision・Parse・ASR・VoiceChatを **1つのゲートウェイ(port 8000)** で束ねるマルチモーダル基盤。

A local multimodal LLM infrastructure that unifies Vision, Parse, ASR, and VoiceChat behind **a single gateway (port 8000)**, starting from NVIDIA Nemotron 9B.

## 発想 / Concept

Nemotronは単体ではテキストLLMだが、NVIDIAはNemotronファミリーとして複数のモダリティ特化モデルを公開している。
これらを **1台のRTX 5090上でオンデマンドに切り替え** ながら使えば、ローカルで完結するマルチモーダルLLMインフラが作れる。

Nemotron alone is a text-only LLM, but NVIDIA publishes multiple modality-specific models under the Nemotron family.
By **swapping them on-demand on a single RTX 5090**, you get a fully local multimodal LLM infrastructure.

- テキスト推論 / Text inference → Nemotron 9B Japanese (18GB VRAM)
- 画像理解 / Image understanding → Nemotron 12B VL (24GB VRAM)
- 文書パース / Document parsing → Nemotron Parse (3GB VRAM)
- 音声認識 / Speech recognition → Nemotron Speech ASR (planned)
- 音声対話 / Voice chat → Nemotron VoiceChat (planned)

VRAMは有限なので同時ロードはしない。ゲートウェイが **リクエストに応じてモデルを入れ替え** 、アイドル10分で自動アンロードする。
外部から見れば常に `localhost:8000` のOpenAI互換APIが1本あるだけ。

VRAM is finite — models are never loaded simultaneously. The gateway **hot-swaps models per request** and auto-unloads after 10 minutes of idle.
From the outside, it's just a single OpenAI-compatible API at `localhost:8000`.

## ディレクトリ構造 — 集約の設計 / Directory Structure — Consolidation by Design

```
nemotron/
├── gateway/          # 統合ゲートウェイ / Unified gateway (FastAPI)
│   ├── gateway.py    #   振り分け・モデル切替・ToolCall書換 / Routing, model swap, ToolCall rewrite
│   └── models.yaml   #   モデルレジストリ / Model registry
├── parsers/          # vLLMカスタムパーサー / Custom vLLM parsers (→ symlink)
│   ├── nemotron_nano_v2_reasoning_parser.py   # <think>タグ抽出 / Thinking extraction
│   ├── nemotron_tool_parser.py                # ToolCall解析 / ToolCall parsing
│   └── nemotron_toolcall_parser_streaming.py  # ストリーミングToolCall / Streaming ToolCall (production)
├── services/         # systemdユニットファイル / systemd unit files (→ symlink)
│   ├── nemotron-gateway.service
│   ├── nemotron-vllm.service
│   ├── nemotron-api.service
│   └── nemotron-web.service
├── models/           # モデルウェイト / Model weights (→ HuggingFace cache symlink)
├── cache/            # vLLMキャッシュ / vLLM cache (→ ~/.cache/vllm)
└── config/           # vLLM設定 / vLLM config (→ ~/.config/vllm)
```

実体は `~/.config/systemd/user/`、`~/.cache/vllm/`、HuggingFace cacheなどに散在しているが、
**symlinkで1ディレクトリに集約** することで、Nemotron関連の全資産を一望できるようにしている。

The actual files live scattered across `~/.config/systemd/user/`, `~/.cache/vllm/`, HuggingFace cache, etc.
**Symlinks consolidate everything into one directory**, making all Nemotron-related assets visible at a glance.

「このディレクトリだけ見ればNemotronの全体像がわかる」が設計目標。

Design goal: **"Look at this one directory to understand the entire Nemotron setup."**

## アーキテクチャ / Architecture

```
Client (OpenAI API)
  │
  ▼
Gateway (port 8000)           ← モデル切替・ToolCall書換・アイドル監視
  │                              Model swap, ToolCall rewrite, idle watchdog
  ▼
vLLM (port 8100, subprocess)  ← 推論エンジン + カスタムパーサー
  │                              Inference engine + custom parsers
  ▼
Model (GPU, VRAM)             ← オンデマンドでロード/アンロード
                                 On-demand load/unload
```

### ゲートウェイの役割 / Gateway Responsibilities

- **モデルレジストリ / Model registry** — `models.yaml`に全モデルを定義。フレームワーク・VRAM・起動引数を宣言的に管理 / All models declared in `models.yaml` with framework, VRAM, and launch args
- **オンデマンドロード / On-demand loading** — リクエスト先のモデルが未ロードなら自動起動、別モデルがロード中なら入れ替え / Auto-starts the requested model; swaps out a different one if loaded
- **アイドル停止 / Idle shutdown** — 10分間リクエストがなければvLLMプロセスを自動停止してVRAM解放 / Kills the vLLM process and frees VRAM after 10 min of inactivity
- **ToolCall書き換え / ToolCall rewrite** — Nemotronの `<TOOLCALL>` XML形式をOpenAI互換の `tool_calls` に変換（モデルごとに `toolcall_rewrite: true` で制御） / Converts Nemotron's `<TOOLCALL>` XML into OpenAI-compatible `tool_calls` (per-model opt-in via `toolcall_rewrite: true`)
- **スワップ競合制御 / Swap contention control** — アクティブリクエスト処理中のモデル入れ替えをブロックし、`503 + Retry-After` を返す / Blocks model swap while requests are in flight; returns `503 + Retry-After`

### カスタムパーサー / Custom Parsers

vLLMのプラグイン機構で注入する3つのパーサー:

Three parsers injected via vLLM's plugin system:

| パーサー / Parser | 役割 / Role |
|---------|------|
| `reasoning_parser` | `<think>...</think>` をストリーミング中にリアルタイム分離し `reasoning_content` フィールドへ出力 / Real-time streaming separation of thinking tags into the `reasoning_content` field |
| `tool_parser` | `<TOOLCALL>` を検出してOpenAI ToolCallオブジェクトに変換 / Detects `<TOOLCALL>` and converts to OpenAI ToolCall objects |
| `tool_parser (streaming)` | 部分JSONの差分送信、30文字先読みバッファ、マルチツール対応 / Partial JSON delta streaming, 30-char lookahead buffer, multi-tool support |

ストリーミングToolCallパーサーは、不完全なJSONフラグメントを `partial_json_parser` で復元しながら差分だけを送出する。

The streaming ToolCall parser reconstructs incomplete JSON fragments via `partial_json_parser` and emits only deltas.

## 動作環境 / Requirements

- Ubuntu 24.04 (WSL2)
- RTX 5090 (32GB VRAM)
- Python 3.12 / vLLM / FastAPI
- systemdユーザーサービスで常駐 / Runs as systemd user services

## 登録モデル / Registered Models

| モデル名 / Model | Params | VRAM | モダリティ / Modality | 状態 / Status |
|---------|--------|------|-----------|------|
| `nemotron-9b-japanese` | 9B | 18GB | Text + ToolCall + Reasoning | Active |
| `nemotron-12b-v2-vl` | 12B | 24GB | Image + Text (Vision) | Active |
| `nemotron-parse` | — | 3GB | Document structure parsing | Active |
| `nemotron-speech-asr` | — | 2GB | Speech recognition (NeMo) | Planned |
| `nemotron-voicechat` | — | 24GB | Voice chat (NeMo) | Planned |

## ライセンス / License

個人の実験プロジェクトです。各モデルのライセンスはNVIDIAの公開条件に従います。

Personal experimental project. Each model is subject to NVIDIA's respective license terms.
