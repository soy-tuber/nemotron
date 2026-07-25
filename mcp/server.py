"""
Nemotron MCP Server — Claude Code から vLLM Gateway を呼び出す
"""

import json

import httpx
from mcp.server.fastmcp import FastMCP

GATEWAY_URL = "http://localhost:8000"

# Gateway 側のモデル起動許容は STARTUP_TIMEOUT_SECONDS=600 秒。
# コールドスタートを待ち切れるよう余裕を持たせる
CHAT_TIMEOUT = 660
STATUS_TIMEOUT = 10

mcp = FastMCP("nemotron")


async def _post_json(path: str, data: dict, timeout: int = CHAT_TIMEOUT) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{GATEWAY_URL}{path}", json=data)
        resp.raise_for_status()
        return resp.json()


async def _get_json(path: str, timeout: int = STATUS_TIMEOUT) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(f"{GATEWAY_URL}{path}")
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
async def nemotron_chat(
    prompt: str,
    system_prompt: str = "",
    max_tokens: int = 4096,
    temperature: float = 0.4,
    model: str = "",
    think: bool = False,
) -> str:
    """Nemotron (vLLM) にプロンプトを送信してレスポンスを返す。
    大量テキスト処理、日本語要約、データ抽出など力仕事向け。
    Gateway がアイドル時は自動起動する（初回は数分かかる場合あり）。

    Args:
        prompt: ユーザープロンプト
        system_prompt: システムプロンプト（省略可）
        max_tokens: 最大出力トークン数
        temperature: 温度パラメータ (0.0-1.0)
        model: モデル名（省略時はデフォルト nemotron-9b-japanese）
        think: 推論（thinking）を有効にする。既定は無効。
            抽出・要約・整形などの力仕事では不要で、出力トークンと
            レイテンシを無駄に消費するため。多段の論理を要する場合のみ True
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if model:
        payload["model"] = model
    if not think:
        # chat_template の `enable_thinking is defined and not enable_thinking`
        # 分岐に入り、空の <think></think> が挿入されて推論がスキップされる。
        # システムプロンプトへの /no_think は効かない（実測で推論が出続けた）
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    try:
        result = await _post_json("/v1/chat/completions", payload)
    except Exception as e:
        # httpx.ReadTimeout は str(e) が空になるため型名でフォールバック
        return f"[ERROR] Nemotron 呼び出し失敗: {str(e) or type(e).__name__}"

    choice = result.get("choices", [{}])[0]
    msg = choice.get("message", {})
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""

    parts = []
    if reasoning:
        parts.append(f"<reasoning>\n{reasoning}\n</reasoning>")
    if content:
        parts.append(content)
    if not parts:
        return f"[WARN] 空レスポンス: {json.dumps(result, ensure_ascii=False)[:500]}"
    return "\n\n".join(parts)


@mcp.tool()
async def nemotron_status() -> str:
    """Nemotron Gateway の状態を確認する（モデル稼働状況、アイドル時間など）"""
    try:
        status = await _get_json("/gateway/status")
        lines = [
            f"モデル: {status.get('current_model') or '(未ロード)'}",
            f"稼働中: {status.get('running', False)}",
            f"Ready: {status.get('ready', False)}",
            f"起動中: {status.get('starting', False)}",
            f"アクティブリクエスト: {status.get('active_requests', 0)}",
            f"アイドル: {status.get('idle_seconds', '-')}秒",
            f"アイドルタイムアウト: {status.get('idle_timeout', '-')}秒",
            f"PID: {status.get('pid') or '-'}",
            f"利用可能モデル: {', '.join(status.get('available_models', []))}",
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"[ERROR] Gateway 接続失敗: {str(e) or type(e).__name__}"


if __name__ == "__main__":
    mcp.run()
