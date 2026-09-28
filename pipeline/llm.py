"""One-shot Claude call through the Claude Agent SDK (runs on a logged-in Claude Code install, no API key)."""
import tempfile
import time
from dataclasses import dataclass

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock, query

MODELS = {"haiku": "claude-haiku-4-5", "sonnet": "claude-sonnet-5", "opus": "claude-opus-5"}


@dataclass
class Completion:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    duration_s: float
    is_error: bool


async def complete(prompt: str, system: str, model: str) -> Completion:
    """One-shot call through the Claude Code CLI: no tools, no CLAUDE.md, no settings."""
    options = ClaudeAgentOptions(
        model=MODELS[model],
        system_prompt=system,
        tools=[],
        setting_sources=[],
        skills=[],
        max_turns=1,
        strict_mcp_config=True,
        cwd=tempfile.gettempdir(),
    )
    start = time.monotonic()
    parts, served_by, result = [], None, None
    async for msg in query(prompt=prompt, options=options):
        if isinstance(msg, AssistantMessage):
            served_by = msg.model
            parts += [b.text for b in msg.content if isinstance(b, TextBlock)]
        elif isinstance(msg, ResultMessage):
            result = msg
    usage = result.usage or {}
    return Completion(
        text="".join(parts),
        model=served_by,
        input_tokens=usage.get("input_tokens", 0)
        + usage.get("cache_read_input_tokens", 0)
        + usage.get("cache_creation_input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        duration_s=time.monotonic() - start,
        is_error=result.is_error,
    )
