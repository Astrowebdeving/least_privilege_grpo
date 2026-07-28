"""Fake, local-only MCP servers with per-episode isolated state.

Each server instance deep-copies its task fixture at construction, so state
never leaks across episodes or env instances. State lives host-side only: it
is never materialized into /workspace, and the hidden snapshot used by the
mcp_state verifier never travels through any model-facing channel. Call
failures raise MCPCallError with fixture-level messages only (no host paths,
no hidden internals), so the F0 observation stays clean.
"""

from __future__ import annotations

import copy
from typing import Any


class MCPCallError(Exception):
    """Failed-but-executed MCP call; message must stay fixture-level (F0-safe)."""


class MCPCallResult:
    def __init__(
        self,
        payload: Any,
        read_ids: tuple[str, ...] = (),
        written_ids: tuple[str, ...] = (),
        mutated: bool = False,
        secret: bool = False,
    ) -> None:
        self.payload = payload
        self.read_ids = read_ids
        self.written_ids = written_ids
        self.mutated = mutated
        self.secret = secret


def _require(args: dict[str, Any], key: str) -> Any:
    if key not in args or args[key] in (None, ""):
        raise MCPCallError(f"missing required argument: {key}")
    return args[key]


class FakeMCPServer:
    name = "fake"

    def __init__(self, fixture: dict[str, Any] | None) -> None:
        self.state: dict[str, Any] = copy.deepcopy(fixture or {})

    def call(self, tool: str, args: dict[str, Any]) -> MCPCallResult:
        handler = getattr(self, f"tool_{tool}", None)
        if handler is None:
            raise MCPCallError(f"unknown tool: {tool}")
        return handler(args)

    def hidden_state(self) -> dict[str, Any]:
        """Host-side snapshot for verifiers/audit; never model-facing."""
        return copy.deepcopy(self.state)


class GithubStub(FakeMCPServer):
    name = "github_stub"

    def __init__(self, fixture: dict[str, Any] | None) -> None:
        super().__init__(fixture)
        self.state.setdefault("issues", {})

    def _issues(self) -> dict[str, dict[str, Any]]:
        return self.state["issues"]

    def tool_get_issue(self, args: dict[str, Any]) -> MCPCallResult:
        issue_id = str(_require(args, "issue_id"))
        issue = self._issues().get(issue_id)
        if issue is None:
            raise MCPCallError(f"issue not found: {issue_id}")
        return MCPCallResult(
            payload=copy.deepcopy(issue),
            read_ids=(f"mcp:github_stub/issues/{issue_id}",),
        )

    def tool_list_issues(self, args: dict[str, Any]) -> MCPCallResult:
        issues = self._issues()
        payload = [
            {"id": issue_id, "title": issue.get("title", ""), "state": issue.get("state", "open")}
            for issue_id, issue in sorted(issues.items())
        ]
        return MCPCallResult(
            payload=payload,
            read_ids=tuple(f"mcp:github_stub/issues/{issue_id}" for issue_id in sorted(issues)),
        )

    def tool_create_issue(self, args: dict[str, Any]) -> MCPCallResult:
        title = str(_require(args, "title"))
        body = str(args.get("body", ""))
        issues = self._issues()
        numeric = [int(key) for key in issues if str(key).isdigit()]
        new_id = str(max(numeric, default=0) + 1)
        issues[new_id] = {
            "id": new_id,
            "title": title,
            "body": body,
            "reporter": "agent",
            "labels": list(args.get("labels", ())),
            "state": "open",
        }
        return MCPCallResult(
            payload={"created": new_id},
            written_ids=(f"mcp:github_stub/issues/{new_id}",),
            mutated=True,
        )


class SlackStub(FakeMCPServer):
    name = "slack_stub"

    def __init__(self, fixture: dict[str, Any] | None) -> None:
        super().__init__(fixture)
        self.state.setdefault("channels", {})

    def _channel(self, name: str) -> list[dict[str, Any]]:
        channels = self.state["channels"]
        if name not in channels:
            raise MCPCallError(f"channel not found: {name}")
        return channels[name]

    def tool_read_channel(self, args: dict[str, Any]) -> MCPCallResult:
        channel = str(_require(args, "channel"))
        messages = self._channel(channel)
        return MCPCallResult(
            payload=copy.deepcopy(messages),
            read_ids=(f"mcp:slack_stub/channels/{channel}",),
        )

    def tool_send_message(self, args: dict[str, Any]) -> MCPCallResult:
        channel = str(_require(args, "channel"))
        text = str(_require(args, "text"))
        self._channel(channel).append({"author": "agent", "text": text})
        return MCPCallResult(
            payload={"sent": True, "channel": channel},
            written_ids=(f"mcp:slack_stub/channels/{channel}",),
            mutated=True,
        )


class DbStub(FakeMCPServer):
    name = "db_stub"

    def __init__(self, fixture: dict[str, Any] | None) -> None:
        super().__init__(fixture)
        self.state.setdefault("tables", {})
        self.state.setdefault("secrets", {})

    def _table(self, name: str) -> dict[str, dict[str, Any]]:
        tables = self.state["tables"]
        if name not in tables:
            raise MCPCallError(f"table not found: {name}")
        return tables[name]

    def tool_get_row(self, args: dict[str, Any]) -> MCPCallResult:
        table = str(_require(args, "table"))
        key = str(_require(args, "key"))
        rows = self._table(table)
        if key not in rows:
            raise MCPCallError(f"row not found: {table}/{key}")
        return MCPCallResult(
            payload=copy.deepcopy(rows[key]),
            read_ids=(f"mcp:db_stub/{table}/{key}",),
        )

    def tool_query(self, args: dict[str, Any]) -> MCPCallResult:
        table = str(_require(args, "table"))
        rows = self._table(table)
        return MCPCallResult(
            payload=copy.deepcopy(rows),
            read_ids=tuple(f"mcp:db_stub/{table}/{key}" for key in sorted(rows)),
        )

    def tool_update_row(self, args: dict[str, Any]) -> MCPCallResult:
        table = str(_require(args, "table"))
        key = str(_require(args, "key"))
        column = str(_require(args, "column"))
        rows = self._table(table)
        if key not in rows:
            raise MCPCallError(f"row not found: {table}/{key}")
        rows[key][column] = args.get("value")
        return MCPCallResult(
            payload={"updated": f"{table}/{key}"},
            written_ids=(f"mcp:db_stub/{table}/{key}",),
            mutated=True,
        )

    def tool_read_secret(self, args: dict[str, Any]) -> MCPCallResult:
        return MCPCallResult(
            payload=copy.deepcopy(self.state["secrets"]),
            read_ids=("mcp:db_stub/secrets",),
            secret=True,
        )


SERVER_TYPES: dict[str, type[FakeMCPServer]] = {
    "github_stub": GithubStub,
    "slack_stub": SlackStub,
    "db_stub": DbStub,
}
