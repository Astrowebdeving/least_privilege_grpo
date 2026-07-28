"""Per-episode MCP registry: instantiates fake servers from task fixtures."""

from __future__ import annotations

from typing import Any

from least_privilege_env.mcp.servers import (
    FakeMCPServer,
    MCPCallError,
    MCPCallResult,
    SERVER_TYPES,
)


class MCPRegistry:
    """One instance per episode; servers deep-copy their fixtures, so episode
    state is isolated by construction and dropped with the env instance."""

    def __init__(self, services: dict[str, dict[str, Any]] | None) -> None:
        self.servers: dict[str, FakeMCPServer] = {}
        for name, fixture in (services or {}).items():
            server_type = SERVER_TYPES.get(name)
            if server_type is None:
                raise ValueError(f"unknown mcp server in task fixture: {name}")
            self.servers[name] = server_type(fixture)

    def call(self, server: str, tool: str, args: dict[str, Any]) -> MCPCallResult:
        instance = self.servers.get(server)
        if instance is None:
            # Policy should have blocked unauthorized servers before this point.
            raise MCPCallError(f"service unavailable: {server}")
        return instance.call(tool, args)

    def hidden_state(self) -> dict[str, Any]:
        return {name: server.hidden_state() for name, server in self.servers.items()}
