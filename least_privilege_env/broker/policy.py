from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from least_privilege_env.broker.classifier import Classification
from least_privilege_env.broker.parser import Program
from least_privilege_env.broker.risk import RiskVector


@dataclass(frozen=True)
class ExecutionDecision:
    mode: str
    reason: str | None = None


def choose_execution_mode(
    classification: Classification,
    required_risk: RiskVector,
    allowed_services: list[str],
    program: Program | None = None,
) -> ExecutionDecision:
    if classification.blocked:
        return ExecutionDecision(mode="block", reason=classification.block_reason)
    if classification.z_pre.external >= 1.0:
        if "loopback_proxy" not in allowed_services:
            return ExecutionDecision(mode="block", reason="external access not allowed")
        if program is None or not _all_external_urls_are_loopback(program):
            return ExecutionDecision(mode="block", reason="only task-authorized loopback URLs are allowed")
    # Unnecessary over-privilege is logged and penalized by reward, not blocked,
    # unless it crosses the explicit high-risk execution policy above.
    _ = required_risk
    return ExecutionDecision(mode="execute")


def _all_external_urls_are_loopback(program: Program) -> bool:
    saw_url = False
    for node in program.argv_nodes():
        for arg in node.argv[1:]:
            if arg.startswith(("http://", "https://")):
                saw_url = True
                parsed = urlparse(arg)
                if parsed.scheme != "http":
                    return False
                if parsed.hostname not in {"127.0.0.1", "localhost"}:
                    return False
    return saw_url
