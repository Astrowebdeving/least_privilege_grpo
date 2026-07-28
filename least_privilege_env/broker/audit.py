from __future__ import annotations

from dataclasses import dataclass, field

from least_privilege_env.broker.risk import RiskVector, max_all


@dataclass
class ActionAudit:
    raw: str
    status: str
    z_pre: RiskVector
    z_post: RiskVector
    z_final: RiskVector
    action_type: str = "terminal"
    reasons: list[str] = field(default_factory=list)
    read_paths: set[str] = field(default_factory=set)
    written_paths: set[str] = field(default_factory=set)
    blocked_severity: float = 0.0
    unknown_score: float = 0.0
    exit_code: int | None = None
    # Host-side excerpt of the model-facing observation (already F0-shaped);
    # kept for rollout review, never re-exposed to the model.
    observation: str = ""


@dataclass
class EscalationAudit:
    code: str
    detail: str


@dataclass
class AuditLog:
    actions: list[ActionAudit] = field(default_factory=list)
    escalation: EscalationAudit | None = None
    final_answer: str | None = None
    frozen: bool = False

    def record(self, item: ActionAudit) -> None:
        if self.frozen:
            raise RuntimeError("episode over")
        self.actions.append(item)

    def record_final(self, text: str) -> None:
        if self.frozen:
            raise RuntimeError("episode over")
        self.final_answer = text
        self.frozen = True

    def record_escalation(self, code: str, detail: str) -> None:
        if self.frozen:
            raise RuntimeError("episode over")
        self.escalation = EscalationAudit(code=code, detail=detail)
        self.frozen = True

    def z_tau(self) -> RiskVector:
        return max_all(item.z_final for item in self.actions)

    def accessed_evidence(self) -> set[str]:
        evidence: set[str] = set()
        for item in self.actions:
            evidence.update(item.read_paths)
            evidence.update(item.written_paths)
        return evidence

    def blocked_severity(self) -> float:
        return min(1.0, sum(item.blocked_severity for item in self.actions))

    def unknown_score(self) -> float:
        return min(1.0, sum(item.unknown_score for item in self.actions))
