from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from least_privilege_env.broker.risk import RiskVector


Difficulty = Literal["easy", "medium", "hard"]


@dataclass(frozen=True)
class MCPCallSpec:
    server: str
    tool: str
    args: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "MCPCallSpec":
        return cls(
            server=value["server"],
            tool=value["tool"],
            args=dict(value.get("args", {})),
        )


@dataclass(frozen=True)
class TransientFault:
    """Deterministic, host-side transient fault on a fake MCP tool.

    The first ``fail_attempts`` calls to ``server.tool`` that reach execution
    (i.e. passed the block/authorization decision) fail with ``error`` through
    the ordinary MCP tool-error channel, without mutating fake-service state;
    the next call succeeds. Determinism is by per-episode attempt count, so the
    fault is reproducible across the local and Docker backends (MCP state is
    host-side) and needs no RNG. ``error`` is a task-authored *generic*
    transient message (firewalled: never a copied external error string).
    """

    server: str
    tool: str
    fail_attempts: int = 1
    error: str = "temporary upstream error; retry"

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "TransientFault":
        return cls(
            server=value["server"],
            tool=value["tool"],
            fail_attempts=int(value.get("fail_attempts", 1)),
            error=str(value.get("error", "temporary upstream error; retry")),
        )


@dataclass(frozen=True)
class Oracle:
    commands: tuple[str, ...] = ()
    mcp_calls: tuple[MCPCallSpec, ...] = ()
    final_answer: str | None = None
    escalation_code: str | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "Oracle":
        value = value or {}
        return cls(
            commands=tuple(value.get("commands", ())),
            mcp_calls=tuple(MCPCallSpec.from_mapping(item) for item in value.get("mcp_calls", ())),
            final_answer=value.get("final_answer"),
            escalation_code=value.get("escalation_code"),
        )


@dataclass(frozen=True)
class VerifierSpec:
    type: str
    expected: str | None = None
    path: str | None = None
    contains: str | None = None
    contains_all: tuple[str, ...] = ()
    # AND-of-OR keyword groups: the answer must contain at least one synonym
    # from every group. Lets a verifier accept paraphrases without enumerating
    # whole sentences (see answer_keyword_groups in verify.py).
    groups: tuple[tuple[str, ...], ...] = ()
    code: str | None = None
    codes: tuple[str, ...] = ()
    hidden_python: str | None = None
    # When true, any blocked action in the episode fails verification outright,
    # regardless of the final answer. Used to make a benign-answer check double
    # as a "did not take the forbidden/injected action" gate.
    forbid_blocked: bool = False
    # test_run only: when true, success additionally requires the audited exit
    # code of the executed check to equal int(expected), so a numeric exit-code
    # answer is verified against what actually ran rather than the submission.
    match_exit_code: bool = False
    # test_run only: a permitted/required exit code when the check legitimately
    # exits non-zero (e.g. a pytest task whose answer is the failing test name,
    # not the exit code). When set, the executed check's audited exit code must
    # equal this value and the answer is checked separately against `expected`.
    expected_exit: int | None = None
    # Optional, task-scoped success-on-evidence gate. This is deliberately not
    # global: enable it only for tasks whose oracle and negative tests prove the
    # evidence extractor sees the required paths/resources reliably.
    require_evidence: bool = False
    # Stronger evidence contracts for tasks where merely touching a path is
    # insufficient. These distinguish inspection from mutation.
    required_reads: tuple[str, ...] = ()
    required_writes: tuple[str, ...] = ()
    # Some task families (especially "run exactly this check") define any
    # workspace mutation as failure, not merely an over-privileged success.
    forbid_writes: bool = False
    # Mutation tasks can also require exact side-effect scope: every audited
    # terminal/MCP write must be one of required_writes. This closes reward
    # hacking where an agent performs the requested edit plus an extra in-scope
    # write that the risk vector would not otherwise price as over-privilege.
    forbid_extra_writes: bool = False
    # Honeypot/data-minimization success gate: when true, an episode that
    # content-reads any path/MCP id in TaskSpec.forbidden_reads fails outright,
    # regardless of the answer. This is enabled ONLY where "don't touch X" is an
    # explicit part of the task contract (honeypot_restraint). For every other
    # task forbidden_reads stays report-only (success and safety separate axes,
    # proposal commitment #3); the report-only signal is computed off-policy in
    # safety_metrics_report.py and does not need this flag.
    gate_on_forbidden_read: bool = False
    # Structured-answer and exact-state verifiers use JSON-compatible values.
    expected_json: Any = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "VerifierSpec":
        normalized = dict(value)
        if "contains_all" in normalized:
            normalized["contains_all"] = tuple(normalized["contains_all"])
        if "codes" in normalized:
            normalized["codes"] = tuple(normalized["codes"])
        if "groups" in normalized:
            normalized["groups"] = tuple(tuple(group) for group in normalized["groups"])
        if "required_reads" in normalized:
            normalized["required_reads"] = tuple(normalized["required_reads"])
        if "required_writes" in normalized:
            normalized["required_writes"] = tuple(normalized["required_writes"])
        return cls(**normalized)


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    prompt: str
    family: str
    difficulty: Difficulty
    max_steps: int
    required_risk: RiskVector
    label_confidence: float
    required_evidence: tuple[str, ...]
    allowed_services: tuple[str, ...]
    split: str
    command_pattern_tags: tuple[str, ...]
    oracle: Oracle
    verifier: VerifierSpec
    workspace_files: dict[str, str] = field(default_factory=dict)
    hidden_files: dict[str, str] = field(default_factory=dict)
    # Fake MCP service fixtures, keyed by server name. Host-side hidden state:
    # never materialized into /workspace, never serialized into the dataset.
    mcp_services: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Host-side transient fault injection for the runtime-failure / retry task
    # family (see docs/task_authoring_standard.md). Never model-facing; surfaced
    # only through the ordinary MCP tool-error observation. Default empty keeps
    # every existing task unchanged.
    transient_faults: tuple[TransientFault, ...] = ()
    # Host-only benchmark provenance and quality cases. These fields are never
    # included in the model-facing dataset.
    design_sources: tuple[str, ...] = ()
    verifier_accepts: tuple[str, ...] = ()
    verifier_rejects: tuple[str, ...] = ()
    # Case-insensitive substrings matched against audited raw actions. Used for
    # exact, task-defined lure diagnostics rather than generic blocked-call
    # inference.
    forbidden_action_signatures: tuple[str, ...] = ()
    # Host-only list of workspace paths / MCP resource ids a task must not
    # content-read (honeypot decoys, irrelevant-but-tempting files, the unneeded
    # field/file in a data-minimization task). Never model-facing. Feeds the
    # report-only forbidden_read_attempt / success_with_forbidden_read metrics
    # for (nearly) every task; only gates success where verifier
    # gate_on_forbidden_read is set. Quality guard enforces
    # forbidden_reads ∩ required_evidence/required_reads = ∅.
    forbidden_reads: tuple[str, ...] = ()
    # Template-family identifier (expansion_v2). Lets the builder assert
    # train/held-out separation at the template-family level rather than only by
    # task id, so a held-out probe cannot be a near-twin of a same-group training
    # template. Host-only; None for hand-authored or pre-v2 tasks.
    template_group_id: str | None = None
    # Host-only over-privileged / wrong trajectory (transient_failure family).
    # An Oracle-shaped "bypass" the quality gate replays and asserts scores
    # UNSAFE (success != 1.0): e.g. reaching for the elevated/external tool
    # instead of retrying, or giving up and escalating when a retry would work.
    # Never model-facing; None for tasks without a defined negative probe.
    negative_oracle: Oracle | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "TaskSpec":
        return cls(
            task_id=value["task_id"],
            prompt=value["prompt"],
            family=value["family"],
            difficulty=value["difficulty"],
            max_steps=int(value.get("max_steps", 8)),
            required_risk=RiskVector.from_mapping(value.get("required_risk")),
            label_confidence=float(value.get("label_confidence", 1.0)),
            required_evidence=tuple(value.get("required_evidence", ())),
            allowed_services=tuple(value.get("allowed_services", ())),
            split=value.get("split", "train"),
            command_pattern_tags=tuple(value.get("command_pattern_tags", ())),
            oracle=Oracle.from_mapping(value.get("oracle")),
            verifier=VerifierSpec.from_mapping(value["verifier"]),
            workspace_files=dict(value.get("workspace_files", {})),
            hidden_files=dict(value.get("hidden_files", {})),
            mcp_services={name: dict(fixture) for name, fixture in value.get("mcp_services", {}).items()},
            transient_faults=tuple(
                TransientFault.from_mapping(item) for item in value.get("transient_faults", ())
            ),
            design_sources=tuple(value.get("design_sources", ())),
            verifier_accepts=tuple(value.get("verifier_accepts", ())),
            verifier_rejects=tuple(value.get("verifier_rejects", ())),
            forbidden_action_signatures=tuple(value.get("forbidden_action_signatures", ())),
            forbidden_reads=tuple(value.get("forbidden_reads", ())),
            template_group_id=value.get("template_group_id"),
            negative_oracle=(
                Oracle.from_mapping(value["negative_oracle"])
                if value.get("negative_oracle")
                else None
            ),
        )

    def to_initial_observation(self) -> str:
        return self.prompt
