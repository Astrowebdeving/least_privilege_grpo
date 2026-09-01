from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from least_privilege_env.broker.classifier import classify_program
from least_privilege_env.broker.parser import ParseError, parse_command
from least_privilege_env.mcp.contracts import KNOWN_MCP_SERVERS, classify_mcp_call, decide_mcp_call
from least_privilege_env.mcp.servers import SERVER_TYPES
from least_privilege_env.tasks.loader import load_tasks
from least_privilege_env.tasks.schema import TaskSpec


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_task(task: TaskSpec) -> ValidationReport:
    report = ValidationReport()
    if task.hidden_files:
        for hidden_path in task.hidden_files:
            if hidden_path in task.workspace_files:
                report.errors.append(f"{task.task_id}: hidden file also appears in workspace: {hidden_path}")
    if task.label_confidence < 0 or task.label_confidence > 1:
        report.errors.append(f"{task.task_id}: label_confidence outside [0,1]")
    is_expansion = "expansion_v1" in task.command_pattern_tags
    if is_expansion and not task.design_sources:
        report.errors.append(f"{task.task_id}: expansion task missing design_sources")
    if is_expansion and not task.oracle.escalation_code and not task.verifier.require_evidence:
        report.errors.append(f"{task.task_id}: expansion task must use task-scoped evidence gating")
    if is_expansion and task.verifier.type in {
        "exact_answer",
        "path_answer",
        "answer_contains_all",
        "answer_keyword_groups",
        "answer_json",
    } and not task.verifier_rejects:
        report.errors.append(f"{task.task_id}: answer task missing verifier_rejects case")
    for path in (*task.verifier.required_reads, *task.verifier.required_writes):
        if path not in task.required_evidence:
            report.errors.append(
                f"{task.task_id}: verifier-specific evidence path {path!r} missing from required_evidence"
            )
    if task.verifier.type in {"answer_json", "mcp_state_exact"} and task.verifier.expected_json is None:
        report.errors.append(f"{task.task_id}: {task.verifier.type} missing expected_json")
    if is_expansion and (
        task.family in {"prompt_injection", "secret_lure"}
        or "prompt_injection" in task.command_pattern_tags
    ) and not task.forbidden_action_signatures:
        report.errors.append(f"{task.task_id}: safety task missing forbidden_action_signatures")
    for command in task.oracle.commands:
        try:
            program = parse_command(command)
        except ParseError as exc:
            report.errors.append(f"{task.task_id}: oracle parse failed for {command!r}: {exc}")
            continue
        classification = classify_program(program)
        if not classification.z_pre.leq(task.required_risk):
            report.errors.append(
                f"{task.task_id}: oracle risk {classification.z_pre.to_dict()} exceeds required {task.required_risk.to_dict()}"
            )

    # MCP surface: fixtures must be known servers wired to allowed_services,
    # and oracle MCP trajectories must be policy-allowed within the envelope.
    for server in task.mcp_services:
        if server not in SERVER_TYPES:
            report.errors.append(f"{task.task_id}: unknown mcp server fixture: {server}")
        if server not in task.allowed_services:
            report.errors.append(f"{task.task_id}: mcp fixture {server} not in allowed_services")
    for service in task.allowed_services:
        if service in KNOWN_MCP_SERVERS and service not in task.mcp_services:
            report.errors.append(f"{task.task_id}: allowed mcp server {service} has no fixture")
    for call in task.oracle.mcp_calls:
        classification = classify_mcp_call(call.server, call.tool, call.args)
        decision = decide_mcp_call(classification, call.server, list(task.allowed_services))
        if decision.mode != "execute":
            report.errors.append(
                f"{task.task_id}: oracle mcp call {call.server}.{call.tool} is not allowed: {decision.reason}"
            )
        if not classification.z_pre.leq(task.required_risk):
            report.errors.append(
                f"{task.task_id}: oracle mcp risk {classification.z_pre.to_dict()} exceeds required {task.required_risk.to_dict()}"
            )
    if task.verifier.type == "mcp_state":
        server = (task.verifier.path or "").partition(":")[0]
        if server not in task.mcp_services:
            report.errors.append(f"{task.task_id}: mcp_state verifier targets unfixtured server {server!r}")

    if not task.oracle.final_answer and not task.oracle.escalation_code and task.verifier.type not in {"contains", "hidden_python"}:
        report.warnings.append(f"{task.task_id}: oracle has no terminal action")
    return report


def validate_all() -> ValidationReport:
    aggregate = ValidationReport()
    tasks = load_tasks()
    ids = [task.task_id for task in tasks]
    duplicates = [task_id for task_id, count in Counter(ids).items() if count > 1]
    for task_id in duplicates:
        aggregate.errors.append(f"duplicate task id: {task_id}")
    families = Counter(task.family for task in tasks)
    if len(tasks) < 30:
        aggregate.errors.append("spike suite must contain at least 30 tasks")
    if "whole_family_heldout" not in {task.split for task in tasks}:
        aggregate.errors.append("missing whole_family_heldout split placeholder")
    for family in (
        "read_only_exact",
        "file_localization",
        "patch_proposal_only",
        "bug_fix_verify",
        "run_the_right_test",
        "log_data_transform",
        "scoped_write",
        "noisy_error_recovery",
        "prompt_injection",
        "secret_lure",
        "fake_external_read",
        "escalation_required",
        "mcp_stub",
    ):
        if families[family] == 0:
            aggregate.errors.append(f"missing family: {family}")
    for task in tasks:
        report = validate_task(task)
        aggregate.errors.extend(report.errors)
        aggregate.warnings.extend(report.warnings)
    return aggregate
