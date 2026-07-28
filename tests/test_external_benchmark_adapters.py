from __future__ import annotations

import json

from scripts.evaluate_metatool_subset import _candidate_names, _score
from scripts.summarize_toolprivbench_suite import _load_rows


SINGLE_PROMPT = """
List of Tools with Names and Descriptions:
1. tool name: NarrowTool, tool description: targeted lookup
2. tool name: BroadTool, tool description: broad lookup
"""


def test_metatool_awareness_parser_uses_yes_no_answer() -> None:
    result = _score(
        {"label": "positive"},
        "unused",
        "Yes. Current information requires an external lookup.",
        "awareness",
    )
    assert result["predicted"] == "yes"
    assert result["correct"] is True


def test_metatool_single_parser_selects_first_named_candidate() -> None:
    result = _score(
        {"tool": "NarrowTool"},
        SINGLE_PROMPT,
        "NarrowTool — it is the targeted option. BroadTool would be unnecessary.",
        "single",
    )
    assert result["predicted"] == "NarrowTool"
    assert result["correct"] is True


def test_metatool_single_parser_handles_none_before_tool_mentions() -> None:
    result = _score(
        {"tool": "None"},
        SINGLE_PROMPT,
        "None. Neither NarrowTool nor BroadTool is required.",
        "single",
    )
    assert result["predicted"] == "None"
    assert result["correct"] is True


def test_metatool_reliability_target_is_none_when_source_tool_is_excluded() -> None:
    result = _score(
        {"tool": "RemovedGroundTruthTool"},
        SINGLE_PROMPT,
        "None. The requested capability is not present in the offered list.",
        "reliability",
    )
    assert result["expected"] == "None"
    assert result["predicted"] == "None"
    assert result["correct"] is True


def test_metatool_multi_parser_is_order_insensitive_for_two_tools() -> None:
    result = _score(
        {"tool": ["NarrowTool", "BroadTool"]},
        SINGLE_PROMPT,
        "BroadTool and NarrowTool cover the two requested operations.",
        "multi",
    )
    assert result["predicted"] == ["BroadTool", "NarrowTool"]
    assert result["correct"] is True


def test_candidate_parser_preserves_tool_spelling() -> None:
    assert _candidate_names(SINGLE_PROMPT) == ["NarrowTool", "BroadTool"]


def test_toolpriv_summarizer_loads_all_official_id_prefixes(tmp_path) -> None:
    for scenario_id in ("V1-SAFE-001", "BIZ-SCOPE-001", "CODE-TEMP-001"):
        (tmp_path / f"{scenario_id}.json").write_text(
            json.dumps({"scenario_id": scenario_id}),
            encoding="utf-8",
        )
    (tmp_path / "summary.json").write_text(
        json.dumps({"total": 3}),
        encoding="utf-8",
    )

    assert set(_load_rows(tmp_path)) == {
        "V1-SAFE-001",
        "BIZ-SCOPE-001",
        "CODE-TEMP-001",
    }
