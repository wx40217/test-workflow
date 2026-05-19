"""
质量闭环工作流的结构化质量报告与路由判断。

本模块只做确定性、可测试的质量门和决策，不直接调用模型。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from src.workflow.validators import validate_fe_be_structure


DecisionRoute = Literal["finalize", "revise", "need_info", "max_rounds"]


@dataclass
class QualityReport:
    """候选测试用例的结构化质量报告。"""

    coverage_issues: list[str] = field(default_factory=list)
    format_issues: list[str] = field(default_factory=list)
    frontend_backend_issues: list[str] = field(default_factory=list)
    truncation_issues: list[str] = field(default_factory=list)
    redundancy_issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    score_breakdown: dict[str, Any] = field(default_factory=dict)
    blocking: bool = False
    score: float = 1.0
    passed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "coverage_issues": list(self.coverage_issues),
            "format_issues": list(self.format_issues),
            "frontend_backend_issues": list(self.frontend_backend_issues),
            "truncation_issues": list(self.truncation_issues),
            "redundancy_issues": list(self.redundancy_issues),
            "warnings": list(self.warnings),
            "score_breakdown": dict(self.score_breakdown),
            "blocking": self.blocking,
            "score": self.score,
            "passed": self.passed,
        }


@dataclass
class QualityDecision:
    """decide 节点的显式路由结果。"""

    route: DecisionRoute
    reason: str
    revision_plan: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "reason": self.reason,
            "revision_plan": list(self.revision_plan),
            "warnings": list(self.warnings),
        }


def build_quality_report(
    candidate: str,
    *,
    user_input: str,
    reviewer_feedback: str = "",
    output_format: str = "markdown",
    frontend_backend_mode: bool = False,
    node_warnings: list[str] | None = None,
    quality_threshold: float = 0.75,
) -> QualityReport:
    """基于确定性规则和评审反馈生成可解释质量报告。"""

    report = QualityReport()
    candidate = candidate or ""
    feedback = reviewer_feedback or ""
    node_warnings = node_warnings or []

    _check_basic_content(candidate, report)
    _check_output_format(candidate, output_format, report)
    _check_redundancy(candidate, report)
    _check_truncation(candidate, feedback, node_warnings, report)
    _check_coverage_from_feedback(feedback, report)
    _check_requirement_keywords(user_input, candidate, report)

    if frontend_backend_mode:
        validation = validate_fe_be_structure(candidate)
        if not validation.is_valid:
            report.frontend_backend_issues.extend(validation.issues)

    report.score_breakdown = _score_breakdown(report)
    report.score = report.score_breakdown["score"]
    report.blocking = bool(
        report.truncation_issues
        or report.frontend_backend_issues
        or _has_severe_format_issue(report.format_issues)
        or _has_severe_coverage_issue(report.coverage_issues)
    )
    report.passed = report.score >= quality_threshold and not report.blocking
    return report


def decide_next_step(
    report: QualityReport,
    *,
    review_round: int,
    max_review_rounds: int,
    validation_failed: bool = False,
) -> QualityDecision:
    """把质量报告映射为显式 LangGraph 分支。"""

    if report.passed and not validation_failed:
        return QualityDecision(route="finalize", reason="质量报告和确定性校验均通过。")

    plan = build_revision_plan(report)
    warnings = list(report.warnings)

    if validation_failed:
        plan.insert(0, "修复确定性结构校验失败项，优先保证输出格式可交付。")

    if review_round >= max_review_rounds:
        warnings.append("已达到最大评审轮次，返回当前最佳候选。")
        return QualityDecision(
            route="max_rounds",
            reason="达到最大评审轮次，停止继续修订。",
            revision_plan=plan,
            warnings=warnings,
        )

    if _needs_more_information(report):
        warnings.append("证据不足，基于已有需求做最小假设继续。")
        return QualityDecision(
            route="need_info",
            reason="存在需求证据不足的问题，但可带假设继续修订。",
            revision_plan=plan or ["补齐缺失场景，并在不污染最终输出的前提下按常见业务假设生成。"],
            warnings=warnings,
        )

    return QualityDecision(
        route="revise",
        reason="质量问题可由修订节点处理。",
        revision_plan=plan,
        warnings=warnings,
    )


def build_revision_plan(report: QualityReport) -> list[str]:
    """把报告中的扣分来源转换成给 revise 节点的修订计划。"""

    plan: list[str] = []
    for issue in report.coverage_issues:
        plan.append(f"补齐覆盖度问题：{issue}")
    for issue in report.format_issues:
        plan.append(f"修复格式问题：{issue}")
    for issue in report.frontend_backend_issues:
        plan.append(f"修复前后端分离结构问题：{issue}")
    for issue in report.truncation_issues:
        plan.append(f"修复截断问题：{issue}")
    for issue in report.redundancy_issues:
        plan.append(f"压缩重复或冗余内容：{issue}")
    return plan


def validate_candidate_structure(candidate: str, *, frontend_backend_mode: bool = False) -> dict[str, Any]:
    """确定性结构校验结果，供 validate 节点写入 metadata。"""

    if not candidate.strip():
        return {"passed": False, "issues": ["最终测试用例为空。"], "repair_hint": ""}

    if frontend_backend_mode:
        validation = validate_fe_be_structure(candidate)
        return {
            "passed": validation.is_valid,
            "issues": list(validation.issues),
            "repair_hint": validation.repair_hint,
        }

    issues: list[str] = []
    if re.search(r"^(以下是|下面是|当然|总结[:：])", candidate.strip(), re.IGNORECASE):
        issues.append("输出包含开场白或总结性元信息。")
    internal_issue = _detect_internal_metadata(candidate)
    if internal_issue:
        issues.append(internal_issue)
    return {"passed": not issues, "issues": issues, "repair_hint": ""}


def _check_basic_content(candidate: str, report: QualityReport) -> None:
    if not candidate.strip():
        report.format_issues.append("输出为空。")
        return
    if len(candidate.strip()) < 40:
        report.format_issues.append("输出过短，无法形成可交付测试用例。")


def _check_output_format(candidate: str, output_format: str, report: QualityReport) -> None:
    stripped = candidate.strip()
    if re.search(r"^(以下是|下面是|当然|总结[:：])", stripped, re.IGNORECASE):
        report.format_issues.append("输出包含开场白或总结性话术。")
    internal_issue = _detect_internal_metadata(stripped)
    if internal_issue:
        report.format_issues.append(internal_issue)
    if output_format == "confluence" and "<table" not in stripped.lower() and "|" not in stripped:
        report.format_issues.append("Confluence 输出缺少表格或可识别结构。")
    if output_format == "markdown" and not re.search(r"(^#{1,4}\s|^\s*[-*]\s|\*\*.+\*\*)", stripped, re.MULTILINE):
        report.format_issues.append("Markdown 输出缺少标题、列表或加粗测试点结构。")


def _check_redundancy(candidate: str, report: QualityReport) -> None:
    normalized_lines = [
        re.sub(r"\s+", " ", line.strip())
        for line in candidate.splitlines()
        if len(line.strip()) > 8
    ]
    if not normalized_lines:
        return
    duplicate_count = len(normalized_lines) - len(set(normalized_lines))
    if duplicate_count >= 2:
        report.redundancy_issues.append(f"存在 {duplicate_count} 行重复内容。")


def _detect_internal_metadata(text: str) -> str:
    lower_text = text.lower()
    internal_fields = [
        "quality_report",
        "agent_trace",
        "metadata",
        "revision_plan",
        "validation_reports",
    ]
    bracket_labels = [
        "质量报告",
        "trace",
        "metadata",
        "元信息",
        "内部推理",
        "修订计划",
    ]
    if any(field in lower_text for field in internal_fields):
        return "输出混入内部质量报告、trace 或 metadata 字段。"
    if any(re.search(rf"\[\s*{re.escape(label)}\s*\]", text, re.IGNORECASE) for label in bracket_labels):
        return "输出包含方括号内部标签或元信息。"
    return ""


def _check_truncation(
    candidate: str,
    feedback: str,
    node_warnings: list[str],
    report: QualityReport,
) -> None:
    all_text = "\n".join([candidate, feedback, "\n".join(node_warnings)])
    if "截断" in all_text or "max_tokens" in all_text or "finish_reason" in all_text:
        report.truncation_issues.append("检测到模型输出可能被截断。")


def _check_coverage_from_feedback(feedback: str, report: QualityReport) -> None:
    if not feedback.strip():
        report.warnings.append("评审反馈为空，质量判断只基于确定性规则。")
        return

    for line in feedback.splitlines():
        stripped = line.strip(" -*\t")
        if not stripped:
            continue
        lowered = stripped.lower()
        if any(marker in stripped for marker in ["遗漏", "缺少", "未覆盖", "不足"]) or any(
            marker in lowered for marker in ["missing", "not covered", "gap"]
        ):
            report.coverage_issues.append(stripped)


def _check_requirement_keywords(user_input: str, candidate: str, report: QualityReport) -> None:
    pairs = [
        ("邮箱", ["邮箱", "email"]),
        ("密码", ["密码", "password"]),
        ("3次失败锁定", ["3次", "三次", "锁定", "失败次数"]),
    ]
    for label, keywords in pairs:
        if label in user_input and not any(keyword in candidate for keyword in keywords):
            report.coverage_issues.append(f"需求提到「{label}」，候选输出未体现对应测试场景。")


def _score_breakdown(report: QualityReport) -> dict[str, Any]:
    deductions = {
        "coverage": {
            "points": min(0.4, 0.12 * len(report.coverage_issues)),
            "issues": list(report.coverage_issues),
        },
        "format": {
            "points": min(0.35, 0.12 * len(report.format_issues)),
            "issues": list(report.format_issues),
        },
        "frontend_backend": {
            "points": min(0.45, 0.18 * len(report.frontend_backend_issues)),
            "issues": list(report.frontend_backend_issues),
        },
        "truncation": {
            "points": min(0.4, 0.25 * len(report.truncation_issues)),
            "issues": list(report.truncation_issues),
        },
        "redundancy": {
            "points": min(0.2, 0.08 * len(report.redundancy_issues)),
            "issues": list(report.redundancy_issues),
        },
    }
    total_deduction = round(sum(item["points"] for item in deductions.values()), 2)
    return {
        "base_score": 1.0,
        "deductions": deductions,
        "total_deduction": total_deduction,
        "score": round(max(0.0, 1.0 - total_deduction), 2),
    }


def _has_severe_format_issue(issues: list[str]) -> bool:
    return any("为空" in issue or "过短" in issue or "内部质量报告" in issue for issue in issues)


def _has_severe_coverage_issue(issues: list[str]) -> bool:
    return any("遗漏" in issue or "缺少" in issue or "未覆盖" in issue for issue in issues)


def _needs_more_information(report: QualityReport) -> bool:
    text = "\n".join(report.coverage_issues + report.warnings)
    return any(marker in text for marker in ["不明确", "无法判断", "证据不足", "需补充", "需要补充"])
