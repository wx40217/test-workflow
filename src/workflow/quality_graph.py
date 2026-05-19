"""
现代质量闭环 LangGraph 工作流。

该图保留现有生成/评审/优化节点，但把质量报告、显式路由、循环修订、
确定性校验和最终收敛作为独立状态节点表达。
"""

from __future__ import annotations

from contextlib import contextmanager
from uuid import uuid4
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from config.settings import settings
from src.workflow.nodes import AnalyzerNode
from src.workflow.quality import (
    QualityDecision,
    QualityReport,
    build_quality_report,
    decide_next_step,
    validate_candidate_structure,
)


class AgentWorkflowState(TypedDict, total=False):
    user_input: str
    additional_instructions: str
    images: list[dict]
    output_format: str
    requirements_scope: str
    rag_context: str
    draft_test_cases: str
    review_feedback: str
    review_round: int
    quality_report: dict[str, Any]
    coverage_gaps: list[str]
    revision_plan: list[str]
    best_candidate: str
    best_score: float
    agent_trace: list[dict[str, Any]]
    final_test_cases: str
    errors: list[str]
    warnings: list[str]
    next_required_information: list[str]
    validation_reports: list[dict[str, Any]]
    validation_passed: bool
    decision: dict[str, Any]
    next_route: str
    current_step: str


class QualityGraphWorkflow:
    """quality-graph 模式执行器。"""

    def __init__(
        self,
        workflow: Any,
        *,
        max_review_rounds: int | None = None,
        quality_threshold: float | None = None,
    ):
        self.workflow = workflow
        self.max_review_rounds = (
            max_review_rounds if max_review_rounds is not None else settings.max_review_rounds
        )
        self.quality_threshold = (
            quality_threshold if quality_threshold is not None else settings.quality_threshold
        )
        self.checkpointer = MemorySaver()
        self._graph = self._build_graph()

    def run(
        self,
        *,
        user_input: str,
        additional_instructions: str = "",
        images: list[dict] | None = None,
        output_format: str = "markdown",
    ) -> dict[str, Any]:
        initial_state: AgentWorkflowState = {
            "user_input": user_input,
            "additional_instructions": additional_instructions,
            "images": images or [],
            "output_format": output_format,
            "requirements_scope": "",
            "rag_context": "",
            "draft_test_cases": "",
            "review_feedback": "",
            "review_round": 0,
            "quality_report": {},
            "coverage_gaps": [],
            "revision_plan": [],
            "best_candidate": "",
            "best_score": 0.0,
            "agent_trace": [],
            "final_test_cases": "",
            "errors": [],
            "warnings": [],
            "next_required_information": [],
            "validation_reports": [],
            "validation_passed": True,
            "decision": {},
            "next_route": "revise",
            "current_step": "start",
        }
        config = {"configurable": {"thread_id": f"quality-graph-{uuid4()}"}}
        return self._graph.invoke(initial_state, config=config)

    def _build_graph(self):
        graph = StateGraph(AgentWorkflowState)
        graph.add_node("plan_scope", self._plan_scope)
        graph.add_node("generate", self._generate)
        graph.add_node("evaluate", self._evaluate)
        graph.add_node("decide", self._decide)
        graph.add_node("revise", self._revise)
        graph.add_node("validate", self._validate)
        graph.add_node("finalize", self._finalize)

        graph.set_entry_point("plan_scope")
        graph.add_edge("plan_scope", "generate")
        graph.add_edge("generate", "evaluate")
        graph.add_edge("evaluate", "decide")
        graph.add_conditional_edges(
            "decide",
            self._route_after_decide,
            {
                "finalize": "validate",
                "revise": "revise",
                "need_info": "revise",
                "max_rounds": "validate",
            },
        )
        graph.add_edge("revise", "evaluate")
        graph.add_conditional_edges(
            "validate",
            self._route_after_validate,
            {
                "finalize": "finalize",
                "revise": "revise",
                "max_rounds": "finalize",
            },
        )
        graph.add_edge("finalize", END)
        return graph.compile(checkpointer=self.checkpointer)

    def _plan_scope(self, state: AgentWorkflowState) -> dict[str, Any]:
        trace = self._trace(state, "plan_scope", "提炼需求范围")
        scope = state["user_input"]
        rag_context = ""

        rag_interface = getattr(self.workflow, "rag_interface", None)
        if rag_interface is not None and rag_interface.is_enabled():
            try:
                documents = rag_interface.retrieve(state["user_input"])
                rag_context = "\n".join(f"{idx}. {doc}" for idx, doc in enumerate(documents, 1))
                trace = trace + [{
                    "node": "retrieve_rag_context",
                    "detail": f"按需检索 {len(documents)} 条测试知识",
                    "review_round": int(state.get("review_round", 0)),
                }]
            except Exception as exc:
                return {
                    "requirements_scope": scope,
                    "rag_context": "",
                    "errors": state.get("errors", []) + [f"RAG检索错误: {exc}"],
                    "agent_trace": trace,
                    "current_step": "plan_scope_rag_error",
                }

        if self.workflow.enable_analyzer and AnalyzerNode.should_analyze(
            state["user_input"],
            self.workflow.analyzer_complexity_threshold,
        ):
            try:
                with self._without_node_rag(self.workflow.analyzer):
                    output = self.workflow.analyzer.invoke(
                        user_input=state["user_input"],
                        additional_instructions=self._instructions_with_rag(state, rag_context),
                    )
                scope = output.content or scope
                if output.is_truncated:
                    return {
                        "requirements_scope": scope,
                        "rag_context": rag_context,
                        "errors": state.get("errors", []) + [output.truncation_warning],
                        "agent_trace": trace,
                        "current_step": "plan_scope_complete",
                    }
            except Exception as exc:
                return {
                    "requirements_scope": scope,
                    "rag_context": rag_context,
                    "errors": state.get("errors", []) + [f"范围规划错误: {exc}"],
                    "agent_trace": trace,
                    "current_step": "plan_scope_error",
                }

        return {
            "requirements_scope": scope,
            "rag_context": rag_context,
            "agent_trace": trace,
            "current_step": "plan_scope_complete",
        }

    def _generate(self, state: AgentWorkflowState) -> dict[str, Any]:
        trace = self._trace(state, "generate", "生成初始候选")
        try:
            with self._without_node_rag(self.workflow.generator):
                output = self.workflow.generator.invoke(
                    user_input=state["user_input"],
                    additional_instructions=self._instructions_with_rag(state),
                    images=state.get("images", []),
                    analysis_result=state.get("requirements_scope", ""),
                )
            errors = state.get("errors", [])
            if output.is_truncated:
                errors = errors + [output.truncation_warning]
            return {
                "draft_test_cases": output.content,
                "best_candidate": output.content,
                "errors": errors,
                "agent_trace": trace,
                "current_step": "generate_complete",
            }
        except Exception as exc:
            return {
                "draft_test_cases": "",
                "errors": state.get("errors", []) + [f"生成错误: {exc}"],
                "warnings": state.get("warnings", []) + ["生成能力不可用，无法产出候选测试用例。"],
                "next_required_information": state.get("next_required_information", []) + [
                    "需要可用的生成模型能力或可替代的候选测试用例输入。"
                ],
                "agent_trace": trace,
                "current_step": "generate_error",
            }

    def _evaluate(self, state: AgentWorkflowState) -> dict[str, Any]:
        trace = self._trace(state, "evaluate", f"第 {state.get('review_round', 0)} 轮质量评估")
        candidate = state.get("draft_test_cases", "")
        feedback = state.get("review_feedback", "")
        errors = state.get("errors", [])

        if candidate:
            try:
                with self._without_node_rag(self.workflow.reviewer):
                    output = self.workflow.reviewer.invoke(
                        original_input=state["user_input"],
                        test_cases=candidate,
                    )
                feedback = output.content
                if output.is_truncated:
                    errors = errors + [output.truncation_warning]
            except Exception as exc:
                feedback = feedback or ""
                errors = errors + [f"评审错误: {exc}"]

        report = build_quality_report(
            candidate,
            user_input=state["user_input"],
            reviewer_feedback=feedback,
            output_format=state.get("output_format", "markdown"),
            frontend_backend_mode=self.workflow._is_frontend_backend_mode(),
            node_warnings=errors,
            quality_threshold=self.quality_threshold,
        )

        best_candidate = state.get("best_candidate", "")
        best_score = float(state.get("best_score", 0.0) or 0.0)
        if candidate and report.score >= best_score:
            best_candidate = candidate
            best_score = report.score

        return {
            "review_feedback": feedback,
            "quality_report": report.to_dict(),
            "coverage_gaps": list(report.coverage_issues),
            "best_candidate": best_candidate,
            "best_score": best_score,
            "errors": errors,
            "agent_trace": trace,
            "current_step": "evaluate_complete",
        }

    def _decide(self, state: AgentWorkflowState) -> dict[str, Any]:
        trace = self._trace(state, "decide", "根据质量报告选择下一步")
        report = self._report_from_state(state)
        decision = decide_next_step(
            report,
            review_round=int(state.get("review_round", 0)),
            max_review_rounds=self.max_review_rounds,
            validation_failed=bool(state.get("validation_reports")) and not bool(state.get("validation_passed", True)),
        )
        return {
            "decision": decision.to_dict(),
            "revision_plan": decision.revision_plan,
            "warnings": state.get("warnings", []) + decision.warnings,
            "next_required_information": self._next_required_information(decision) or state.get("next_required_information", []),
            "next_route": decision.route,
            "agent_trace": trace,
            "current_step": f"decide_{decision.route}",
        }

    def _revise(self, state: AgentWorkflowState) -> dict[str, Any]:
        trace = self._trace(state, "revise", "按质量报告和修订计划修复候选")
        plan = state.get("revision_plan", []) or ["按质量报告修复候选测试用例。"]
        candidate = state.get("draft_test_cases") or state.get("best_candidate", "")
        feedback = self._revision_feedback(state, plan, candidate)
        errors = state.get("errors", [])

        try:
            with self._without_node_rag(self.workflow.optimizer):
                output = self.workflow.optimizer.invoke(
                    original_input=state["user_input"],
                    initial_test_cases=candidate,
                    review_feedback=feedback,
                    output_format=state.get("output_format", "markdown"),
                )
            errors = errors + ([output.truncation_warning] if output.is_truncated else [])
            return {
                "draft_test_cases": output.content,
                "review_round": int(state.get("review_round", 0)) + 1,
                "errors": errors,
                "agent_trace": trace,
                "current_step": "revise_complete",
            }
        except Exception as exc:
            return {
                "draft_test_cases": candidate,
                "review_round": int(state.get("review_round", 0)) + 1,
                "errors": errors + [f"修订错误: {exc}"],
                "warnings": state.get("warnings", []) + ["修订失败，保留当前最佳候选。"],
                "next_required_information": state.get("next_required_information", []) + [
                    "需要可用的修订/优化模型能力，或人工提供可合并的修订内容。"
                ],
                "agent_trace": trace,
                "current_step": "revise_error",
            }

    def _validate(self, state: AgentWorkflowState) -> dict[str, Any]:
        trace = self._trace(state, "validate", "执行确定性结构校验")
        if state.get("next_route") == "max_rounds" and state.get("best_candidate"):
            candidate = state.get("best_candidate", "")
            candidate_source = "best_candidate"
        else:
            candidate = state.get("draft_test_cases") or state.get("best_candidate", "")
            candidate_source = "draft_test_cases" if state.get("draft_test_cases") else "best_candidate"
        validation = validate_candidate_structure(
            candidate,
            frontend_backend_mode=self.workflow._is_frontend_backend_mode(),
        )
        validation = {
            **validation,
            "candidate_source": candidate_source,
            "candidate_preview": candidate[:120],
        }
        reports = state.get("validation_reports", []) + [validation]

        if validation["passed"]:
            return {
                "validation_passed": True,
                "validation_reports": reports,
                "next_route": "finalize",
                "agent_trace": trace,
                "current_step": "validate_passed",
            }

        if int(state.get("review_round", 0)) >= self.max_review_rounds:
            return {
                "validation_passed": False,
                "validation_reports": reports,
                "next_route": "max_rounds",
                "warnings": state.get("warnings", []) + ["确定性校验失败且已达到最大评审轮次，返回最佳候选。"],
                "agent_trace": trace,
                "current_step": "validate_failed_max_rounds",
            }

        return {
            "validation_passed": False,
            "validation_reports": reports,
            "revision_plan": state.get("revision_plan", []) + [
                "修复确定性校验失败：" + "；".join(validation.get("issues", []))
            ],
            "next_route": "revise",
            "agent_trace": trace,
            "current_step": "validate_failed_revise",
        }

    def _finalize(self, state: AgentWorkflowState) -> dict[str, Any]:
        trace = self._trace(state, "finalize", "收敛并返回最佳候选")
        report = self._report_from_state(state)
        candidate = state.get("draft_test_cases") or ""
        best = state.get("best_candidate") or candidate
        final = candidate if report.passed or not best else best
        warnings = state.get("warnings", [])
        if state.get("next_route") == "max_rounds" and best:
            final = best
        if not final:
            warnings = warnings + ["没有可交付候选输出。"]
        return {
            "final_test_cases": final,
            "warnings": warnings,
            "agent_trace": trace,
            "current_step": "finalize_complete",
        }

    def _route_after_decide(self, state: AgentWorkflowState) -> str:
        return state.get("next_route", "revise")

    def _route_after_validate(self, state: AgentWorkflowState) -> str:
        return state.get("next_route", "finalize")

    def _trace(self, state: AgentWorkflowState, node: str, detail: str) -> list[dict[str, Any]]:
        return state.get("agent_trace", []) + [
            {
                "node": node,
                "detail": detail,
                "review_round": int(state.get("review_round", 0)),
            }
        ]

    def _report_from_state(self, state: AgentWorkflowState) -> QualityReport:
        data = state.get("quality_report") or {}
        return QualityReport(
            coverage_issues=list(data.get("coverage_issues", [])),
            format_issues=list(data.get("format_issues", [])),
            frontend_backend_issues=list(data.get("frontend_backend_issues", [])),
            truncation_issues=list(data.get("truncation_issues", [])),
            redundancy_issues=list(data.get("redundancy_issues", [])),
            warnings=list(data.get("warnings", [])),
            score_breakdown=dict(data.get("score_breakdown", {})),
            blocking=bool(data.get("blocking", False)),
            score=float(data.get("score", 0.0)),
            passed=bool(data.get("passed", False)),
        )

    def _revision_feedback(
        self,
        state: AgentWorkflowState,
        plan: list[str],
        candidate: str,
    ) -> str:
        report = state.get("quality_report", {})
        plan_text = "\n".join(f"- {item}" for item in plan)
        return (
            f"{state.get('review_feedback', '')}\n\n"
            "## 质量闭环修订计划\n"
            f"{plan_text}\n\n"
            "## 继续修订所依据的假设或待补信息\n"
            f"{state.get('next_required_information', [])}\n\n"
            "## 结构化质量报告\n"
            f"{report}\n\n"
            "## 待修订候选\n"
            f"{candidate}\n\n"
            "只输出修订后的最终测试用例，不要输出质量报告、解释、trace 或内部元信息。"
        )

    def _next_required_information(self, decision: QualityDecision) -> list[str]:
        if decision.route != "need_info":
            return []
        return [
            "需要补充不明确的业务规则、边界条件或验收标准；本轮将基于常见业务假设继续修订。",
        ]

    def _instructions_with_rag(
        self,
        state: AgentWorkflowState,
        rag_context: str | None = None,
    ) -> str:
        instructions = state.get("additional_instructions", "")
        context = rag_context if rag_context is not None else state.get("rag_context", "")
        if not context:
            return instructions
        return f"{instructions}\n\n## RAG 检索上下文（quality-graph 按需检索）\n{context}"

    @contextmanager
    def _without_node_rag(self, node: Any):
        original = getattr(node, "rag_interface", None)
        node.rag_interface = None
        try:
            yield
        finally:
            node.rag_interface = original
