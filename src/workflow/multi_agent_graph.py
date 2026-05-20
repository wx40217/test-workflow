"""多 Agent Quality Graph 工作流。

该模式在现有节点能力上引入显式角色、共享候选池和 Orchestrator 质量门。
LLM 角色只能产出分析、候选、评审和修订，确定性 Validator 的结果始终进入
Orchestrator 路由，Finalizer 不会绕过校验重写内容。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, TypedDict
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from config.settings import settings
from src.workflow.candidate_pool import CandidatePool
from src.workflow.nodes import AnalyzerNode
from src.workflow.quality import (
    QualityReport,
    build_quality_report,
    build_revision_plan,
    validate_candidate_structure,
)


class MultiAgentQualityState(TypedDict, total=False):
    user_input: str
    additional_instructions: str
    images: list[dict]
    output_format: str
    requirement_analysis: dict[str, Any]
    retrieval_context: dict[str, Any]
    candidate_pool: list[dict[str, Any]]
    current_candidate_id: str
    review_reports: list[dict[str, Any]]
    revision_plan: list[str]
    validation_reports: list[dict[str, Any]]
    best_candidate: str
    best_candidate_score: float
    final_test_cases: str
    agent_rounds: int
    no_improvement_rounds: int
    errors: list[str]
    warnings: list[str]
    agent_trace: list[dict[str, Any]]
    active_agents: list[str]
    quality_report: dict[str, Any]
    quality_passed: bool
    validation_passed: bool
    orchestrator_decision: dict[str, Any]
    orchestrator_decisions: list[dict[str, Any]]
    next_route: str
    current_step: str


class MultiAgentQualityGraphWorkflow:
    """multi-agent-quality-graph 模式执行器。"""

    ACTIVE_AGENTS = [
        "Orchestrator",
        "Quality Gate",
        "Planner Agent",
        "Retrieval Agent",
        "Generator Agent",
        "Reviewer Agent",
        "Optimizer Agent",
        "Validator",
        "Finalizer Agent",
    ]

    def __init__(
        self,
        workflow: Any,
        *,
        max_agent_rounds: int | None = None,
        quality_threshold: float | None = None,
        candidate_pool_size: int | None = None,
        stop_on_no_improvement_rounds: int | None = None,
    ):
        self.workflow = workflow
        self.max_agent_rounds = (
            max_agent_rounds if max_agent_rounds is not None else settings.max_agent_rounds
        )
        self.quality_threshold = (
            quality_threshold if quality_threshold is not None else settings.quality_threshold
        )
        self.candidate_pool_size = (
            candidate_pool_size if candidate_pool_size is not None else settings.candidate_pool_size
        )
        self.stop_on_no_improvement_rounds = (
            stop_on_no_improvement_rounds
            if stop_on_no_improvement_rounds is not None
            else settings.stop_on_no_improvement_rounds
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
        initial_state: MultiAgentQualityState = {
            "user_input": user_input,
            "additional_instructions": additional_instructions,
            "images": images or [],
            "output_format": output_format,
            "requirement_analysis": {},
            "retrieval_context": {"query": "", "documents": [], "source_summary": ""},
            "candidate_pool": [],
            "current_candidate_id": "",
            "review_reports": [],
            "revision_plan": [],
            "validation_reports": [],
            "best_candidate": "",
            "best_candidate_score": 0.0,
            "final_test_cases": "",
            "agent_rounds": 0,
            "no_improvement_rounds": 0,
            "errors": [],
            "warnings": [],
            "agent_trace": [],
            "active_agents": list(self.ACTIVE_AGENTS),
            "quality_report": {},
            "quality_passed": False,
            "validation_passed": False,
            "orchestrator_decision": {},
            "orchestrator_decisions": [],
            "next_route": "optimizer",
            "current_step": "start",
        }
        config = {"configurable": {"thread_id": f"multi-agent-quality-graph-{uuid4()}"}}
        return self._graph.invoke(initial_state, config=config)

    def _build_graph(self):
        graph = StateGraph(MultiAgentQualityState)
        graph.add_node("planner", self._planner)
        graph.add_node("retrieval", self._retrieval)
        graph.add_node("generator", self._generator)
        graph.add_node("reviewer", self._reviewer)
        graph.add_node("validator", self._validator)
        graph.add_node("orchestrator", self._orchestrator)
        graph.add_node("optimizer", self._optimizer)
        graph.add_node("finalizer", self._finalizer)

        graph.set_entry_point("planner")
        graph.add_edge("planner", "retrieval")
        graph.add_edge("retrieval", "generator")
        graph.add_edge("generator", "reviewer")
        graph.add_edge("reviewer", "validator")
        graph.add_edge("validator", "orchestrator")
        graph.add_conditional_edges(
            "orchestrator",
            self._route_after_orchestrator,
            {
                "optimizer": "optimizer",
                "finalizer": "finalizer",
            },
        )
        graph.add_edge("optimizer", "reviewer")
        graph.add_edge("finalizer", END)
        return graph.compile(checkpointer=self.checkpointer)

    def _planner(self, state: MultiAgentQualityState) -> dict[str, Any]:
        trace = self._trace(state, "Planner Agent", "输出需求分析、测试范围、风险点和测试策略")
        scope_text = state["user_input"]
        errors = state.get("errors", [])

        if self.workflow.enable_analyzer and AnalyzerNode.should_analyze(
            state["user_input"],
            self.workflow.analyzer_complexity_threshold,
        ):
            try:
                with self._without_node_rag(self.workflow.analyzer):
                    output = self.workflow.analyzer.invoke(
                        user_input=state["user_input"],
                        additional_instructions=state.get("additional_instructions", ""),
                    )
                scope_text = output.content or scope_text
                if output.is_truncated:
                    errors = errors + [output.truncation_warning]
            except Exception as exc:
                errors = errors + [f"Planner Agent错误: {exc}"]

        return {
            "requirement_analysis": {
                "scope": scope_text,
                "risks": self._extract_risks(state["user_input"]),
                "strategy": "覆盖主流程、异常路径、边界条件和输出格式约束。",
            },
            "errors": errors,
            "agent_trace": trace,
            "current_step": "planner_complete",
        }

    def _retrieval(self, state: MultiAgentQualityState) -> dict[str, Any]:
        trace = self._trace(state, "Retrieval Agent", "按 Planner 产物选择性检索参考资料")
        rag_interface = getattr(self.workflow, "rag_interface", None)
        if rag_interface is None or not rag_interface.is_enabled():
            return {
                "retrieval_context": {"query": state["user_input"], "documents": [], "source_summary": "RAG未启用。"},
                "agent_trace": trace,
                "current_step": "retrieval_skipped",
            }

        try:
            documents = list(rag_interface.retrieve(state["user_input"]))
            return {
                "retrieval_context": {
                    "query": state["user_input"],
                    "documents": documents,
                    "source_summary": f"检索到 {len(documents)} 条参考资料。",
                },
                "agent_trace": trace,
                "current_step": "retrieval_complete",
            }
        except Exception as exc:
            return {
                "retrieval_context": {"query": state["user_input"], "documents": [], "source_summary": "RAG检索失败。"},
                "errors": state.get("errors", []) + [f"Retrieval Agent错误: {exc}"],
                "agent_trace": trace,
                "current_step": "retrieval_error",
            }

    def _generator(self, state: MultiAgentQualityState) -> dict[str, Any]:
        trace = self._trace(state, "Generator Agent", "基于需求范围和检索上下文生成候选")
        errors = state.get("errors", [])
        try:
            with self._without_node_rag(self.workflow.generator):
                output = self.workflow.generator.invoke(
                    user_input=state["user_input"],
                    additional_instructions=self._instructions_with_retrieval(state),
                    images=state.get("images", []),
                    analysis_result=str(state.get("requirement_analysis", {})),
                )
            errors = errors + ([output.truncation_warning] if output.is_truncated else [])
            return self._add_candidate_update(
                state,
                content=output.content,
                source_agent="Generator Agent",
                step="generator",
                errors=errors,
                trace=trace,
            )
        except Exception as exc:
            return {
                "errors": errors + [f"Generator Agent错误: {exc}"],
                "warnings": state.get("warnings", []) + ["生成候选失败，无法推进到有效候选。"],
                "agent_trace": trace,
                "current_step": "generator_error",
            }

    def _reviewer(self, state: MultiAgentQualityState) -> dict[str, Any]:
        trace = self._trace(state, "Reviewer Agent", "评审候选并输出结构化遗漏与修订建议")
        candidate = self._current_candidate(state)
        feedback = ""
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
                errors = errors + [f"Reviewer Agent错误: {exc}"]

        report = build_quality_report(
            candidate,
            user_input=state["user_input"],
            reviewer_feedback=feedback,
            output_format=state.get("output_format", "markdown"),
            frontend_backend_mode=self.workflow._is_frontend_backend_mode(),
            node_warnings=errors,
            quality_threshold=self.quality_threshold,
        )
        review_report = self._structured_review_report(feedback, report)
        pool = CandidatePool(state.get("candidate_pool", []), max_size=self.candidate_pool_size)
        pool.update_quality(
            state.get("current_candidate_id", ""),
            review_summary=review_report,
            quality_score=report.score,
        )
        best = pool.best()
        previous_best = float(state.get("best_candidate_score", 0.0) or 0.0)
        no_improvement = int(state.get("no_improvement_rounds", 0))
        if best and report.score <= previous_best and state.get("candidate_pool"):
            no_improvement += 1
        else:
            no_improvement = 0

        return {
            "candidate_pool": pool.to_list(),
            "review_reports": state.get("review_reports", []) + [review_report],
            "revision_plan": build_revision_plan(report),
            "quality_report": report.to_dict(),
            "best_candidate": best.content if best else state.get("best_candidate", ""),
            "best_candidate_score": best.quality_score if best else state.get("best_candidate_score", 0.0),
            "no_improvement_rounds": no_improvement,
            "quality_passed": report.passed,
            "errors": errors,
            "agent_trace": trace,
            "current_step": "reviewer_complete",
        }

    def _validator(self, state: MultiAgentQualityState) -> dict[str, Any]:
        trace = self._trace(state, "Validator", "执行确定性格式、结构和禁用元信息校验")
        candidate = self._current_candidate(state) or state.get("best_candidate", "")
        validation = validate_candidate_structure(
            candidate,
            frontend_backend_mode=self.workflow._is_frontend_backend_mode(),
        )
        report = state.get("quality_report", {})
        deterministic_issues = (
            list(report.get("format_issues", []))
            + list(report.get("frontend_backend_issues", []))
            + list(report.get("truncation_issues", []))
        )
        if deterministic_issues:
            validation = {
                **validation,
                "passed": False,
                "issues": list(dict.fromkeys(list(validation.get("issues", [])) + deterministic_issues)),
            }
        validation = {
            **validation,
            "candidate_id": state.get("current_candidate_id", ""),
            "candidate_preview": candidate[:120],
        }
        pool = CandidatePool(state.get("candidate_pool", []), max_size=self.candidate_pool_size)
        pool.update_quality(
            state.get("current_candidate_id", ""),
            validation_summary=validation,
            is_valid=bool(validation.get("passed")),
        )
        best = pool.best()
        return {
            "candidate_pool": pool.to_list(),
            "validation_reports": state.get("validation_reports", []) + [validation],
            "validation_passed": bool(validation.get("passed")),
            "best_candidate": best.content if best else state.get("best_candidate", ""),
            "best_candidate_score": best.quality_score if best else state.get("best_candidate_score", 0.0),
            "agent_trace": trace,
            "current_step": "validator_complete",
        }

    def _orchestrator(self, state: MultiAgentQualityState) -> dict[str, Any]:
        trace = self._trace(state, "Orchestrator / Quality Gate", "根据质量报告和 Validator 结果统一路由")
        report = self._report_from_state(state)
        validation_passed = bool(state.get("validation_passed", False))
        review_blocking = bool(report.blocking or not report.passed)
        validation_failed = bool(state.get("validation_reports")) and not validation_passed
        revision_plan = list(state.get("revision_plan", []))
        warnings = state.get("warnings", [])
        route = "finalizer"
        reason = "Reviewer 与 Validator 均通过。"

        if validation_failed:
            last_validation = state.get("validation_reports", [])[-1]
            issues = "；".join(last_validation.get("issues", []))
            revision_plan = revision_plan + [f"修复确定性校验失败：{issues}"]

        if review_blocking or validation_failed:
            route = "optimizer"
            reason = "存在评审阻塞问题或确定性校验失败。"

        if int(state.get("agent_rounds", 0)) >= self.max_agent_rounds:
            route = "finalizer"
            reason = "达到最大多 Agent 修订轮次，返回最佳候选。"
            warnings = warnings + ["已达到最大多 Agent 修订轮次，返回当前最佳候选。"]

        if int(state.get("no_improvement_rounds", 0)) >= self.stop_on_no_improvement_rounds:
            route = "finalizer"
            reason = "候选质量连续多轮未提升，停止继续修订。"
            warnings = warnings + ["候选质量连续多轮未提升，返回当前最佳候选。"]

        if not state.get("candidate_pool"):
            route = "finalizer"
            reason = "没有可用候选，停止并返回失败证据。"
            warnings = warnings + ["没有可用候选输出。"]

        decision = {
            "route": route,
            "reason": reason,
            "review_blocking": review_blocking,
            "validation_failed": validation_failed,
            "revision_plan": revision_plan,
        }
        return {
            "next_route": route,
            "orchestrator_decision": decision,
            "orchestrator_decisions": state.get("orchestrator_decisions", []) + [decision],
            "revision_plan": revision_plan,
            "warnings": warnings,
            "agent_trace": trace,
            "current_step": f"orchestrator_{route}",
        }

    def _optimizer(self, state: MultiAgentQualityState) -> dict[str, Any]:
        trace = self._trace(state, "Optimizer Agent", "按评审报告和校验失败项修订当前最佳候选")
        candidate = state.get("best_candidate") or self._current_candidate(state)
        feedback = self._optimizer_feedback(state, candidate)
        errors = state.get("errors", [])
        round_no = int(state.get("agent_rounds", 0)) + 1
        try:
            with self._without_node_rag(self.workflow.optimizer):
                output = self.workflow.optimizer.invoke(
                    original_input=state["user_input"],
                    initial_test_cases=candidate,
                    review_feedback=feedback,
                    output_format=state.get("output_format", "markdown"),
                )
            errors = errors + ([output.truncation_warning] if output.is_truncated else [])
            return self._add_candidate_update(
                state,
                content=output.content,
                source_agent="Optimizer Agent",
                step="optimizer",
                errors=errors,
                trace=trace,
                agent_rounds=round_no,
            )
        except Exception as exc:
            return {
                "agent_rounds": round_no,
                "errors": errors + [f"Optimizer Agent错误: {exc}"],
                "warnings": state.get("warnings", []) + ["修订失败，保留当前最佳候选。"],
                "agent_trace": trace,
                "current_step": "optimizer_error",
            }

    def _finalizer(self, state: MultiAgentQualityState) -> dict[str, Any]:
        trace = self._trace(state, "Finalizer Agent", "汇总已通过候选或返回最佳候选和失败证据")
        pool = CandidatePool(state.get("candidate_pool", []), max_size=self.candidate_pool_size)
        best = pool.best()
        passed_candidate = None
        for record in pool.to_list():
            if record.get("is_valid") and record.get("quality_score", 0.0) >= self.quality_threshold:
                passed_candidate = record
                break
        final = (passed_candidate or (best.to_dict() if best else {})).get("content", "")
        warnings = state.get("warnings", [])
        quality_passed = bool(passed_candidate)
        if not quality_passed and final:
            warnings = warnings + ["没有通过全部质量门的候选，返回当前最佳候选和失败证据。"]
        elif not final:
            warnings = warnings + ["没有可交付候选输出。"]
        return {
            "final_test_cases": final,
            "quality_passed": quality_passed,
            "warnings": warnings,
            "agent_trace": trace,
            "current_step": "finalizer_complete",
        }

    def _add_candidate_update(
        self,
        state: MultiAgentQualityState,
        *,
        content: str,
        source_agent: str,
        step: str,
        errors: list[str],
        trace: list[dict[str, Any]],
        agent_rounds: int | None = None,
    ) -> dict[str, Any]:
        pool = CandidatePool(state.get("candidate_pool", []), max_size=self.candidate_pool_size)
        record = pool.add_candidate(
            content,
            source_agent=source_agent,
            round=int(state.get("agent_rounds", 0)),
            created_at_step=step,
        )
        return {
            "candidate_pool": pool.to_list(),
            "current_candidate_id": record.id,
            "errors": errors,
            "agent_trace": trace,
            "agent_rounds": int(state.get("agent_rounds", 0)) if agent_rounds is None else agent_rounds,
            "current_step": f"{step}_complete",
        }

    def _route_after_orchestrator(self, state: MultiAgentQualityState) -> str:
        return state.get("next_route", "optimizer")

    def _trace(self, state: MultiAgentQualityState, agent: str, detail: str) -> list[dict[str, Any]]:
        return state.get("agent_trace", []) + [
            {
                "agent": agent,
                "node": agent,
                "detail": detail,
                "round": int(state.get("agent_rounds", 0)),
            }
        ]

    def _current_candidate(self, state: MultiAgentQualityState) -> str:
        pool = CandidatePool(state.get("candidate_pool", []), max_size=self.candidate_pool_size)
        record = pool.get(state.get("current_candidate_id", ""))
        return record.content if record else ""

    def _report_from_state(self, state: MultiAgentQualityState) -> QualityReport:
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

    def _structured_review_report(self, feedback: str, report: QualityReport) -> dict[str, Any]:
        return {
            "blocking_issues": (
                list(report.coverage_issues)
                + list(report.format_issues)
                + list(report.frontend_backend_issues)
                + list(report.truncation_issues)
            ),
            "coverage_gaps": list(report.coverage_issues),
            "format_issues": list(report.format_issues) + list(report.frontend_backend_issues),
            "duplication_issues": list(report.redundancy_issues),
            "revision_suggestions": build_revision_plan(report),
            "raw_feedback": feedback,
            "quality_score": report.score,
            "passed": report.passed,
        }

    def _optimizer_feedback(self, state: MultiAgentQualityState, candidate: str) -> str:
        plan_text = "\n".join(f"- {item}" for item in state.get("revision_plan", []))
        return (
            f"{(state.get('review_reports') or [{}])[-1].get('raw_feedback', '')}\n\n"
            "## 多 Agent 修订计划\n"
            f"{plan_text}\n\n"
            "## Validator 失败证据\n"
            f"{state.get('validation_reports', [])[-1:]}\n\n"
            "## 当前最佳候选\n"
            f"{candidate}\n\n"
            "只输出修订后的测试用例正文，不要输出质量报告、trace、metadata 或解释。"
        )

    def _instructions_with_retrieval(self, state: MultiAgentQualityState) -> str:
        instructions = state.get("additional_instructions", "")
        retrieval = state.get("retrieval_context", {})
        documents = retrieval.get("documents") or []
        if not documents:
            return instructions
        refs = "\n".join(f"{idx}. {doc}" for idx, doc in enumerate(documents, 1))
        return f"{instructions}\n\n## Retrieval Agent 参考上下文\n{refs}"

    def _extract_risks(self, user_input: str) -> list[str]:
        risks = []
        if any(item in user_input for item in ["锁定", "失败", "异常"]):
            risks.append("异常路径和安全边界容易遗漏。")
        if any(item in user_input for item in ["退款", "优惠券", "积分", "开票", "风控"]):
            risks.append("跨域副作用和补偿流程需要覆盖。")
        return risks or ["需求范围较小，重点验证主流程和边界条件。"]

    @contextmanager
    def _without_node_rag(self, node: Any):
        original = getattr(node, "rag_interface", None)
        node.rag_interface = None
        try:
            yield
        finally:
            node.rag_interface = original
