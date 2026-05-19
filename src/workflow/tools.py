"""
ReAct 模式使用的受限工具封装。

这些工具只暴露当前仓库已有的需求分析、RAG 检索、生成、评审、优化、
结构校验和结构修复能力，不提供任意代码执行、文件写入或外部 API 工具。
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.workflow.nodes import AnalyzerNode, NodeOutput
from src.workflow.validators import validate_fe_be_structure


@dataclass
class ToolExecutionResult:
    """工具执行结果，供 Agent 循环和测试断言使用。"""

    ok: bool
    content: str = ""
    issues: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_observation(self) -> str:
        status = "ok" if self.ok else "error"
        parts = [f"status: {status}"]
        if self.content:
            parts.append(f"content:\n{self.content}")
        if self.issues:
            parts.append("issues:\n" + "\n".join(f"- {issue}" for issue in self.issues))
        if self.metadata:
            safe_metadata = {
                key: value
                for key, value in self.metadata.items()
                if "key" not in key.lower() and "prompt" not in key.lower()
            }
            if safe_metadata:
                parts.append(f"metadata: {safe_metadata}")
        return "\n".join(parts)


@dataclass
class ReactToolState:
    """ReAct 工具之间共享的窄状态。"""

    user_input: str
    additional_instructions: str = ""
    images: list[dict] = field(default_factory=list)
    output_format: str = "markdown"
    analysis_result: str = ""
    rag_context: str = ""
    generated_test_cases: str = ""
    review_feedback: str = ""
    final_test_cases: str = ""
    validation_passed: bool = True
    validation_issues: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    agent_steps: list[dict[str, Any]] = field(default_factory=list)
    review_has_blocking_issues: bool = False


class EmptyArgs(BaseModel):
    """无参数工具输入。"""


class RagRetrieveArgs(BaseModel):
    query: str = Field(..., description="用于检索测试知识库的查询")


class ValidateOutputArgs(BaseModel):
    candidate_content: str = Field("", description="待校验内容；为空时校验当前最终测试用例")


class RepairOutputArgs(BaseModel):
    candidate_content: str = Field("", description="待修复内容；为空时修复当前最终测试用例")


class ReactWorkflowTools:
    """将现有工作流节点封装为 ReAct 可调用的白名单工具。"""

    def __init__(self, workflow: Any, state: ReactToolState):
        self.workflow = workflow
        self.state = state

    def as_langchain_tools(self) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(
                name="analyze_requirements",
                description="分析复杂需求并提炼测试范围。需要时调用，输出会供生成工具使用。",
                func=self.analyze_requirements,
                args_schema=EmptyArgs,
            ),
            StructuredTool.from_function(
                name="retrieve_rag_context",
                description="从已配置的 RAG 知识库检索相关测试方法资料。仅在需要参考知识时调用。",
                func=self.retrieve_rag_context,
                args_schema=RagRetrieveArgs,
            ),
            StructuredTool.from_function(
                name="generate_test_cases",
                description="基于需求、可选分析结果和可选 RAG 上下文生成初始测试用例。",
                func=self.generate_test_cases,
                args_schema=EmptyArgs,
            ),
            StructuredTool.from_function(
                name="review_test_cases",
                description="评审当前初始测试用例，找出阻塞问题和改进建议。",
                func=self.review_test_cases,
                args_schema=EmptyArgs,
            ),
            StructuredTool.from_function(
                name="optimize_test_cases",
                description="根据评审反馈优化当前测试用例，产出候选最终结果。",
                func=self.optimize_test_cases,
                args_schema=EmptyArgs,
            ),
            StructuredTool.from_function(
                name="validate_output_structure",
                description="校验当前最终输出结构，frontend_backend 模式会检查三列表格结构。",
                func=self.validate_output_structure,
                args_schema=ValidateOutputArgs,
            ),
            StructuredTool.from_function(
                name="repair_output_structure",
                description="按结构校验问题修复当前最终输出，特别用于 frontend_backend 模式。",
                func=self.repair_output_structure,
                args_schema=RepairOutputArgs,
            ),
        ]

    def tool_map(self) -> dict[str, Callable[..., str]]:
        return {tool.name: tool.func for tool in self.as_langchain_tools()}

    def analyze_requirements(self) -> str:
        if not self.workflow.enable_analyzer:
            result = ToolExecutionResult(
                ok=True,
                content="",
                metadata={"skipped": True, "reason": "analyzer disabled"},
            )
            return result.to_observation()

        if not AnalyzerNode.should_analyze(
            self.state.user_input,
            self.workflow.analyzer_complexity_threshold,
        ):
            result = ToolExecutionResult(
                ok=True,
                content="",
                metadata={"skipped": True, "reason": "simple requirement"},
            )
            return result.to_observation()

        try:
            with self._without_node_rag(self.workflow.analyzer):
                output = self.workflow.analyzer.invoke(
                    user_input=self.state.user_input,
                    additional_instructions=self.state.additional_instructions,
                )
            self._merge_node_output_warning(output)
            self.state.analysis_result = output.content
            return ToolExecutionResult(ok=True, content=output.content).to_observation()
        except Exception as exc:
            return self._record_error("分析工具错误", exc).to_observation()

    def retrieve_rag_context(self, query: str) -> str:
        rag_interface = self.workflow.rag_interface
        if rag_interface is None or not rag_interface.is_enabled():
            return ToolExecutionResult(
                ok=True,
                content="",
                metadata={"skipped": True, "reason": "rag disabled"},
            ).to_observation()

        try:
            documents = rag_interface.retrieve(query)
            context = "\n".join(f"{idx}. {doc}" for idx, doc in enumerate(documents, 1))
            self.state.rag_context = context
            return ToolExecutionResult(
                ok=True,
                content=context,
                metadata={"documents": len(documents)},
            ).to_observation()
        except Exception as exc:
            return self._record_error("RAG检索工具错误", exc).to_observation()

    def generate_test_cases(self) -> str:
        try:
            instructions = self._instructions_with_rag()
            with self._without_node_rag(self.workflow.generator):
                output = self.workflow.generator.invoke(
                    user_input=self.state.user_input,
                    additional_instructions=instructions,
                    images=self.state.images,
                    analysis_result=self.state.analysis_result,
                )
            self._merge_node_output_warning(output)
            self.state.generated_test_cases = output.content
            return ToolExecutionResult(ok=True, content=output.content).to_observation()
        except Exception as exc:
            return self._record_error("生成工具错误", exc).to_observation()

    def review_test_cases(self) -> str:
        if not self.state.generated_test_cases:
            return ToolExecutionResult(
                ok=False,
                issues=["缺少 generated_test_cases，需先调用 generate_test_cases。"],
            ).to_observation()

        try:
            with self._without_node_rag(self.workflow.reviewer):
                output = self.workflow.reviewer.invoke(
                    original_input=self.state.user_input,
                    test_cases=self.state.generated_test_cases,
                )
            self._merge_node_output_warning(output)
            self.state.review_feedback = output.content
            self.state.review_has_blocking_issues = self._has_blocking_review(output.content)
            return ToolExecutionResult(
                ok=True,
                content=output.content,
                metadata={"has_blocking_issues": self.state.review_has_blocking_issues},
            ).to_observation()
        except Exception as exc:
            return self._record_error("评审工具错误", exc).to_observation()

    def optimize_test_cases(self) -> str:
        if not self.state.generated_test_cases:
            return ToolExecutionResult(
                ok=False,
                issues=["缺少 generated_test_cases，需先调用 generate_test_cases。"],
            ).to_observation()

        try:
            feedback = self.state.review_feedback or "未提供评审反馈，请保持现有覆盖并按输出格式整理。"
            with self._without_node_rag(self.workflow.optimizer):
                output = self.workflow.optimizer.invoke(
                    original_input=self.state.user_input,
                    initial_test_cases=self.state.generated_test_cases,
                    review_feedback=feedback,
                    output_format=self.state.output_format,
                )
            self._merge_node_output_warning(output)
            self.state.final_test_cases = output.content
            return ToolExecutionResult(ok=True, content=output.content).to_observation()
        except Exception as exc:
            return self._record_error("优化工具错误", exc).to_observation()

    def validate_output_structure(self, candidate_content: str = "") -> str:
        content = candidate_content or self.state.final_test_cases
        if not content:
            self.state.validation_passed = False
            self.state.validation_issues = ["最终测试用例为空。"]
            return ToolExecutionResult(ok=False, issues=self.state.validation_issues).to_observation()

        if not self.workflow._is_frontend_backend_mode():
            self.state.validation_passed = True
            self.state.validation_issues = []
            return ToolExecutionResult(
                ok=True,
                content="非 frontend_backend 模式，无需前后端结构校验。",
                metadata={"validation_passed": True},
            ).to_observation()

        validation = validate_fe_be_structure(content)
        self.state.validation_passed = validation.is_valid
        self.state.validation_issues = list(validation.issues)
        return ToolExecutionResult(
            ok=validation.is_valid,
            issues=list(validation.issues),
            metadata={
                "validation_passed": validation.is_valid,
                "repair_hint": validation.repair_hint,
            },
        ).to_observation()

    def repair_output_structure(self, candidate_content: str = "") -> str:
        content = candidate_content or self.state.final_test_cases
        if not content:
            return ToolExecutionResult(ok=False, issues=["没有可修复的最终测试用例。"]).to_observation()

        validation = validate_fe_be_structure(content)
        if validation.is_valid:
            self.state.validation_passed = True
            self.state.validation_issues = []
            return ToolExecutionResult(ok=True, content=content, metadata={"already_valid": True}).to_observation()

        repair_requirements = "\n".join(f"- {issue}" for issue in validation.issues)
        repair_feedback = (
            f"{self.state.review_feedback}\n\n"
            "## 输出结构修复要求\n"
            f"{validation.repair_hint}\n\n"
            "## 当前结构问题\n"
            f"{repair_requirements}\n\n"
            "## 待修复测试用例\n"
            f"{content}"
        )

        try:
            with self._without_node_rag(self.workflow.optimizer):
                output = self.workflow.optimizer.invoke(
                    original_input=self.state.user_input,
                    initial_test_cases=self.state.generated_test_cases or content,
                    review_feedback=repair_feedback,
                    output_format=self.state.output_format,
                )
            self._merge_node_output_warning(output)
            self.state.final_test_cases = output.content
            retry_validation = validate_fe_be_structure(output.content)
            self.state.validation_passed = retry_validation.is_valid
            self.state.validation_issues = list(retry_validation.issues)
            return ToolExecutionResult(
                ok=retry_validation.is_valid,
                content=output.content,
                issues=list(retry_validation.issues),
                metadata={"validation_passed": retry_validation.is_valid},
            ).to_observation()
        except Exception as exc:
            return self._record_error("结构修复工具错误", exc).to_observation()

    def _instructions_with_rag(self) -> str:
        if not self.state.rag_context:
            return self.state.additional_instructions
        rag_block = f"\n\n## RAG 检索上下文（由 Agent 按需调用）\n{self.state.rag_context}"
        return f"{self.state.additional_instructions}{rag_block}"

    def _merge_node_output_warning(self, output: NodeOutput) -> None:
        if output.is_truncated and output.truncation_warning:
            self.state.errors.append(output.truncation_warning)

    def _record_error(self, prefix: str, exc: Exception) -> ToolExecutionResult:
        message = f"{prefix}: {exc}"
        self.state.errors.append(message)
        return ToolExecutionResult(ok=False, issues=[message])

    @staticmethod
    def _has_blocking_review(feedback: str) -> bool:
        normalized = feedback.strip().lower()
        if not normalized:
            return False
        pass_markers = ["无阻塞", "无严重", "通过", "no blocking", "no blocker", "approved"]
        fail_markers = ["阻塞", "严重", "必须修复", "blocker", "blocking", "critical"]
        if any(marker in normalized for marker in pass_markers):
            return False
        return any(marker in normalized for marker in fail_markers)

    @contextmanager
    def _without_node_rag(self, node: Any):
        original = getattr(node, "rag_interface", None)
        node.rag_interface = None
        try:
            yield
        finally:
            node.rag_interface = original
