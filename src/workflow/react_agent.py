"""
传统 ReAct 风格的测试用例生成 Agent。

实现基于 LangChain tool calling：模型只能调用 tools.py 中的白名单工具，
循环在满足停止条件或达到最大步数时结束。
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from config.settings import settings
from src.workflow.graph_types import WorkflowResultProtocol
from src.workflow.tools import ReactToolState, ReactWorkflowTools


class ReactAgentDependencyError(RuntimeError):
    """当前依赖或模型不支持 ReAct tool calling 时抛出。"""


class TestCaseReactAgent:
    """受限工具白名单 ReAct Agent 执行器。"""

    def __init__(
        self,
        workflow: Any,
        llm: Any = None,
        max_steps: Optional[int] = None,
    ):
        self.workflow = workflow
        self.llm = llm
        self.max_steps = max_steps if max_steps is not None else settings.max_agent_steps

    def run(
        self,
        user_input: str,
        additional_instructions: str = "",
        images: Optional[list[dict]] = None,
        output_format: str = "markdown",
    ) -> WorkflowResultProtocol:
        state = ReactToolState(
            user_input=user_input,
            additional_instructions=additional_instructions,
            images=images or [],
            output_format=output_format,
            validation_passed=not self.workflow._is_frontend_backend_mode(),
        )
        toolset = ReactWorkflowTools(self.workflow, state)
        tools = toolset.as_langchain_tools()
        tool_map = {tool.name: tool for tool in tools}
        llm = self._get_llm_with_tools(tools)

        messages: list[Any] = [
            SystemMessage(content=self._system_prompt(output_format)),
            HumanMessage(content=self._user_prompt(user_input, additional_instructions)),
        ]

        reached_stop = False
        step_count = 0
        while step_count < self.max_steps:
            response = llm.invoke(messages)
            messages.append(response)
            tool_calls = self._extract_tool_calls(response)

            if not tool_calls:
                final_text = self._extract_text(response)
                if final_text and not state.final_test_cases:
                    state.final_test_cases = final_text
                reached_stop = self._should_stop(state)
                step_count += 1
                state.agent_steps.append({
                    "step": step_count,
                    "type": "final",
                    "content_preview": final_text[:200],
                    "stop": reached_stop,
                })
                break

            for tool_call in tool_calls:
                if step_count >= self.max_steps:
                    break
                step_count += 1
                tool_name = tool_call["name"]
                tool_args = tool_call.get("args") or {}
                tool_id = tool_call.get("id") or f"tool-{step_count}-{tool_name}"

                if tool_name not in tool_map:
                    observation = f"status: error\nissues:\n- 工具不在白名单中: {tool_name}"
                else:
                    try:
                        observation = tool_map[tool_name].invoke(tool_args)
                    except Exception as exc:
                        observation = f"status: error\nissues:\n- {tool_name} 调用失败: {exc}"
                        state.errors.append(f"{tool_name} 调用失败: {exc}")

                state.tools_used.append(tool_name)
                state.agent_steps.append({
                    "step": step_count,
                    "type": "tool",
                    "tool": tool_name,
                    "args": self._safe_args(tool_args),
                    "observation_preview": observation[:300],
                })
                messages.append(ToolMessage(content=observation, tool_call_id=tool_id))

                if self._should_stop(state):
                    reached_stop = True
                    break

            if reached_stop:
                break

        if not reached_stop and step_count >= self.max_steps:
            state.errors.append(f"ReAct Agent达到最大步数限制: {self.max_steps}")

        if (
            state.final_test_cases
            and self.workflow._is_frontend_backend_mode()
            and not state.validation_passed
            and step_count < self.max_steps
        ):
            validation_observation = toolset.validate_output_structure()
            step_count += 1
            state.tools_used.append("validate_output_structure")
            state.agent_steps.append({
                "step": step_count,
                "type": "tool",
                "tool": "validate_output_structure",
                "args": {},
                "observation_preview": validation_observation[:300],
            })

        success = bool(state.final_test_cases) and state.validation_passed and not state.review_has_blocking_issues
        return WorkflowResultProtocol(
            success=success,
            final_test_cases=state.final_test_cases,
            generated_test_cases=state.generated_test_cases,
            review_feedback=state.review_feedback,
            errors=state.errors,
            metadata={
                "agent_mode": "react",
                "agent_steps": state.agent_steps,
                "tools_used": state.tools_used,
                "validation_passed": state.validation_passed,
                "output_format": output_format,
                "max_agent_steps": self.max_steps,
                "reached_stop_condition": reached_stop,
            },
        )

    def _get_llm_with_tools(self, tools: list[Any]) -> Any:
        llm = self.llm or self.workflow.generator._get_llm()
        if not hasattr(llm, "bind_tools"):
            raise ReactAgentDependencyError(
                "当前模型客户端不支持 bind_tools，无法运行 react 模式。请使用支持 tool calling 的 LangChain ChatModel。"
            )
        try:
            return llm.bind_tools(tools)
        except Exception as exc:
            raise ReactAgentDependencyError(
                f"绑定 ReAct 工具失败，当前依赖或模型可能不支持 tool calling: {exc}"
            ) from exc

    def _should_stop(self, state: ReactToolState) -> bool:
        return bool(state.final_test_cases) and state.validation_passed and not state.review_has_blocking_issues

    @staticmethod
    def _extract_tool_calls(response: Any) -> list[dict[str, Any]]:
        tool_calls = getattr(response, "tool_calls", None)
        if tool_calls:
            return list(tool_calls)
        additional_kwargs = getattr(response, "additional_kwargs", {}) or {}
        return list(additional_kwargs.get("tool_calls") or [])

    @staticmethod
    def _extract_text(response: Any) -> str:
        content = getattr(response, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "".join(parts)
        return str(content) if content else ""

    @staticmethod
    def _safe_args(args: dict[str, Any]) -> dict[str, Any]:
        safe: dict[str, Any] = {}
        for key, value in args.items():
            if "key" in key.lower() or "prompt" in key.lower():
                safe[key] = "***"
            else:
                text = str(value)
                safe[key] = text[:200] if len(text) > 200 else value
        return safe

    @staticmethod
    def _system_prompt(output_format: str) -> str:
        return (
            "你是传统 ReAct 测试用例生成 Agent。你只能调用提供的白名单工具，"
            "不得请求 shell、文件写入或外部 API。按需选择需求分析、RAG 检索、生成、评审、优化、校验和修复工具。"
            "停止条件：最终测试用例非空、输出结构校验通过、评审无阻塞问题。"
            f"必须保持 {output_format} 输出约束；frontend_backend 模式必须通过结构校验。"
        )

    @staticmethod
    def _user_prompt(user_input: str, additional_instructions: str) -> str:
        if additional_instructions:
            return f"需求：\n{user_input}\n\n额外指示：\n{additional_instructions}"
        return f"需求：\n{user_input}"
