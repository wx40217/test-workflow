"""
测试用例生成工作流的LLM节点模块。

本模块定义三个主要节点：
1. GeneratorNode - 根据用户输入生成初始测试用例
2. ReviewerNode - 评审测试用例并提供反馈
3. OptimizerNode - 根据反馈优化测试用例
"""

from typing import Any, Optional
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI

import sys
sys.path.insert(0, '/workspace')

from config.settings import ModelConfig, settings
from config.prompts import PromptTemplates
from src.rag.interface import RAGInterface


@dataclass
class NodeOutput:
    """节点输出结果，包含内容和截断检测信息。"""
    content: str
    is_truncated: bool = False
    truncation_warning: str = ""


class TruncationDetector:
    """输出截断检测器，基于 API 返回的 finish_reason。"""

    NODE_NAMES = {
        "analyzer": "分析器",
        "generator": "生成器",
        "reviewer": "评审员",
        "optimizer": "优化器"
    }

    @classmethod
    def detect_from_finish_reason(
        cls,
        content: str,
        finish_reason: Optional[str],
        node_type: str = "unknown"
    ) -> NodeOutput:
        """
        基于 finish_reason 构建节点输出。

        参数:
            content: 模型返回的完整文本
            finish_reason: API 返回的结束原因
            node_type: 节点类型（用于生成更具体的警告信息）

        返回:
            NodeOutput 对象
        """
        is_truncated = finish_reason == "length"
        warning = ""
        if is_truncated:
            node_name = cls.NODE_NAMES.get(node_type, node_type)
            warning = (
                f"{node_name}输出被截断（达到 max_tokens 限制），"
                f"建议增加 {node_type.upper()}_MAX_TOKENS 配置"
            )

        return NodeOutput(
            content=content,
            is_truncated=is_truncated,
            truncation_warning=warning
        )

    @classmethod
    def detect_from_response(cls, response: AIMessage, node_type: str = "unknown") -> NodeOutput:
        """
        从 LangChain 响应对象检测截断。

        参数:
            response: LangChain AIMessage 响应对象
            node_type: 节点类型（用于生成更具体的警告信息）

        返回:
            NodeOutput 对象
        """
        content = response.content if isinstance(response.content, str) else str(response.content)
        finish_reason = None
        if hasattr(response, 'response_metadata') and response.response_metadata:
            finish_reason = response.response_metadata.get("finish_reason")
        return cls.detect_from_finish_reason(content, finish_reason, node_type)


class BaseNode:
    """所有LLM节点的基类。"""

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        rag_interface: Optional[RAGInterface] = None
    ):
        """
        初始化节点。

        参数:
            config: 模型配置（未提供时使用默认配置）
            rag_interface: 可选的RAG接口，用于知识检索
        """
        self.config = config
        self.rag_interface = rag_interface
        self._llm = None
        self.stream_to_console = False

    def _get_llm(self) -> ChatOpenAI:
        """获取或创建LLM实例。"""
        if self._llm is None:
            if self.config is None:
                raise ValueError("需要模型配置")

            self._llm = ChatOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                model=self.config.model_name,
                use_responses_api=self.config.use_responses_api,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
                streaming=True,
            )
        return self._llm

    def _get_rag_context(self, query: str) -> str:
        """
        从RAG检索相关上下文（如果启用）。

        参数:
            query: 搜索查询

        返回:
            格式化的检索上下文字符串
        """
        if self.rag_interface is None or not self.rag_interface.is_enabled():
            return ""

        try:
            documents = self.rag_interface.retrieve(query)
            if documents:
                context_parts = []
                for i, doc in enumerate(documents, 1):
                    context_parts.append(f"{i}. {doc}")
                return "\n".join(context_parts)
        except Exception as e:
            print(f"警告: RAG检索失败: {e}")

        return ""

    def _create_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        images: Optional[list[dict]] = None
    ) -> list:
        """
        创建LLM调用的消息列表。

        参数:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            images: 可选的图片列表（用于视觉模型）

        返回:
            消息列表
        """
        messages = [SystemMessage(content=system_prompt)]

        if images:
            # 对于视觉模型，在用户消息中包含图片
            content = [{"type": "text", "text": user_prompt}]
            for img in images:
                content.append(img)
            messages.append(HumanMessage(content=content))
        else:
            messages.append(HumanMessage(content=user_prompt))

        return messages

    def _invoke_llm(self, messages: list, node_type: str = "unknown") -> NodeOutput:
        """
        调用LLM并检测截断。

        所有节点统一使用流式接口，避免网关对长时间无返回请求的超时拦截。

        参数:
            messages: 消息列表
            node_type: 节点类型

        返回:
            NodeOutput 对象（包含内容和截断信息）
        """
        llm = self._get_llm()
        content_parts = []
        finish_reason = None
        has_stream_output = False

        for chunk in llm.stream(messages):
            chunk_text = self._extract_chunk_text(chunk)
            if chunk_text:
                content_parts.append(chunk_text)
                if self.stream_to_console:
                    if not has_stream_output:
                        node_names = {
                            "analyzer": "分析器",
                            "generator": "生成器",
                            "reviewer": "评审员",
                            "optimizer": "优化器"
                        }
                        node_name = node_names.get(node_type, node_type)
                        print(f"\n[{node_name}] 流式输出开始：", flush=True)
                        has_stream_output = True
                    print(chunk_text, end="", flush=True)

            chunk_finish_reason = self._extract_finish_reason(chunk)
            if chunk_finish_reason:
                finish_reason = chunk_finish_reason

        content = "".join(content_parts)
        if self.stream_to_console and has_stream_output:
            print("\n", flush=True)
        return TruncationDetector.detect_from_finish_reason(content, finish_reason, node_type)

    @staticmethod
    def _extract_finish_reason(message: Any) -> Optional[str]:
        """
        从响应对象/分块对象提取 finish_reason。
        """
        if hasattr(message, "response_metadata") and message.response_metadata:
            finish_reason = message.response_metadata.get("finish_reason")
            if finish_reason:
                return finish_reason

        if hasattr(message, "additional_kwargs") and message.additional_kwargs:
            finish_reason = message.additional_kwargs.get("finish_reason")
            if finish_reason:
                return finish_reason

        return None

    @staticmethod
    def _extract_chunk_text(chunk: Any) -> str:
        """
        从流式分块中提取文本内容。

        chunk.content 可能是字符串，也可能是多模态内容列表。
        """
        content = getattr(chunk, "content", "")
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                    continue
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
            return "".join(text_parts)

        return str(content) if content else ""

    def invoke(self, **kwargs) -> str:
        """
        调用节点。必须由子类实现。

        返回:
            LLM响应字符串
        """
        raise NotImplementedError("子类必须实现invoke()方法")


class GeneratorNode(BaseNode):
    """
    节点一：测试用例生成器

    负责根据用户输入生成初始测试用例。
    使用配置的生成器模型（默认与优化器相同）。
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        rag_interface: Optional[RAGInterface] = None
    ):
        super().__init__(config or settings.get_generator_config(), rag_interface)

    def invoke(
        self,
        user_input: str,
        additional_instructions: str = "",
        images: Optional[list[dict]] = None,
        analysis_result: str = ""
    ) -> NodeOutput:
        """
        根据用户输入生成测试用例。

        参数:
            user_input: 用户输入（需求等）
            additional_instructions: 用户的额外指示
            images: 可选的图片（用于视觉模型）
            analysis_result: 需求分析结果（如有）

        返回:
            NodeOutput 对象（包含内容和截断信息）
        """
        # 获取RAG上下文（如果可用）
        rag_context = self._get_rag_context(user_input)

        # 如果有分析结果，合并到输入中
        combined_input = user_input
        if analysis_result:
            combined_input = f"{user_input}\n\n## 需求分析结果：\n{analysis_result}"

        # 获取提示词
        system_prompt, user_prompt = PromptTemplates.get_generator_prompts(
            user_input=combined_input,
            additional_instructions=additional_instructions,
            rag_context=rag_context
        )

        # 创建消息并调用LLM
        messages = self._create_messages(system_prompt, user_prompt, images)
        return self._invoke_llm(messages, "generator")


class AnalyzerNode(BaseNode):
    """
    节点零：需求分析器（可选）

    负责分析用户需求，输出结构化的测试范围定义。
    仅在需求复杂度较高时启用。
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        rag_interface: Optional[RAGInterface] = None
    ):
        super().__init__(config or settings.get_analyzer_config(), rag_interface)

    def invoke(
        self,
        user_input: str,
        additional_instructions: str = ""
    ) -> NodeOutput:
        """
        分析用户需求。

        参数:
            user_input: 用户输入（需求等）
            additional_instructions: 用户的额外指示

        返回:
            NodeOutput 对象（包含内容和截断信息）
        """
        # 获取RAG上下文（如果可用）
        rag_context = self._get_rag_context(user_input)

        # 获取提示词
        system_prompt, user_prompt = PromptTemplates.get_analyzer_prompts(
            user_input=user_input,
            additional_instructions=additional_instructions,
            rag_context=rag_context
        )

        # 创建消息并调用LLM
        messages = self._create_messages(system_prompt, user_prompt)
        return self._invoke_llm(messages, "analyzer")

    @staticmethod
    def should_analyze(user_input: str, threshold: int = 2) -> bool:
        """
        判断是否需要进行需求分析。

        参数:
            user_input: 用户输入
            threshold: 复杂度阈值（满足几个条件时触发分析）

        返回:
            是否需要分析
        """
        complexity_indicators = [
            len(user_input) > 200,  # 需求描述较长
            user_input.count("，") + user_input.count(",") > 5,  # 多个子需求
            any(kw in user_input for kw in ["并且", "同时", "以及", "或者", "包括", "需要"]),  # 复合逻辑
            "?" in user_input or "？" in user_input,  # 有不确定性
            user_input.count("\n") > 3,  # 多行输入
        ]
        return sum(complexity_indicators) >= threshold


class ReviewerNode(BaseNode):
    """
    节点二：测试用例评审员

    负责评审生成的测试用例并提供反馈。
    使用更强大的推理模型进行深入分析。
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        rag_interface: Optional[RAGInterface] = None
    ):
        super().__init__(config or settings.get_reviewer_config(), rag_interface)

    def invoke(
        self,
        original_input: str,
        test_cases: str
    ) -> NodeOutput:
        """
        评审生成的测试用例。

        参数:
            original_input: 原始用户输入
            test_cases: 生成器节点生成的测试用例

        返回:
            NodeOutput 对象（包含内容和截断信息）
        """
        # 获取RAG上下文（如果可用）
        rag_context = self._get_rag_context(original_input)

        # 获取提示词
        system_prompt, user_prompt = PromptTemplates.get_reviewer_prompts(
            original_input=original_input,
            test_cases=test_cases,
            rag_context=rag_context
        )

        # 创建消息并调用LLM
        messages = self._create_messages(system_prompt, user_prompt)
        return self._invoke_llm(messages, "reviewer")


class OptimizerNode(BaseNode):
    """
    节点三：测试用例优化器

    负责根据评审反馈优化测试用例。
    为保持一致性，使用与生成器相同的模型。
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        rag_interface: Optional[RAGInterface] = None
    ):
        super().__init__(config or settings.get_optimizer_config(), rag_interface)

    def invoke(
        self,
        original_input: str,
        initial_test_cases: str,
        review_feedback: str,
        output_format: str = "markdown"
    ) -> NodeOutput:
        """
        根据评审反馈优化测试用例。

        参数:
            original_input: 原始用户输入
            initial_test_cases: 生成器的初始测试用例
            review_feedback: 评审员的反馈
            output_format: 期望的输出格式 (markdown/confluence)

        返回:
            NodeOutput 对象（包含内容和截断信息）
        """
        # 获取RAG上下文（如果可用）
        rag_context = self._get_rag_context(original_input)

        # 获取提示词
        system_prompt, user_prompt = PromptTemplates.get_optimizer_prompts(
            original_input=original_input,
            initial_test_cases=initial_test_cases,
            review_feedback=review_feedback,
            output_format=output_format,
            rag_context=rag_context
        )

        # 创建消息并调用LLM
        messages = self._create_messages(system_prompt, user_prompt)
        return self._invoke_llm(messages, "optimizer")


def create_nodes(
    generator_config: Optional[ModelConfig] = None,
    reviewer_config: Optional[ModelConfig] = None,
    optimizer_config: Optional[ModelConfig] = None,
    analyzer_config: Optional[ModelConfig] = None,
    rag_interface: Optional[RAGInterface] = None
) -> tuple[GeneratorNode, ReviewerNode, OptimizerNode, AnalyzerNode]:
    """
    创建所有节点的工厂函数。

    参数:
        generator_config: 生成器节点的配置
        reviewer_config: 评审员节点的配置
        optimizer_config: 优化器节点的配置
        analyzer_config: 分析器节点的配置
        rag_interface: 可选的共享RAG接口

    返回:
        (GeneratorNode, ReviewerNode, OptimizerNode, AnalyzerNode) 元组
    """
    generator = GeneratorNode(generator_config, rag_interface)
    reviewer = ReviewerNode(reviewer_config, rag_interface)
    optimizer = OptimizerNode(optimizer_config, rag_interface)
    analyzer = AnalyzerNode(analyzer_config, rag_interface)

    return generator, reviewer, optimizer, analyzer
