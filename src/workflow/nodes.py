"""
测试用例生成工作流的LLM节点模块。

本模块定义三个主要节点：
1. GeneratorNode - 根据用户输入生成初始测试用例
2. ReviewerNode - 评审测试用例并提供反馈
3. OptimizerNode - 根据反馈优化测试用例
"""

from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import sys
sys.path.insert(0, '/workspace')

from config.settings import ModelConfig, settings
from config.prompts import PromptTemplates
from src.rag.interface import RAGInterface


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
    
    def _get_llm(self) -> ChatOpenAI:
        """获取或创建LLM实例。"""
        if self._llm is None:
            if self.config is None:
                raise ValueError("需要模型配置")
            
            self._llm = ChatOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
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
        images: Optional[list[dict]] = None
    ) -> str:
        """
        根据用户输入生成测试用例。
        
        参数:
            user_input: 用户输入（需求等）
            additional_instructions: 用户的额外指示
            images: 可选的图片（用于视觉模型）
            
        返回:
            生成的测试用例字符串
        """
        # 获取RAG上下文（如果可用）
        rag_context = self._get_rag_context(user_input)
        
        # 获取提示词
        system_prompt, user_prompt = PromptTemplates.get_generator_prompts(
            user_input=user_input,
            additional_instructions=additional_instructions,
            rag_context=rag_context
        )
        
        # 创建消息并调用LLM
        messages = self._create_messages(system_prompt, user_prompt, images)
        llm = self._get_llm()
        response = llm.invoke(messages)
        
        return response.content


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
    ) -> str:
        """
        评审生成的测试用例。
        
        参数:
            original_input: 原始用户输入
            test_cases: 生成器节点生成的测试用例
            
        返回:
            评审反馈字符串
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
        llm = self._get_llm()
        response = llm.invoke(messages)
        
        return response.content


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
    ) -> str:
        """
        根据评审反馈优化测试用例。
        
        参数:
            original_input: 原始用户输入
            initial_test_cases: 生成器的初始测试用例
            review_feedback: 评审员的反馈
            output_format: 期望的输出格式 (markdown/confluence)
            
        返回:
            优化后的测试用例字符串
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
        llm = self._get_llm()
        response = llm.invoke(messages)
        
        return response.content


def create_nodes(
    generator_config: Optional[ModelConfig] = None,
    reviewer_config: Optional[ModelConfig] = None,
    optimizer_config: Optional[ModelConfig] = None,
    rag_interface: Optional[RAGInterface] = None
) -> tuple[GeneratorNode, ReviewerNode, OptimizerNode]:
    """
    创建所有三个节点的工厂函数。
    
    参数:
        generator_config: 生成器节点的配置
        reviewer_config: 评审员节点的配置
        optimizer_config: 优化器节点的配置
        rag_interface: 可选的共享RAG接口
        
    返回:
        (GeneratorNode, ReviewerNode, OptimizerNode) 元组
    """
    generator = GeneratorNode(generator_config, rag_interface)
    reviewer = ReviewerNode(reviewer_config, rag_interface)
    optimizer = OptimizerNode(optimizer_config, rag_interface)
    
    return generator, reviewer, optimizer
