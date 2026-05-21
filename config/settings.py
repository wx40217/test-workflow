"""
测试用例生成器工作流配置模块

本模块包含所有可配置参数，包括：
- API密钥和基础URL
- 各节点的模型配置
- RAG设置（用于未来扩展）
"""

import os
from pathlib import Path
from typing import Any, Optional
from pydantic import Field
from pydantic_settings import BaseSettings

from src.llm.providers import provider_defaults, resolve_provider_defaults

# 自动查找并加载.env文件
# 优先级：当前目录 > 项目根目录
def _find_env_file() -> str:
    """查找.env文件路径。"""
    # 当前工作目录
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        return str(cwd_env)
    
    # 项目根目录（config目录的父目录）
    project_root = Path(__file__).parent.parent
    root_env = project_root / ".env"
    if root_env.exists():
        return str(root_env)
    
    # 默认返回当前目录（即使不存在也不会报错）
    return ".env"


ENV_FILE_PATH = _find_env_file()


class ModelConfig:
    """
    单个LLM节点的模型配置。
    
    属性:
        provider: 模型提供商预设名称
        api_key: 模型提供商的API密钥
        base_url: API端点的基础URL
        model_name: 使用的模型名称
        use_responses_api: 是否使用 OpenAI Responses API
        test_case_split_mode: 测试用例分离模式（mixed/frontend_backend）
        test_case_split_strict: 分离模式校验是否严格执行
        temperature: 采样温度 (0.0-2.0)
        max_tokens: 响应的最大token数
        timeout: 请求超时时间（秒）
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "",
        model_name: str = "",
        provider: str = "openai",
        use_responses_api: Optional[bool] = None,
        test_case_split_mode: str = "mixed",
        test_case_split_strict: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 120,
        reasoning_effort: Optional[str] = None,
        extra_params: Optional[dict[str, Any]] = None,
        supports_tools: Optional[bool] = None,
        supports_vision: Optional[bool] = None,
        thinking: Optional[bool] = None,
    ):
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.use_responses_api = use_responses_api
        self.test_case_split_mode = test_case_split_mode
        self.test_case_split_strict = test_case_split_strict
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self.extra_params = extra_params or {}
        self.supports_tools = supports_tools
        self.supports_vision = supports_vision
        self.thinking = thinking
        resolve_provider_defaults(self)

    def to_dict(self) -> dict:
        """将配置转换为字典。"""
        return {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model_name": self.model_name,
            "provider": self.provider,
            "use_responses_api": self.use_responses_api,
            "test_case_split_mode": self.test_case_split_mode,
            "test_case_split_strict": self.test_case_split_strict,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "reasoning_effort": self.reasoning_effort,
            "extra_params": self.extra_params,
            "supports_tools": self.supports_tools,
            "supports_vision": self.supports_vision,
            "thinking": self.thinking,
        }


class Settings(BaseSettings):
    """
    应用程序全局设置。
    
    设置可通过以下方式配置：
    1. 环境变量
    2. .env文件
    3. 直接赋值
    
    环境变量命名规范：
    - MODEL_PROVIDER, MODEL_API_KEY, MODEL_BASE_URL, MODEL_NAME
    - GENERATOR_API_KEY, GENERATOR_BASE_URL, GENERATOR_MODEL_NAME
    - REVIEWER_API_KEY, REVIEWER_BASE_URL, REVIEWER_MODEL_NAME
    - OPTIMIZER_API_KEY, OPTIMIZER_BASE_URL, OPTIMIZER_MODEL_NAME
    """

    # ============================================
    # 全局模型 provider 预设
    # ============================================
    model_provider: str = Field(
        default="openai",
        description="模型提供商：openai、deepseek、openai-compatible 或 anthropic"
    )
    model_api_key: str = Field(
        default="",
        description="全局模型 API 密钥"
    )
    model_base_url: str = Field(
        default="",
        description="全局模型 API 基础 URL"
    )
    model_name: str = Field(
        default="",
        description="全局模型名称"
    )
    
    # ============================================
    # 节点一：生成器（测试用例生成）
    # ============================================
    generator_api_key: str = Field(
        default="",
        description="生成器模型的API密钥"
    )
    generator_base_url: str = Field(
        default="",
        description="生成器模型API的基础URL"
    )
    generator_model_name: str = Field(
        default="gpt-4o",
        description="用于测试用例生成的模型名称"
    )
    generator_temperature: float = Field(
        default=0.7,
        description="生成器模型的温度（越高越有创造性）"
    )
    generator_max_tokens: int = Field(
        default=8192,
        description="生成器响应的最大token数"
    )
    generator_reasoning_effort: Optional[str] = Field(
        default=None,
        description="生成器推理力度（仅 reasoning 模型生效，如 gpt-5.4, o1 系列）"
    )

    # ============================================
    # 节点二：评审员（测试用例评审）
    # 使用更强大的思考模型
    # ============================================
    reviewer_api_key: str = Field(
        default="",
        description="评审员模型的API密钥"
    )
    reviewer_base_url: str = Field(
        default="",
        description="评审员模型API的基础URL"
    )
    reviewer_model_name: str = Field(
        default="o1-preview",
        description="用于测试用例评审的模型名称（应使用推理模型）"
    )
    reviewer_temperature: float = Field(
        default=1.0,
        description="评审员模型的温度"
    )
    reviewer_max_tokens: int = Field(
        default=8192,
        description="评审员响应的最大token数"
    )
    reviewer_reasoning_effort: Optional[str] = Field(
        default=None,
        description="评审员推理力度（仅 reasoning 模型生效，如 gpt-5.4, o1 系列）"
    )

    # ============================================
    # 节点三：优化器（测试用例优化）
    # 与生成器使用相同模型
    # ============================================
    optimizer_api_key: str = Field(
        default="",
        description="优化器模型的API密钥"
    )
    optimizer_base_url: str = Field(
        default="",
        description="优化器模型API的基础URL"
    )
    optimizer_model_name: str = Field(
        default="gpt-4o",
        description="用于测试用例优化的模型名称"
    )
    optimizer_temperature: float = Field(
        default=0.5,
        description="优化器模型的温度（较低以获得更精确的输出）"
    )
    optimizer_max_tokens: int = Field(
        default=8192,
        description="优化器响应的最大token数"
    )
    optimizer_reasoning_effort: Optional[str] = Field(
        default=None,
        description="优化器推理力度（仅 reasoning 模型生效，如 gpt-5.4, o1 系列）"
    )

    # ============================================
    # RAG配置（用于未来扩展）
    # ============================================
    rag_enabled: bool = Field(
        default=False,
        description="启用RAG功能"
    )
    rag_collection_name: str = Field(
        default="test_case_knowledge",
        description="向量存储集合名称"
    )
    rag_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="RAG使用的嵌入模型"
    )
    rag_top_k: int = Field(
        default=5,
        description="检索的文档数量"
    )
    
    # ============================================
    # 工作流配置
    # ============================================
    enable_analyzer: bool = Field(
        default=False,
        description="是否启用需求分析节点"
    )
    analyzer_complexity_threshold: int = Field(
        default=2,
        description="触发需求分析的复杂度阈值（满足几个条件时启用）"
    )
    max_review_rounds: int = Field(
        default=1,
        description="最大评审轮次"
    )
    max_agent_rounds: int = Field(
        default=2,
        description="multi-agent-quality-graph 最大修订轮次"
    )
    quality_threshold: float = Field(
        default=0.75,
        description="quality-graph / multi-agent-quality-graph 模式的质量通过阈值"
    )
    candidate_pool_size: int = Field(
        default=5,
        description="multi-agent-quality-graph 候选池最大保留数量"
    )
    stop_on_no_improvement_rounds: int = Field(
        default=2,
        description="multi-agent-quality-graph 连续无质量提升时提前停止的轮数"
    )
    show_agent_trace: bool = Field(
        default=False,
        description="是否在详细输出中展示质量闭环 trace"
    )
    agent_mode: str = Field(
        default="workflow",
        description="Agent执行模式：workflow（默认线性工作流）、react、quality-graph 或 multi-agent-quality-graph"
    )
    max_agent_steps: int = Field(
        default=10,
        description="ReAct Agent最大工具调用步数"
    )
    # 预留：大需求拆分功能
    enable_requirement_split: bool = Field(
        default=False,
        description="是否启用大需求拆分（预留功能，暂未实现）"
    )
    requirement_split_threshold: int = Field(
        default=1000,
        description="触发需求拆分的字符数阈值（预留功能）"
    )

    # ============================================
    # 节点零：需求分析器（可选）
    # 未配置时默认使用生成器的配置
    # ============================================
    analyzer_api_key: str = Field(
        default="",
        description="分析器模型的API密钥"
    )
    analyzer_base_url: str = Field(
        default="",
        description="分析器模型API的基础URL"
    )
    analyzer_model_name: str = Field(
        default="gpt-4o",
        description="用于需求分析的模型名称"
    )
    analyzer_temperature: float = Field(
        default=0.3,
        description="分析器模型的温度（较低以获得更精确的分析）"
    )
    analyzer_max_tokens: int = Field(
        default=4096,
        description="分析器响应的最大token数"
    )
    analyzer_reasoning_effort: Optional[str] = Field(
        default=None,
        description="分析器推理力度（仅 reasoning 模型生效，如 gpt-5.4, o1 系列）"
    )

    # ============================================
    # 通用设置
    # ============================================
    test_case_split_mode: str = Field(
        default="mixed",
        description="测试用例分离模式：mixed（默认）或 frontend_backend（前后端分离）"
    )
    test_case_split_strict: bool = Field(
        default=True,
        description="前后端分离模式下是否严格校验结构并自动修复"
    )
    use_responses_api: bool = Field(
        default=True,
        description="是否使用 OpenAI Responses API（仅 openai provider 默认启用）"
    )
    model_supports_tools: Optional[bool] = Field(
        default=None,
        description="当前模型是否支持 tool calling；未设置时由 provider 预设推断"
    )
    request_timeout: int = Field(
        default=120,
        description="默认请求超时时间（秒）"
    )
    max_retries: int = Field(
        default=3,
        description="API调用的最大重试次数"
    )
    
    model_config = {
        "env_file": ENV_FILE_PATH,
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "env_ignore_empty": True,
    }

    def _provider(self) -> str:
        return self.model_provider

    def _provider_default(self):
        return provider_defaults(self._provider())

    def _global_api_key(self) -> str:
        return self.model_api_key or os.getenv("OPENAI_API_KEY", "")

    def _global_base_url(self) -> str:
        return self.model_base_url or self._provider_default().default_base_url or ""

    def _global_model_name(self) -> str:
        return self.model_name or self._provider_default().default_model or self.generator_model_name

    def _explicit_value(self, field_name: str) -> Any:
        return getattr(self, field_name) if field_name in self.model_fields_set else None

    def _use_responses_api(self) -> bool:
        if "use_responses_api" in self.model_fields_set:
            return self.use_responses_api
        return self._provider_default().default_use_responses_api

    def _node_config(
        self,
        *,
        api_key_field: str,
        base_url_field: str,
        model_name_field: str,
        temperature: float,
        max_tokens: int,
        reasoning_effort_field: str,
        default_model_name: str,
    ) -> ModelConfig:
        node_api_key = self._explicit_value(api_key_field) or ""
        node_base_url = self._explicit_value(base_url_field) or ""
        node_model_name = self._explicit_value(model_name_field) or ""
        node_reasoning_effort = self._explicit_value(reasoning_effort_field)
        return ModelConfig(
            provider=self._provider(),
            api_key=node_api_key or self._global_api_key(),
            base_url=node_base_url or self._global_base_url(),
            model_name=node_model_name or self._global_model_name() or default_model_name,
            use_responses_api=self._use_responses_api(),
            test_case_split_mode=self.test_case_split_mode,
            test_case_split_strict=self.test_case_split_strict,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self.request_timeout,
            reasoning_effort=node_reasoning_effort,
            supports_tools=self.model_supports_tools,
        )
    
    def get_generator_config(self) -> ModelConfig:
        """获取生成器节点的配置。"""
        return self._node_config(
            api_key_field="generator_api_key",
            base_url_field="generator_base_url",
            model_name_field="generator_model_name",
            temperature=self.generator_temperature,
            max_tokens=self.generator_max_tokens,
            reasoning_effort_field="generator_reasoning_effort",
            default_model_name=self.generator_model_name,
        )
    
    def get_reviewer_config(self) -> ModelConfig:
        """获取评审员节点的配置。"""
        return self._node_config(
            api_key_field="reviewer_api_key",
            base_url_field="reviewer_base_url",
            model_name_field="reviewer_model_name",
            temperature=self.reviewer_temperature,
            max_tokens=self.reviewer_max_tokens,
            reasoning_effort_field="reviewer_reasoning_effort",
            default_model_name=self.reviewer_model_name,
        )
    
    def get_optimizer_config(self) -> ModelConfig:
        """获取优化器节点的配置。"""
        return self._node_config(
            api_key_field="optimizer_api_key",
            base_url_field="optimizer_base_url",
            model_name_field="optimizer_model_name",
            temperature=self.optimizer_temperature,
            max_tokens=self.optimizer_max_tokens,
            reasoning_effort_field="optimizer_reasoning_effort",
            default_model_name=self.optimizer_model_name,
        )

    def get_analyzer_config(self) -> ModelConfig:
        """获取分析器节点的配置。未配置时使用生成器配置。"""
        return self._node_config(
            api_key_field="analyzer_api_key",
            base_url_field="analyzer_base_url",
            model_name_field="analyzer_model_name",
            temperature=self.analyzer_temperature,
            max_tokens=self.analyzer_max_tokens,
            reasoning_effort_field="analyzer_reasoning_effort",
            default_model_name=self.generator_model_name,
        )

    def use_same_key_for_all(self, api_key: str, base_url: Optional[str] = None):
        """
        便捷方法：为所有节点使用相同的API密钥（和可选的基础URL）。
        """
        self.generator_api_key = api_key
        self.reviewer_api_key = api_key
        self.optimizer_api_key = api_key
        
        if base_url:
            self.generator_base_url = base_url
            self.reviewer_base_url = base_url
            self.optimizer_base_url = base_url


# 全局设置实例
settings = Settings()


def configure_from_env():
    """
    从环境变量配置设置。
    
    此函数在模块导入时自动调用，
    但可以再次调用以刷新环境变量中的设置。
    """
    global settings
    settings = Settings()
    
    # 如果优化器设置未显式设置，使用生成器设置
    if not settings.optimizer_api_key and settings.generator_api_key:
        settings.optimizer_api_key = settings.generator_api_key
    if not settings.optimizer_base_url and settings.generator_base_url:
        settings.optimizer_base_url = settings.generator_base_url
    if settings.optimizer_model_name == "gpt-4o" and settings.generator_model_name != "gpt-4o":
        settings.optimizer_model_name = settings.generator_model_name
    
    return settings


# 导入时自动配置
configure_from_env()
