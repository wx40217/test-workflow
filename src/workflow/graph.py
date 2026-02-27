"""
测试用例生成的LangGraph工作流定义。

本模块定义完整的工作流图，协调三个节点：
生成器 -> 评审员 -> 优化器
"""

from dataclasses import dataclass, field
from typing import Any, Optional, TypedDict, Annotated
from operator import add

from langgraph.graph import StateGraph, END

import sys
sys.path.insert(0, '/workspace')

from config.settings import ModelConfig, settings
from src.workflow.nodes import GeneratorNode, ReviewerNode, OptimizerNode, AnalyzerNode, create_nodes
from src.input_handler.handlers import InputHandler, ProcessedInput, MultiInput
from src.output_formatter.formatters import OutputFormatter, OutputFormat
from src.rag.interface import RAGInterface


class WorkflowState(TypedDict):
    """
    工作流的状态定义。

    此状态在节点之间传递，每一步都会更新。
    """
    # 输入数据
    user_input: str
    additional_instructions: str
    images: list[dict]
    output_format: str

    # 中间结果
    analysis_result: str
    generated_test_cases: str
    review_feedback: str

    # 最终输出
    final_test_cases: str

    # 元数据
    errors: list[str]
    current_step: str


@dataclass
class WorkflowResult:
    """
    工作流执行结果的容器。
    
    属性:
        success: 工作流是否成功完成
        final_test_cases: 最终优化的测试用例
        generated_test_cases: 初始测试用例（评审前）
        review_feedback: 评审员的反馈
        errors: 发生的任何错误
        metadata: 执行的额外元数据
    """
    success: bool
    final_test_cases: str
    generated_test_cases: str = ""
    review_feedback: str = ""
    errors: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class TestCaseWorkflow:
    """
    测试用例生成的主工作流类。

    协调工作流节点：
    0. 分析器（可选）：分析需求复杂度
    1. 生成器：创建初始测试用例
    2. 评审员：评审并提供反馈
    3. 优化器：创建最终优化的测试用例

    使用方式:
        workflow = TestCaseWorkflow()
        result = workflow.run("用户需求内容")
        print(result.final_test_cases)
    """

    def __init__(
        self,
        generator_config: Optional[ModelConfig] = None,
        reviewer_config: Optional[ModelConfig] = None,
        optimizer_config: Optional[ModelConfig] = None,
        analyzer_config: Optional[ModelConfig] = None,
        rag_interface: Optional[RAGInterface] = None,
        output_format: str = "markdown",
        enable_analyzer: Optional[bool] = None,
        analyzer_complexity_threshold: Optional[int] = None
    ):
        """
        初始化工作流。

        参数:
            generator_config: 生成器节点的配置
            reviewer_config: 评审员节点的配置
            optimizer_config: 优化器节点的配置
            analyzer_config: 分析器节点的配置
            rag_interface: 可选的RAG接口用于知识检索
            output_format: 默认输出格式 (markdown/confluence)
            enable_analyzer: 是否启用需求分析节点（None时使用settings配置）
            analyzer_complexity_threshold: 分析器复杂度阈值（None时使用settings配置）
        """
        # 创建节点
        self.generator, self.reviewer, self.optimizer, self.analyzer = create_nodes(
            generator_config=generator_config,
            reviewer_config=reviewer_config,
            optimizer_config=optimizer_config,
            analyzer_config=analyzer_config,
            rag_interface=rag_interface
        )

        self.input_handler = InputHandler()
        self.output_formatter = OutputFormatter()
        self.default_output_format = output_format
        self.rag_interface = rag_interface

        # 工作流配置
        self.enable_analyzer = enable_analyzer if enable_analyzer is not None else settings.enable_analyzer
        self.analyzer_complexity_threshold = analyzer_complexity_threshold if analyzer_complexity_threshold is not None else settings.analyzer_complexity_threshold

        # 构建工作流图
        self._graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """
        构建LangGraph工作流。

        返回:
            编译后的StateGraph
        """
        # 使用状态模式创建图
        workflow = StateGraph(WorkflowState)

        # 添加节点
        workflow.add_node("analyze", self._analyze_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("review", self._review_node)
        workflow.add_node("optimize", self._optimize_node)

        # 定义条件路由
        workflow.set_entry_point("analyze")

        # 分析节点后根据结果决定下一步
        workflow.add_edge("analyze", "generate")
        workflow.add_edge("generate", "review")
        workflow.add_edge("review", "optimize")
        workflow.add_edge("optimize", END)

        # 编译图
        return workflow.compile()

    def _analyze_node(self, state: WorkflowState) -> dict:
        """
        需求分析节点函数（条件性执行）。

        参数:
            state: 当前工作流状态

        返回:
            更新的状态字段
        """
        user_input = state["user_input"]

        # 判断是否需要分析
        if not self.enable_analyzer:
            return {
                "analysis_result": "",
                "current_step": "analyze_skipped_disabled"
            }

        if not AnalyzerNode.should_analyze(user_input, self.analyzer_complexity_threshold):
            return {
                "analysis_result": "",
                "current_step": "analyze_skipped_simple"
            }

        try:
            output = self.analyzer.invoke(
                user_input=user_input,
                additional_instructions=state.get("additional_instructions", "")
            )

            errors = state.get("errors", [])
            if output.is_truncated:
                errors = errors + [output.truncation_warning]

            return {
                "analysis_result": output.content,
                "errors": errors,
                "current_step": "analyze_complete"
            }
        except Exception as e:
            return {
                "analysis_result": "",
                "errors": state.get("errors", []) + [f"分析器错误: {str(e)}"],
                "current_step": "analyze_error"
            }

    def _generate_node(self, state: WorkflowState) -> dict:
        """
        生成器节点函数。

        参数:
            state: 当前工作流状态

        返回:
            更新的状态字段
        """
        try:
            output = self.generator.invoke(
                user_input=state["user_input"],
                additional_instructions=state.get("additional_instructions", ""),
                images=state.get("images"),
                analysis_result=state.get("analysis_result", "")
            )

            errors = state.get("errors", [])
            if output.is_truncated:
                errors = errors + [output.truncation_warning]

            return {
                "generated_test_cases": output.content,
                "errors": errors,
                "current_step": "generate_complete"
            }
        except Exception as e:
            return {
                "generated_test_cases": "",
                "errors": state.get("errors", []) + [f"生成器错误: {str(e)}"],
                "current_step": "generate_error"
            }
    
    def _review_node(self, state: WorkflowState) -> dict:
        """
        评审员节点函数。

        参数:
            state: 当前工作流状态

        返回:
            更新的状态字段
        """
        # 如果生成失败则跳过评审
        if not state.get("generated_test_cases"):
            return {
                "review_feedback": "由于生成失败而跳过",
                "current_step": "review_skipped"
            }

        try:
            output = self.reviewer.invoke(
                original_input=state["user_input"],
                test_cases=state["generated_test_cases"]
            )

            errors = state.get("errors", [])
            if output.is_truncated:
                errors = errors + [output.truncation_warning]

            return {
                "review_feedback": output.content,
                "errors": errors,
                "current_step": "review_complete"
            }
        except Exception as e:
            return {
                "review_feedback": "",
                "errors": state.get("errors", []) + [f"评审员错误: {str(e)}"],
                "current_step": "review_error"
            }

    def _optimize_node(self, state: WorkflowState) -> dict:
        """
        优化器节点函数。

        参数:
            state: 当前工作流状态

        返回:
            更新的状态字段
        """
        # 如果生成和评审都失败，返回空
        if not state.get("generated_test_cases"):
            return {
                "final_test_cases": "",
                "current_step": "optimize_skipped"
            }

        # 如果评审失败但生成成功，使用生成的测试用例作为最终结果
        if not state.get("review_feedback"):
            return {
                "final_test_cases": state["generated_test_cases"],
                "current_step": "optimize_skipped_using_generated"
            }

        try:
            output = self.optimizer.invoke(
                original_input=state["user_input"],
                initial_test_cases=state["generated_test_cases"],
                review_feedback=state["review_feedback"],
                output_format=state.get("output_format", "markdown")
            )

            errors = state.get("errors", [])
            if output.is_truncated:
                errors = errors + [output.truncation_warning]

            return {
                "final_test_cases": output.content,
                "errors": errors,
                "current_step": "optimize_complete"
            }
        except Exception as e:
            # 出错时回退到生成的测试用例
            return {
                "final_test_cases": state["generated_test_cases"],
                "errors": state.get("errors", []) + [f"优化器错误: {str(e)}"],
                "current_step": "optimize_error"
            }
    
    def _prepare_input(
        self,
        input_source: Any,
        additional_instructions: str = ""
    ) -> tuple[str, list[dict]]:
        """
        从各种来源准备输入。
        
        参数:
            input_source: 文本、文件路径、输入列表或ProcessedInput
            additional_instructions: 额外指示
            
        返回:
            (text_content, images) 元组
        """
        if isinstance(input_source, ProcessedInput):
            return input_source.text_content, input_source.images
        
        if isinstance(input_source, MultiInput):
            return input_source.get_combined_text(), input_source.get_all_images()
        
        if isinstance(input_source, str):
            # 首先尝试作为文件/目录处理
            processed = self.input_handler.process(input_source)
            if isinstance(processed, ProcessedInput):
                return processed.text_content, processed.images
            else:
                return processed.get_combined_text(), processed.get_all_images()
        
        if isinstance(input_source, list):
            processed = self.input_handler.process_multiple(input_source)
            return processed.get_combined_text(), processed.get_all_images()
        
        # 默认：作为文本处理
        return str(input_source), []
    
    def run(
        self,
        input_source: Any,
        additional_instructions: str = "",
        output_format: Optional[str] = None
    ) -> WorkflowResult:
        """
        运行完整工作流。
        
        参数:
            input_source: 用户输入（文本、文件路径、输入列表等）
            additional_instructions: 测试用例生成的额外指示
            output_format: 输出格式 (markdown/confluence)，未指定时使用默认值
            
        返回:
            包含最终测试用例和元数据的WorkflowResult
        """
        # 准备输入
        text_content, images = self._prepare_input(input_source, additional_instructions)
        
        # 初始化状态
        initial_state: WorkflowState = {
            "user_input": text_content,
            "additional_instructions": additional_instructions,
            "images": images,
            "output_format": output_format or self.default_output_format,
            "analysis_result": "",
            "generated_test_cases": "",
            "review_feedback": "",
            "final_test_cases": "",
            "errors": [],
            "current_step": "start"
        }
        
        # 运行工作流
        try:
            final_state = self._graph.invoke(initial_state)
            
            # 构建结果
            success = bool(final_state.get("final_test_cases"))
            return WorkflowResult(
                success=success,
                final_test_cases=final_state.get("final_test_cases", ""),
                generated_test_cases=final_state.get("generated_test_cases", ""),
                review_feedback=final_state.get("review_feedback", ""),
                errors=final_state.get("errors", []),
                metadata={
                    "output_format": final_state.get("output_format"),
                    "current_step": final_state.get("current_step"),
                    "has_images": len(images) > 0
                }
            )
        except Exception as e:
            return WorkflowResult(
                success=False,
                final_test_cases="",
                errors=[f"工作流执行错误: {str(e)}"]
            )
    
    def run_step_by_step(
        self,
        input_source: Any,
        additional_instructions: str = "",
        output_format: Optional[str] = None
    ):
        """
        逐步运行工作流，每一步后产出结果。

        这对于流式处理或进度跟踪很有用。

        参数:
            input_source: 用户输入
            additional_instructions: 额外指示
            output_format: 输出格式

        产出:
            (step_name, step_result) 元组
        """
        # 准备输入
        text_content, images = self._prepare_input(input_source, additional_instructions)

        # 收集截断警告
        truncation_warnings = []
        analysis_result = ""

        # 步骤0：分析（条件执行）
        if not self.enable_analyzer:
            yield ("analyze_skipped_disabled", None)
        elif not AnalyzerNode.should_analyze(text_content, self.analyzer_complexity_threshold):
            yield ("analyze_skipped_simple", None)
        else:
            yield ("analyzing", None)
            try:
                output = self.analyzer.invoke(
                    user_input=text_content,
                    additional_instructions=additional_instructions
                )
                analysis_result = output.content
                if output.is_truncated:
                    truncation_warnings.append(output.truncation_warning)
                yield ("analyzed", analysis_result)
            except Exception as e:
                yield ("analyze_error", str(e))
                # 分析失败时继续后续流程
                analysis_result = ""

        # 步骤1：生成
        yield ("generating", None)
        try:
            output = self.generator.invoke(
                user_input=text_content,
                additional_instructions=additional_instructions,
                images=images,
                analysis_result=analysis_result
            )
            generated = output.content
            if output.is_truncated:
                truncation_warnings.append(output.truncation_warning)
            yield ("generated", generated)
        except Exception as e:
            yield ("generate_error", str(e))
            return

        # 步骤2：评审
        yield ("reviewing", None)
        try:
            output = self.reviewer.invoke(
                original_input=text_content,
                test_cases=generated
            )
            feedback = output.content
            if output.is_truncated:
                truncation_warnings.append(output.truncation_warning)
            yield ("reviewed", feedback)
        except Exception as e:
            yield ("review_error", str(e))
            # 继续使用生成的测试用例
            feedback = ""

        # 步骤3：优化
        yield ("optimizing", None)
        try:
            if feedback:
                output = self.optimizer.invoke(
                    original_input=text_content,
                    initial_test_cases=generated,
                    review_feedback=feedback,
                    output_format=output_format or self.default_output_format
                )
                final = output.content
                if output.is_truncated:
                    truncation_warnings.append(output.truncation_warning)
            else:
                final = generated
            yield ("completed", final)
            # 如果有截断警告，额外yield一次
            if truncation_warnings:
                yield ("truncation_warnings", truncation_warnings)
        except Exception as e:
            yield ("optimize_error", str(e))
            yield ("completed_with_fallback", generated)
    
    def format_output(
        self,
        test_cases: str,
        format_type: Optional[str] = None
    ) -> str:
        """
        格式化测试用例输出。
        
        参数:
            test_cases: 要格式化的测试用例
            format_type: 输出格式 (markdown/confluence)
            
        返回:
            格式化的测试用例
        """
        output_format = OutputFormat(format_type or self.default_output_format)
        return self.output_formatter.format(test_cases, output_format)


def create_workflow(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    generator_model: Optional[str] = None,
    reviewer_model: Optional[str] = None,
    optimizer_model: Optional[str] = None,
    output_format: str = "markdown",
    enable_rag: bool = False,
    rag_config: Optional[dict] = None
) -> TestCaseWorkflow:
    """
    创建自定义配置工作流的工厂函数。
    
    这是一个便捷函数，用于快速设置。
    
    参数:
        api_key: API密钥（如果未设置单独密钥，则用于所有节点）
        base_url: 基础URL（如果未设置单独URL，则用于所有节点）
        generator_model: 生成器节点的模型名称
        reviewer_model: 评审员节点的模型名称
        optimizer_model: 优化器节点的模型名称
        output_format: 默认输出格式
        enable_rag: 是否启用RAG
        rag_config: RAG配置字典
        
    返回:
        配置好的TestCaseWorkflow实例
    """
    # 使用提供的值或回退到设置
    gen_key = api_key or settings.generator_api_key
    gen_url = base_url or settings.generator_base_url
    gen_model = generator_model or settings.generator_model_name
    
    rev_key = api_key or settings.reviewer_api_key
    rev_url = base_url or settings.reviewer_base_url
    rev_model = reviewer_model or settings.reviewer_model_name
    
    opt_key = api_key or settings.optimizer_api_key
    opt_url = base_url or settings.optimizer_base_url
    opt_model = optimizer_model or settings.optimizer_model_name
    
    # 创建配置
    generator_config = ModelConfig(
        api_key=gen_key,
        base_url=gen_url,
        model_name=gen_model,
        temperature=settings.generator_temperature,
        max_tokens=settings.generator_max_tokens,
        timeout=settings.request_timeout
    )
    
    reviewer_config = ModelConfig(
        api_key=rev_key,
        base_url=rev_url,
        model_name=rev_model,
        temperature=settings.reviewer_temperature,
        max_tokens=settings.reviewer_max_tokens,
        timeout=settings.request_timeout
    )
    
    optimizer_config = ModelConfig(
        api_key=opt_key,
        base_url=opt_url,
        model_name=opt_model,
        temperature=settings.optimizer_temperature,
        max_tokens=settings.optimizer_max_tokens,
        timeout=settings.request_timeout
    )
    
    # 如果启用则创建RAG接口
    rag_interface = None
    if enable_rag:
        from src.rag.interface import RAGConfig
        if rag_config:
            rag_cfg = RAGConfig(**rag_config)
        else:
            rag_cfg = RAGConfig(enabled=True)
        rag_interface = RAGInterface(config=rag_cfg)
    
    return TestCaseWorkflow(
        generator_config=generator_config,
        reviewer_config=reviewer_config,
        optimizer_config=optimizer_config,
        rag_interface=rag_interface,
        output_format=output_format
    )
