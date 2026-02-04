"""
测试用例生成器 - 主程序入口

基于LangGraph的工作流，用于生成、评审和优化测试用例。

使用方式:
    # 基本用法 - 文本输入
    python main.py --input "你的需求描述"
    
    # 文件输入
    python main.py --file requirements.docx
    
    # 多文件输入
    python main.py --files doc1.pdf doc2.docx
    
    # 自定义输出格式
    python main.py --input "..." --format confluence
    
    # 编程方式使用
    from main import generate_test_cases
    result = generate_test_cases("你的需求", api_key="sk-...")
"""

import argparse
import os
import sys
from datetime import datetime

# 添加工作区到路径
sys.path.insert(0, '/workspace')

from config.settings import settings, ModelConfig
from config.prompts import PromptTemplates
from src.workflow.graph import TestCaseWorkflow, create_workflow, WorkflowResult
from src.input_handler.handlers import InputHandler
from src.output_formatter.formatters import OutputFormatter, OutputFormat
from src.rag.interface import RAGInterface, RAGConfig


# 项目目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUTS_DIR = os.path.join(PROJECT_ROOT, "inputs")
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "templates")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")


class ProgressPrinter:
    """进度打印器，提供友好的执行进度输出。"""

    STEPS = {
        "analyze": ("分析", "正在分析需求复杂度..."),
        "analyze_skipped": ("分析", "跳过（未启用或需求简单）"),
        "generate": ("生成", "正在生成初始测试用例..."),
        "review": ("评审", "正在评审测试用例..."),
        "optimize": ("优化", "正在优化测试用例..."),
        "complete": ("完成", "测试用例生成完成"),
    }

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.start_time = None

    def start(self):
        """开始计时。"""
        self.start_time = datetime.now()
        if self.enabled:
            print()
            print("=" * 50)
            print("  测试用例生成工作流")
            print("=" * 50)

    def step(self, step_name: str, status: str = "running", detail: str = ""):
        """打印步骤进度。"""
        if not self.enabled:
            return

        step_info = self.STEPS.get(step_name, (step_name, ""))
        label, default_msg = step_info

        if status == "running":
            print(f"\n[{label}] {default_msg}")
        elif status == "done":
            msg = detail if detail else "完成"
            print(f"[{label}] ✓ {msg}")
        elif status == "skip":
            print(f"[{label}] - {detail if detail else '跳过'}")
        elif status == "error":
            print(f"[{label}] ✗ 错误: {detail}")

    def finish(self, success: bool, test_count: int = 0, output_path: str = ""):
        """打印完成信息。"""
        if not self.enabled:
            return

        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0

        print()
        print("=" * 50)
        if success:
            print(f"  ✓ 生成完成 | 耗时: {elapsed:.1f}秒")
            if test_count > 0:
                print(f"  测试点数量: {test_count}")
            if output_path:
                print(f"  输出文件: {output_path}")
        else:
            print(f"  ✗ 生成失败 | 耗时: {elapsed:.1f}秒")
        print("=" * 50)
        print()


def count_test_cases(content: str) -> int:
    """统计测试用例数量（粗略统计加粗的测试点）。"""
    import re
    # 匹配 **xxx** 格式的测试点
    matches = re.findall(r'\*\*[^*]+\*\*', content)
    return len(matches)


def generate_output_filename(input_content: str) -> str:
    """根据输入内容生成输出文件名。"""
    # 提取前20个字符作为文件名的一部分
    clean_name = input_content[:20].strip()
    # 移除不合法的文件名字符
    for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\n', '\r']:
        clean_name = clean_name.replace(char, '_')
    clean_name = clean_name.strip('_') or "test_cases"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{clean_name}.md"


def load_prompts_config(config_file: str = None) -> None:
    """
    加载提示词配置文件。
    
    参数:
        config_file: 配置文件路径，支持JSON/YAML格式
    """
    if config_file is None:
        config_file = os.getenv("PROMPTS_CONFIG_FILE")
    
    if config_file and os.path.exists(config_file):
        try:
            PromptTemplates.load_from_file(config_file)
            print(f"已加载提示词配置: {config_file}")
        except Exception as e:
            print(f"警告: 加载提示词配置失败: {e}")


def generate_test_cases(
    input_content: str,
    api_key: str = None,
    base_url: str = None,
    generator_model: str = None,
    reviewer_model: str = None,
    optimizer_model: str = None,
    output_format: str = "markdown",
    additional_instructions: str = "",
    enable_rag: bool = False,
    rag_documents: list[str] = None,
    verbose: bool = True,
    auto_save: bool = False
) -> WorkflowResult:
    """
    从输入内容生成测试用例。

    这是测试用例生成器的主要编程接口。

    参数:
        input_content: 输入内容（文本、文件路径或路径列表）
        api_key: LLM的API密钥（未提供时使用环境变量）
        base_url: LLM API的基础URL（未提供时使用环境变量）
        generator_model: 用于测试用例生成的模型
        reviewer_model: 用于测试用例评审的模型
        optimizer_model: 用于测试用例优化的模型
        output_format: 输出格式 (markdown/confluence)
        additional_instructions: 生成的额外指示
        enable_rag: 是否启用RAG
        rag_documents: 要添加到RAG知识库的文档
        verbose: 是否打印进度信息（默认True）
        auto_save: 是否自动保存到outputs目录

    返回:
        包含最终测试用例的WorkflowResult

    示例:
        result = generate_test_cases(
            "登录功能需求...",
            api_key="sk-...",
            output_format="markdown"
        )
        print(result.final_test_cases)
    """
    progress = ProgressPrinter(enabled=verbose)
    progress.start()

    # 使用提供的API密钥或回退到设置
    api_key = api_key or settings.generator_api_key or os.getenv("OPENAI_API_KEY")
    base_url = base_url or settings.generator_base_url

    if not api_key:
        raise ValueError(
            "需要API密钥。通过api_key参数、"
            "GENERATOR_API_KEY环境变量或OPENAI_API_KEY提供。"
        )

    # 如果启用则创建RAG接口
    rag_interface = None
    if enable_rag:
        rag_config = RAGConfig(
            enabled=True,
            embedding_api_key=api_key,
            embedding_base_url=base_url
        )
        rag_interface = RAGInterface(rag_config)

        # 如果提供则添加文档到RAG
        if rag_documents:
            for doc in rag_documents:
                if os.path.exists(doc):
                    rag_interface.add_from_file(doc)
                else:
                    rag_interface.add_documents([doc])

    # 创建工作流
    workflow = create_workflow(
        api_key=api_key,
        base_url=base_url,
        generator_model=generator_model,
        reviewer_model=reviewer_model,
        optimizer_model=optimizer_model,
        output_format=output_format,
        enable_rag=enable_rag
    )

    # 如果创建了则附加RAG接口
    if rag_interface:
        workflow.rag_interface = rag_interface
        workflow.generator.rag_interface = rag_interface
        workflow.reviewer.rag_interface = rag_interface
        workflow.optimizer.rag_interface = rag_interface

    # 使用 step-by-step 运行以获取进度
    result = None
    output_path = ""

    for step, step_result in workflow.run_step_by_step(
        input_content,
        additional_instructions=additional_instructions,
        output_format=output_format
    ):
        if step == "generating":
            # 先检查分析节点状态
            if workflow.enable_analyzer:
                progress.step("analyze", "running")
            else:
                progress.step("analyze", "skip", "未启用")
            progress.step("generate", "running")
        elif step == "generated":
            progress.step("generate", "done")
        elif step == "generate_error":
            progress.step("generate", "error", step_result)
        elif step == "reviewing":
            progress.step("review", "running")
        elif step == "reviewed":
            progress.step("review", "done")
        elif step == "review_error":
            progress.step("review", "error", step_result)
        elif step == "optimizing":
            progress.step("optimize", "running")
        elif step == "completed":
            progress.step("optimize", "done")
            # 构建最终结果
            result = WorkflowResult(
                success=True,
                final_test_cases=step_result,
                generated_test_cases="",
                review_feedback="",
                errors=[]
            )
        elif step == "completed_with_fallback":
            progress.step("optimize", "skip", "使用初始版本")
            result = WorkflowResult(
                success=True,
                final_test_cases=step_result,
                generated_test_cases=step_result,
                review_feedback="",
                errors=["优化失败，使用初始版本"]
            )

    # 如果没有通过step-by-step获取到结果，直接运行
    if result is None:
        result = workflow.run(
            input_content,
            additional_instructions=additional_instructions,
            output_format=output_format
        )

    # 自动保存到outputs目录
    if auto_save and result.success and result.final_test_cases:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        filename = generate_output_filename(input_content)
        output_path = os.path.join(OUTPUTS_DIR, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result.final_test_cases)

    # 统计测试点数量
    test_count = count_test_cases(result.final_test_cases) if result.final_test_cases else 0

    progress.finish(result.success, test_count, output_path)

    return result


def run_interactive():
    """
    以交互模式运行测试用例生成器。
    """
    print("=" * 60)
    print("测试用例生成器 - 交互模式")
    print("=" * 60)
    print()
    
    # 检查API密钥
    api_key = settings.generator_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("请输入您的API密钥: ").strip()
        if not api_key:
            print("错误: 需要API密钥")
            return
    
    # 获取基础URL
    base_url = settings.generator_base_url
    custom_url = input(f"基础URL [{base_url}]: ").strip()
    if custom_url:
        base_url = custom_url
    
    # 获取输出格式
    output_format = input("输出格式 (markdown/confluence) [markdown]: ").strip()
    if output_format not in ["markdown", "confluence", ""]:
        output_format = "markdown"
    elif output_format == "":
        output_format = "markdown"
    
    print()
    print("请输入您的需求（完成后按Ctrl+D或Ctrl+Z）:")
    print("-" * 40)
    
    lines = []
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        pass
    
    input_content = "\n".join(lines)
    
    if not input_content.strip():
        print("错误: 未提供输入")
        return
    
    print()
    print("-" * 40)
    print("正在处理...")
    print()
    
    try:
        result = generate_test_cases(
            input_content,
            api_key=api_key,
            base_url=base_url,
            output_format=output_format,
            verbose=True
        )
        
        print()
        print("=" * 60)
        print("最终测试用例")
        print("=" * 60)
        print()
        print(result.final_test_cases)
        
        if result.errors:
            print()
            print("警告/错误:")
            for error in result.errors:
                print(f"  - {error}")
        
    except Exception as e:
        print(f"错误: {e}")


def main():
    """CLI主入口点。"""
    parser = argparse.ArgumentParser(
        description="使用LLM工作流从需求生成测试用例",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从文本输入生成
  python main.py --input "用户登录功能: 用户可以使用邮箱和密码登录"
  
  # 从文件生成
  python main.py --file requirements.docx
  
  # 从多个文件生成
  python main.py --files doc1.pdf doc2.docx screenshot.png
  
  # 使用自定义模型
  python main.py --input "..." --generator-model gpt-4o --reviewer-model o1-preview
  
  # 输出Confluence格式
  python main.py --input "..." --format confluence
  
  # 交互模式
  python main.py --interactive
"""
    )
    
    # 输入选项（互斥）
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input", "-i",
        type=str,
        help="直接文本输入（需求）"
    )
    input_group.add_argument(
        "--file", "-f",
        type=str,
        help="输入文件路径"
    )
    input_group.add_argument(
        "--files",
        nargs="+",
        help="多个输入文件的路径"
    )
    input_group.add_argument(
        "--interactive",
        action="store_true",
        help="以交互模式运行"
    )
    
    # API配置
    parser.add_argument(
        "--api-key",
        type=str,
        help="API密钥（或设置OPENAI_API_KEY环境变量）"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        help="API基础URL"
    )
    
    # 模型配置
    parser.add_argument(
        "--generator-model",
        type=str,
        help="用于测试用例生成的模型"
    )
    parser.add_argument(
        "--reviewer-model",
        type=str,
        help="用于测试用例评审的模型"
    )
    parser.add_argument(
        "--optimizer-model",
        type=str,
        help="用于测试用例优化的模型"
    )
    
    # 输出选项
    parser.add_argument(
        "--format",
        choices=["markdown", "confluence"],
        default="markdown",
        help="输出格式（默认: markdown）"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="输出文件路径（未指定时输出到标准输出）"
    )
    parser.add_argument(
        "--auto-save",
        action="store_true",
        help="自动保存到outputs目录"
    )
    
    # 额外选项
    parser.add_argument(
        "--instructions",
        type=str,
        default="",
        help="测试用例生成的额外指示"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="静默模式，不打印进度信息"
    )
    
    # RAG选项
    parser.add_argument(
        "--enable-rag",
        action="store_true",
        help="启用RAG（检索增强生成）"
    )
    parser.add_argument(
        "--rag-documents",
        nargs="+",
        help="要添加到RAG知识库的文档"
    )
    
    # 提示词配置
    parser.add_argument(
        "--prompts-config",
        type=str,
        help="提示词配置文件路径（JSON/YAML）"
    )
    
    args = parser.parse_args()
    
    # 如果指定则加载提示词配置
    load_prompts_config(args.prompts_config)
    
    # 处理交互模式
    if args.interactive:
        run_interactive()
        return
    
    # 确定输入内容
    if args.input:
        input_content = args.input
    elif args.file:
        if not os.path.exists(args.file):
            print(f"错误: 文件未找到: {args.file}")
            sys.exit(1)
        input_content = args.file
    elif args.files:
        for f in args.files:
            if not os.path.exists(f):
                print(f"错误: 文件未找到: {f}")
                sys.exit(1)
        input_content = args.files
    else:
        parser.print_help()
        print("\n错误: 请通过 --input、--file、--files 或 --interactive 提供输入")
        sys.exit(1)
    
    # 运行生成
    try:
        result = generate_test_cases(
            input_content,
            api_key=args.api_key,
            base_url=args.base_url,
            generator_model=args.generator_model,
            reviewer_model=args.reviewer_model,
            optimizer_model=args.optimizer_model,
            output_format=args.format,
            additional_instructions=args.instructions,
            enable_rag=args.enable_rag,
            rag_documents=args.rag_documents,
            verbose=not args.quiet if hasattr(args, 'quiet') else True,
            auto_save=args.auto_save
        )
        
        # 输出结果
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result.final_test_cases)
            if not args.quiet:
                print(f"输出已写入: {args.output}")
        elif not args.quiet:
            print()
            print("=" * 50)
            print("  最终测试用例")
            print("=" * 50)
            print()
            print(result.final_test_cases)

        # 打印错误/警告
        if result.errors and not args.quiet:
            print("\n警告/错误:")
            for error in result.errors:
                print(f"  - {error}")
        
        # 使用适当的退出码退出
        sys.exit(0 if result.success else 1)
        
    except ValueError as e:
        print(f"配置错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
