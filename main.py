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

# 添加工作区到路径
sys.path.insert(0, '/workspace')

from config.settings import settings, ModelConfig
from config.prompts import PromptTemplates
from src.workflow.graph import TestCaseWorkflow, create_workflow, WorkflowResult
from src.input_handler.handlers import InputHandler
from src.output_formatter.formatters import OutputFormatter, OutputFormat
from src.rag.interface import RAGInterface, RAGConfig


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
    verbose: bool = False
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
        verbose: 是否打印进度信息
        
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
    
    if verbose:
        print("正在启动测试用例生成工作流...")
        print(f"  生成器模型: {generator_model or settings.generator_model_name}")
        print(f"  评审员模型: {reviewer_model or settings.reviewer_model_name}")
        print(f"  优化器模型: {optimizer_model or settings.optimizer_model_name}")
        print(f"  输出格式: {output_format}")
        print(f"  RAG启用: {enable_rag}")
        print()
    
    # 如果verbose则运行带进度跟踪的工作流
    if verbose:
        for step, result in workflow.run_step_by_step(
            input_content,
            additional_instructions=additional_instructions,
            output_format=output_format
        ):
            if result is None:
                print(f"  [{step}]...")
            elif step.endswith("error"):
                print(f"  [{step}] 错误: {result}")
            else:
                print(f"  [{step}] 完成")
        
        # 获取最终结果
        result = workflow.run(
            input_content,
            additional_instructions=additional_instructions,
            output_format=output_format
        )
    else:
        result = workflow.run(
            input_content,
            additional_instructions=additional_instructions,
            output_format=output_format
        )
    
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
    
    # 额外选项
    parser.add_argument(
        "--instructions",
        type=str,
        default="",
        help="测试用例生成的额外指示"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="打印进度信息"
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
            verbose=args.verbose
        )
        
        # 输出结果
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result.final_test_cases)
            if args.verbose:
                print(f"\n输出已写入: {args.output}")
        else:
            if args.verbose:
                print()
                print("=" * 60)
                print("最终测试用例")
                print("=" * 60)
                print()
            print(result.final_test_cases)
        
        # 打印错误/警告
        if result.errors and args.verbose:
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
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
