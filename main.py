"""
Test Case Generator - Main Entry Point

A LangGraph-based workflow for generating, reviewing, and optimizing test cases.

Usage:
    # Basic usage with text input
    python main.py --input "Your requirements here"
    
    # With file input
    python main.py --file requirements.docx
    
    # With multiple files
    python main.py --files doc1.pdf doc2.docx
    
    # With custom output format
    python main.py --input "..." --format confluence
    
    # Programmatic usage
    from main import generate_test_cases
    result = generate_test_cases("Your requirements", api_key="sk-...")
"""

import argparse
import os
import sys

# Add workspace to path
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
    
    Args:
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
    Generate test cases from input content.
    
    This is the main programmatic interface for the test case generator.
    
    Args:
        input_content: The input content (text, file path, or list of paths)
        api_key: API key for LLM (uses env var if not provided)
        base_url: Base URL for LLM API (uses env var if not provided)
        generator_model: Model for test case generation
        reviewer_model: Model for test case review
        optimizer_model: Model for test case optimization
        output_format: Output format (markdown/confluence)
        additional_instructions: Additional instructions for generation
        enable_rag: Whether to enable RAG
        rag_documents: Documents to add to RAG knowledge base
        verbose: Whether to print progress information
        
    Returns:
        WorkflowResult containing the final test cases
        
    Example:
        result = generate_test_cases(
            "Login feature requirements...",
            api_key="sk-...",
            output_format="markdown"
        )
        print(result.final_test_cases)
    """
    # Use provided API key or fall back to settings
    api_key = api_key or settings.generator_api_key or os.getenv("OPENAI_API_KEY")
    base_url = base_url or settings.generator_base_url
    
    if not api_key:
        raise ValueError(
            "API key is required. Provide via api_key parameter, "
            "GENERATOR_API_KEY environment variable, or OPENAI_API_KEY."
        )
    
    # Create RAG interface if enabled
    rag_interface = None
    if enable_rag:
        rag_config = RAGConfig(
            enabled=True,
            embedding_api_key=api_key,
            embedding_base_url=base_url
        )
        rag_interface = RAGInterface(rag_config)
        
        # Add documents to RAG if provided
        if rag_documents:
            for doc in rag_documents:
                if os.path.exists(doc):
                    rag_interface.add_from_file(doc)
                else:
                    rag_interface.add_documents([doc])
    
    # Create workflow
    workflow = create_workflow(
        api_key=api_key,
        base_url=base_url,
        generator_model=generator_model,
        reviewer_model=reviewer_model,
        optimizer_model=optimizer_model,
        output_format=output_format,
        enable_rag=enable_rag
    )
    
    # Attach RAG interface if created
    if rag_interface:
        workflow.rag_interface = rag_interface
        workflow.generator.rag_interface = rag_interface
        workflow.reviewer.rag_interface = rag_interface
        workflow.optimizer.rag_interface = rag_interface
    
    if verbose:
        print("Starting test case generation workflow...")
        print(f"  Generator model: {generator_model or settings.generator_model_name}")
        print(f"  Reviewer model: {reviewer_model or settings.reviewer_model_name}")
        print(f"  Optimizer model: {optimizer_model or settings.optimizer_model_name}")
        print(f"  Output format: {output_format}")
        print(f"  RAG enabled: {enable_rag}")
        print()
    
    # Run workflow with progress tracking if verbose
    if verbose:
        for step, result in workflow.run_step_by_step(
            input_content,
            additional_instructions=additional_instructions,
            output_format=output_format
        ):
            if result is None:
                print(f"  [{step}]...")
            elif step.endswith("error"):
                print(f"  [{step}] Error: {result}")
            else:
                print(f"  [{step}] Done")
        
        # Get final result
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
    Run the test case generator in interactive mode.
    """
    print("=" * 60)
    print("Test Case Generator - Interactive Mode")
    print("=" * 60)
    print()
    
    # Check for API key
    api_key = settings.generator_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Enter your API key: ").strip()
        if not api_key:
            print("Error: API key is required")
            return
    
    # Get base URL
    base_url = settings.generator_base_url
    custom_url = input(f"Base URL [{base_url}]: ").strip()
    if custom_url:
        base_url = custom_url
    
    # Get output format
    output_format = input("Output format (markdown/confluence) [markdown]: ").strip()
    if output_format not in ["markdown", "confluence", ""]:
        output_format = "markdown"
    elif output_format == "":
        output_format = "markdown"
    
    print()
    print("Enter your requirements (press Ctrl+D or Ctrl+Z when done):")
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
        print("Error: No input provided")
        return
    
    print()
    print("-" * 40)
    print("Processing...")
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
        print("FINAL TEST CASES")
        print("=" * 60)
        print()
        print(result.final_test_cases)
        
        if result.errors:
            print()
            print("Warnings/Errors:")
            for error in result.errors:
                print(f"  - {error}")
        
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Generate test cases from requirements using LLM workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from text input
  python main.py --input "User login feature: users can login with email and password"
  
  # Generate from file
  python main.py --file requirements.docx
  
  # Generate from multiple files
  python main.py --files doc1.pdf doc2.docx screenshot.png
  
  # Use custom models
  python main.py --input "..." --generator-model gpt-4o --reviewer-model o1-preview
  
  # Output in Confluence format
  python main.py --input "..." --format confluence
  
  # Interactive mode
  python main.py --interactive
"""
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input", "-i",
        type=str,
        help="Direct text input (requirements)"
    )
    input_group.add_argument(
        "--file", "-f",
        type=str,
        help="Path to input file"
    )
    input_group.add_argument(
        "--files",
        nargs="+",
        help="Paths to multiple input files"
    )
    input_group.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    # API configuration
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        help="API base URL"
    )
    
    # Model configuration
    parser.add_argument(
        "--generator-model",
        type=str,
        help="Model for test case generation"
    )
    parser.add_argument(
        "--reviewer-model",
        type=str,
        help="Model for test case review"
    )
    parser.add_argument(
        "--optimizer-model",
        type=str,
        help="Model for test case optimization"
    )
    
    # Output options
    parser.add_argument(
        "--format",
        choices=["markdown", "confluence"],
        default="markdown",
        help="Output format (default: markdown)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path (prints to stdout if not specified)"
    )
    
    # Additional options
    parser.add_argument(
        "--instructions",
        type=str,
        default="",
        help="Additional instructions for test case generation"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress information"
    )
    
    # RAG options
    parser.add_argument(
        "--enable-rag",
        action="store_true",
        help="Enable RAG (Retrieval-Augmented Generation)"
    )
    parser.add_argument(
        "--rag-documents",
        nargs="+",
        help="Documents to add to RAG knowledge base"
    )
    
    # Prompt configuration
    parser.add_argument(
        "--prompts-config",
        type=str,
        help="Path to prompts configuration file (JSON/YAML)"
    )
    
    args = parser.parse_args()
    
    # Load prompts configuration if specified
    load_prompts_config(args.prompts_config)
    
    # Handle interactive mode
    if args.interactive:
        run_interactive()
        return
    
    # Determine input content
    if args.input:
        input_content = args.input
    elif args.file:
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        input_content = args.file
    elif args.files:
        for f in args.files:
            if not os.path.exists(f):
                print(f"Error: File not found: {f}")
                sys.exit(1)
        input_content = args.files
    else:
        parser.print_help()
        print("\nError: Please provide input via --input, --file, --files, or --interactive")
        sys.exit(1)
    
    # Run generation
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
        
        # Output results
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result.final_test_cases)
            if args.verbose:
                print(f"\nOutput written to: {args.output}")
        else:
            if args.verbose:
                print()
                print("=" * 60)
                print("FINAL TEST CASES")
                print("=" * 60)
                print()
            print(result.final_test_cases)
        
        # Print errors/warnings
        if result.errors and args.verbose:
            print("\nWarnings/Errors:")
            for error in result.errors:
                print(f"  - {error}")
        
        # Exit with appropriate code
        sys.exit(0 if result.success else 1)
        
    except ValueError as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
