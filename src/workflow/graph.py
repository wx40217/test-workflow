"""
LangGraph workflow definition for test case generation.

This module defines the complete workflow graph that orchestrates
the three nodes: Generator -> Reviewer -> Optimizer
"""

from dataclasses import dataclass, field
from typing import Any, Optional, TypedDict, Annotated
from operator import add

from langgraph.graph import StateGraph, END

import sys
sys.path.insert(0, '/workspace')

from config.settings import ModelConfig, settings
from src.workflow.nodes import GeneratorNode, ReviewerNode, OptimizerNode, create_nodes
from src.input_handler.handlers import InputHandler, ProcessedInput, MultiInput
from src.output_formatter.formatters import OutputFormatter, OutputFormat
from src.rag.interface import RAGInterface


class WorkflowState(TypedDict):
    """
    State definition for the workflow.
    
    This state is passed between nodes and updated at each step.
    """
    # Input data
    user_input: str
    additional_instructions: str
    images: list[dict]
    output_format: str
    
    # Intermediate results
    generated_test_cases: str
    review_feedback: str
    
    # Final output
    final_test_cases: str
    
    # Metadata
    errors: list[str]
    current_step: str


@dataclass
class WorkflowResult:
    """
    Container for workflow execution results.
    
    Attributes:
        success: Whether the workflow completed successfully
        final_test_cases: The final optimized test cases
        generated_test_cases: Initial test cases (before review)
        review_feedback: Feedback from the reviewer
        errors: Any errors that occurred
        metadata: Additional metadata about the execution
    """
    success: bool
    final_test_cases: str
    generated_test_cases: str = ""
    review_feedback: str = ""
    errors: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class TestCaseWorkflow:
    """
    Main workflow class for test case generation.
    
    Orchestrates the three-node workflow:
    1. Generator: Creates initial test cases
    2. Reviewer: Reviews and provides feedback
    3. Optimizer: Creates final optimized test cases
    
    Usage:
        workflow = TestCaseWorkflow()
        result = workflow.run("User requirements here")
        print(result.final_test_cases)
    """
    
    def __init__(
        self,
        generator_config: Optional[ModelConfig] = None,
        reviewer_config: Optional[ModelConfig] = None,
        optimizer_config: Optional[ModelConfig] = None,
        rag_interface: Optional[RAGInterface] = None,
        output_format: str = "markdown"
    ):
        """
        Initialize the workflow.
        
        Args:
            generator_config: Configuration for the generator node
            reviewer_config: Configuration for the reviewer node
            optimizer_config: Configuration for the optimizer node
            rag_interface: Optional RAG interface for knowledge retrieval
            output_format: Default output format (markdown/confluence)
        """
        # Create nodes
        self.generator, self.reviewer, self.optimizer = create_nodes(
            generator_config=generator_config,
            reviewer_config=reviewer_config,
            optimizer_config=optimizer_config,
            rag_interface=rag_interface
        )
        
        self.input_handler = InputHandler()
        self.output_formatter = OutputFormatter()
        self.default_output_format = output_format
        self.rag_interface = rag_interface
        
        # Build the workflow graph
        self._graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.
        
        Returns:
            Compiled StateGraph
        """
        # Create the graph with state schema
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("review", self._review_node)
        workflow.add_node("optimize", self._optimize_node)
        
        # Define edges (linear flow: generate -> review -> optimize -> end)
        workflow.set_entry_point("generate")
        workflow.add_edge("generate", "review")
        workflow.add_edge("review", "optimize")
        workflow.add_edge("optimize", END)
        
        # Compile the graph
        return workflow.compile()
    
    def _generate_node(self, state: WorkflowState) -> dict:
        """
        Generator node function.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state fields
        """
        try:
            test_cases = self.generator.invoke(
                user_input=state["user_input"],
                additional_instructions=state.get("additional_instructions", ""),
                images=state.get("images")
            )
            return {
                "generated_test_cases": test_cases,
                "current_step": "generate_complete"
            }
        except Exception as e:
            return {
                "generated_test_cases": "",
                "errors": state.get("errors", []) + [f"Generator error: {str(e)}"],
                "current_step": "generate_error"
            }
    
    def _review_node(self, state: WorkflowState) -> dict:
        """
        Reviewer node function.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state fields
        """
        # Skip review if generation failed
        if not state.get("generated_test_cases"):
            return {
                "review_feedback": "Skipped due to generation failure",
                "current_step": "review_skipped"
            }
        
        try:
            feedback = self.reviewer.invoke(
                original_input=state["user_input"],
                test_cases=state["generated_test_cases"]
            )
            return {
                "review_feedback": feedback,
                "current_step": "review_complete"
            }
        except Exception as e:
            return {
                "review_feedback": "",
                "errors": state.get("errors", []) + [f"Reviewer error: {str(e)}"],
                "current_step": "review_error"
            }
    
    def _optimize_node(self, state: WorkflowState) -> dict:
        """
        Optimizer node function.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state fields
        """
        # If both generation and review failed, return empty
        if not state.get("generated_test_cases"):
            return {
                "final_test_cases": "",
                "current_step": "optimize_skipped"
            }
        
        # If review failed but generation succeeded, use generated test cases as final
        if not state.get("review_feedback"):
            return {
                "final_test_cases": state["generated_test_cases"],
                "current_step": "optimize_skipped_using_generated"
            }
        
        try:
            final_cases = self.optimizer.invoke(
                original_input=state["user_input"],
                initial_test_cases=state["generated_test_cases"],
                review_feedback=state["review_feedback"],
                output_format=state.get("output_format", "markdown")
            )
            return {
                "final_test_cases": final_cases,
                "current_step": "optimize_complete"
            }
        except Exception as e:
            # Fall back to generated test cases on error
            return {
                "final_test_cases": state["generated_test_cases"],
                "errors": state.get("errors", []) + [f"Optimizer error: {str(e)}"],
                "current_step": "optimize_error"
            }
    
    def _prepare_input(
        self,
        input_source: Any,
        additional_instructions: str = ""
    ) -> tuple[str, list[dict]]:
        """
        Prepare input from various sources.
        
        Args:
            input_source: Text, file path, list of inputs, or ProcessedInput
            additional_instructions: Additional instructions
            
        Returns:
            Tuple of (text_content, images)
        """
        if isinstance(input_source, ProcessedInput):
            return input_source.text_content, input_source.images
        
        if isinstance(input_source, MultiInput):
            return input_source.get_combined_text(), input_source.get_all_images()
        
        if isinstance(input_source, str):
            # Try to process as file/directory first
            processed = self.input_handler.process(input_source)
            if isinstance(processed, ProcessedInput):
                return processed.text_content, processed.images
            else:
                return processed.get_combined_text(), processed.get_all_images()
        
        if isinstance(input_source, list):
            processed = self.input_handler.process_multiple(input_source)
            return processed.get_combined_text(), processed.get_all_images()
        
        # Default: treat as text
        return str(input_source), []
    
    def run(
        self,
        input_source: Any,
        additional_instructions: str = "",
        output_format: Optional[str] = None
    ) -> WorkflowResult:
        """
        Run the complete workflow.
        
        Args:
            input_source: User input (text, file path, list of inputs, etc.)
            additional_instructions: Additional instructions for test case generation
            output_format: Output format (markdown/confluence), uses default if not specified
            
        Returns:
            WorkflowResult containing the final test cases and metadata
        """
        # Prepare input
        text_content, images = self._prepare_input(input_source, additional_instructions)
        
        # Initialize state
        initial_state: WorkflowState = {
            "user_input": text_content,
            "additional_instructions": additional_instructions,
            "images": images,
            "output_format": output_format or self.default_output_format,
            "generated_test_cases": "",
            "review_feedback": "",
            "final_test_cases": "",
            "errors": [],
            "current_step": "start"
        }
        
        # Run the workflow
        try:
            final_state = self._graph.invoke(initial_state)
            
            # Build result
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
                errors=[f"Workflow execution error: {str(e)}"]
            )
    
    def run_step_by_step(
        self,
        input_source: Any,
        additional_instructions: str = "",
        output_format: Optional[str] = None
    ):
        """
        Run the workflow step by step, yielding results after each step.
        
        This is useful for streaming or progress tracking.
        
        Args:
            input_source: User input
            additional_instructions: Additional instructions
            output_format: Output format
            
        Yields:
            Tuple of (step_name, step_result)
        """
        # Prepare input
        text_content, images = self._prepare_input(input_source, additional_instructions)
        
        # Step 1: Generate
        yield ("generating", None)
        try:
            generated = self.generator.invoke(
                user_input=text_content,
                additional_instructions=additional_instructions,
                images=images
            )
            yield ("generated", generated)
        except Exception as e:
            yield ("generate_error", str(e))
            return
        
        # Step 2: Review
        yield ("reviewing", None)
        try:
            feedback = self.reviewer.invoke(
                original_input=text_content,
                test_cases=generated
            )
            yield ("reviewed", feedback)
        except Exception as e:
            yield ("review_error", str(e))
            # Continue with generated test cases
            feedback = ""
        
        # Step 3: Optimize
        yield ("optimizing", None)
        try:
            if feedback:
                final = self.optimizer.invoke(
                    original_input=text_content,
                    initial_test_cases=generated,
                    review_feedback=feedback,
                    output_format=output_format or self.default_output_format
                )
            else:
                final = generated
            yield ("completed", final)
        except Exception as e:
            yield ("optimize_error", str(e))
            yield ("completed_with_fallback", generated)
    
    def format_output(
        self,
        test_cases: str,
        format_type: Optional[str] = None
    ) -> str:
        """
        Format the test cases output.
        
        Args:
            test_cases: The test cases to format
            format_type: Output format (markdown/confluence)
            
        Returns:
            Formatted test cases
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
    Factory function to create a workflow with custom configuration.
    
    This is a convenience function for quick setup.
    
    Args:
        api_key: API key (used for all nodes if individual keys not set)
        base_url: Base URL (used for all nodes if individual URLs not set)
        generator_model: Model name for generator node
        reviewer_model: Model name for reviewer node
        optimizer_model: Model name for optimizer node
        output_format: Default output format
        enable_rag: Whether to enable RAG
        rag_config: RAG configuration dictionary
        
    Returns:
        Configured TestCaseWorkflow instance
    """
    # Use provided values or fall back to settings
    gen_key = api_key or settings.generator_api_key
    gen_url = base_url or settings.generator_base_url
    gen_model = generator_model or settings.generator_model_name
    
    rev_key = api_key or settings.reviewer_api_key
    rev_url = base_url or settings.reviewer_base_url
    rev_model = reviewer_model or settings.reviewer_model_name
    
    opt_key = api_key or settings.optimizer_api_key
    opt_url = base_url or settings.optimizer_base_url
    opt_model = optimizer_model or settings.optimizer_model_name
    
    # Create configs
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
    
    # Create RAG interface if enabled
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
