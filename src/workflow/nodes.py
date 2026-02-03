"""
LLM nodes for the test case generation workflow.

This module defines three main nodes:
1. GeneratorNode - Generates initial test cases from user input
2. ReviewerNode - Reviews test cases and provides feedback
3. OptimizerNode - Optimizes test cases based on feedback
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
    """Base class for all LLM nodes."""
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        rag_interface: Optional[RAGInterface] = None
    ):
        """
        Initialize the node.
        
        Args:
            config: Model configuration (uses default if not provided)
            rag_interface: Optional RAG interface for knowledge retrieval
        """
        self.config = config
        self.rag_interface = rag_interface
        self._llm = None
    
    def _get_llm(self) -> ChatOpenAI:
        """Get or create the LLM instance."""
        if self._llm is None:
            if self.config is None:
                raise ValueError("Model configuration is required")
            
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
        Retrieve relevant context from RAG if enabled.
        
        Args:
            query: The query to search for
            
        Returns:
            Retrieved context as a formatted string
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
            print(f"Warning: RAG retrieval failed: {e}")
        
        return ""
    
    def _create_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        images: Optional[list[dict]] = None
    ) -> list:
        """
        Create message list for LLM invocation.
        
        Args:
            system_prompt: The system prompt
            user_prompt: The user prompt
            images: Optional list of images (for vision models)
            
        Returns:
            List of messages
        """
        messages = [SystemMessage(content=system_prompt)]
        
        if images:
            # For vision models, include images in the user message
            content = [{"type": "text", "text": user_prompt}]
            for img in images:
                content.append(img)
            messages.append(HumanMessage(content=content))
        else:
            messages.append(HumanMessage(content=user_prompt))
        
        return messages
    
    def invoke(self, **kwargs) -> str:
        """
        Invoke the node. Must be implemented by subclasses.
        
        Returns:
            The LLM response as a string
        """
        raise NotImplementedError("Subclasses must implement invoke()")


class GeneratorNode(BaseNode):
    """
    Node 1: Test Case Generator
    
    Responsible for generating initial test cases from user input.
    Uses the configured generator model (default: same as optimizer).
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
        Generate test cases from user input.
        
        Args:
            user_input: The user's input (requirements, etc.)
            additional_instructions: Additional instructions from user
            images: Optional images (for vision models)
            
        Returns:
            Generated test cases as a string
        """
        # Get RAG context if available
        rag_context = self._get_rag_context(user_input)
        
        # Get prompts
        system_prompt, user_prompt = PromptTemplates.get_generator_prompts(
            user_input=user_input,
            additional_instructions=additional_instructions,
            rag_context=rag_context
        )
        
        # Create messages and invoke LLM
        messages = self._create_messages(system_prompt, user_prompt, images)
        llm = self._get_llm()
        response = llm.invoke(messages)
        
        return response.content


class ReviewerNode(BaseNode):
    """
    Node 2: Test Case Reviewer
    
    Responsible for reviewing generated test cases and providing feedback.
    Uses a more powerful reasoning model for thorough analysis.
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
        Review the generated test cases.
        
        Args:
            original_input: The original user input
            test_cases: The test cases generated by the generator node
            
        Returns:
            Review feedback as a string
        """
        # Get RAG context if available
        rag_context = self._get_rag_context(original_input)
        
        # Get prompts
        system_prompt, user_prompt = PromptTemplates.get_reviewer_prompts(
            original_input=original_input,
            test_cases=test_cases,
            rag_context=rag_context
        )
        
        # Create messages and invoke LLM
        messages = self._create_messages(system_prompt, user_prompt)
        llm = self._get_llm()
        response = llm.invoke(messages)
        
        return response.content


class OptimizerNode(BaseNode):
    """
    Node 3: Test Case Optimizer
    
    Responsible for optimizing test cases based on review feedback.
    Uses the same model as the generator for consistency.
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
        Optimize test cases based on review feedback.
        
        Args:
            original_input: The original user input
            initial_test_cases: The initial test cases from generator
            review_feedback: The feedback from the reviewer
            output_format: Desired output format (markdown/confluence)
            
        Returns:
            Optimized test cases as a string
        """
        # Get RAG context if available
        rag_context = self._get_rag_context(original_input)
        
        # Get prompts
        system_prompt, user_prompt = PromptTemplates.get_optimizer_prompts(
            original_input=original_input,
            initial_test_cases=initial_test_cases,
            review_feedback=review_feedback,
            output_format=output_format,
            rag_context=rag_context
        )
        
        # Create messages and invoke LLM
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
    Factory function to create all three nodes.
    
    Args:
        generator_config: Configuration for the generator node
        reviewer_config: Configuration for the reviewer node
        optimizer_config: Configuration for the optimizer node
        rag_interface: Optional shared RAG interface
        
    Returns:
        Tuple of (GeneratorNode, ReviewerNode, OptimizerNode)
    """
    generator = GeneratorNode(generator_config, rag_interface)
    reviewer = ReviewerNode(reviewer_config, rag_interface)
    optimizer = OptimizerNode(optimizer_config, rag_interface)
    
    return generator, reviewer, optimizer
