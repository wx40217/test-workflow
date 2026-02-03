"""
Prompt templates for the Test Case Generator Workflow.

This module contains all prompt templates used by the three nodes:
1. Generator Node - Generates initial test cases
2. Reviewer Node - Reviews and provides feedback
3. Optimizer Node - Optimizes based on feedback

All prompts are customizable and can be modified as needed.
"""

from typing import Optional


class PromptTemplates:
    """
    Container for all prompt templates used in the workflow.
    
    Each prompt is a class attribute that can be customized.
    Prompts use Python string formatting with named placeholders.
    """
    
    # ============================================
    # Node 1: Generator Prompts
    # ============================================
    
    GENERATOR_SYSTEM_PROMPT: str = """You are an expert software test engineer specializing in test case design. 
Your task is to analyze the provided requirements/documentation and generate comprehensive test cases.

Follow these guidelines:
1. Create test cases that cover all functional requirements
2. Include positive and negative test scenarios
3. Consider edge cases and boundary conditions
4. Include test cases for error handling and exception scenarios
5. Organize test cases in a hierarchical structure with clear parent-child relationships

Output Format:
Generate test cases in a hierarchical nested list format. Use consistent indentation to show relationships:
- Level 1: Test Suite/Module name
  - Level 2: Test Category/Feature
    - Level 3: Specific Test Case
      - Level 4: Test Steps (if needed)

Each test case should include:
- A clear, descriptive title
- Preconditions (if any)
- Expected results

{rag_context}"""

    GENERATOR_USER_PROMPT: str = """Please analyze the following input and generate comprehensive test cases:

---
{user_input}
---

{additional_instructions}

Generate test cases in the hierarchical list format as specified. Ensure thorough coverage of all requirements mentioned."""

    # ============================================
    # Node 2: Reviewer Prompts
    # ============================================
    
    REVIEWER_SYSTEM_PROMPT: str = """You are a senior QA lead and test architect with extensive experience in test case review.
Your task is to critically review the provided test cases and provide detailed feedback for improvement.

Review Criteria:
1. **Coverage**: Are all requirements adequately covered? Any missing scenarios?
2. **Completeness**: Does each test case have clear steps, preconditions, and expected results?
3. **Clarity**: Are the test cases well-written and unambiguous?
4. **Structure**: Is the hierarchy logical and well-organized?
5. **Edge Cases**: Are boundary conditions and edge cases covered?
6. **Negative Testing**: Are error scenarios and exception handling tested?
7. **Redundancy**: Are there duplicate or overlapping test cases?
8. **Feasibility**: Are the test cases practical and executable?

Provide your feedback in a structured format:
1. **Overall Assessment**: Brief summary of the test case quality
2. **Strengths**: What is done well
3. **Issues Found**: Specific problems identified
4. **Missing Coverage**: Test scenarios that should be added
5. **Suggested Improvements**: Concrete recommendations for each issue

Be thorough but constructive in your feedback.

{rag_context}"""

    REVIEWER_USER_PROMPT: str = """Please review the following test cases generated from the original requirements.

## Original Requirements/Input:
{original_input}

## Generated Test Cases:
{test_cases}

---

Provide a comprehensive review with specific, actionable feedback for improvement."""

    # ============================================
    # Node 3: Optimizer Prompts
    # ============================================
    
    OPTIMIZER_SYSTEM_PROMPT: str = """You are an expert test case designer responsible for finalizing test cases.
Your task is to optimize and improve test cases based on review feedback.

Guidelines:
1. Address ALL issues mentioned in the review feedback
2. Add any missing test cases identified
3. Improve clarity and structure as suggested
4. Remove any redundant test cases
5. Ensure consistent formatting throughout
6. Maintain the hierarchical structure

Output Format:
Generate the final test cases in one of the following formats based on the output_format parameter:

For Confluence Task List format:
- Use bullet points with proper indentation
- Each level should be indented consistently
- Format: "- [ ] Test Case Name: Description"

For Markdown format:
- Use unordered lists with proper indentation (2 or 4 spaces per level)
- Format: "- Test Case Name: Description"

The output should be clean, well-organized, and ready for direct use.

{rag_context}"""

    OPTIMIZER_USER_PROMPT: str = """Based on the review feedback, please optimize and finalize the test cases.

## Original Requirements:
{original_input}

## Initial Test Cases:
{initial_test_cases}

## Review Feedback:
{review_feedback}

## Output Format:
{output_format}

---

Generate the optimized, final version of the test cases. Address all feedback points and ensure comprehensive coverage."""

    # ============================================
    # RAG Context Templates
    # ============================================
    
    RAG_CONTEXT_TEMPLATE: str = """
## Reference Materials (from knowledge base):
The following relevant information has been retrieved from the knowledge base to assist you:

{retrieved_documents}

Use this information to enhance your response where applicable.
"""

    RAG_EMPTY_CONTEXT: str = ""

    @classmethod
    def get_generator_prompts(
        cls,
        user_input: str,
        additional_instructions: str = "",
        rag_context: str = ""
    ) -> tuple[str, str]:
        """
        Get formatted prompts for the generator node.
        
        Args:
            user_input: The user's input (requirements, documentation, etc.)
            additional_instructions: Any additional instructions from the user
            rag_context: Context retrieved from RAG (if enabled)
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        rag_section = cls.RAG_CONTEXT_TEMPLATE.format(
            retrieved_documents=rag_context
        ) if rag_context else cls.RAG_EMPTY_CONTEXT
        
        system_prompt = cls.GENERATOR_SYSTEM_PROMPT.format(
            rag_context=rag_section
        )
        
        user_prompt = cls.GENERATOR_USER_PROMPT.format(
            user_input=user_input,
            additional_instructions=additional_instructions
        )
        
        return system_prompt, user_prompt
    
    @classmethod
    def get_reviewer_prompts(
        cls,
        original_input: str,
        test_cases: str,
        rag_context: str = ""
    ) -> tuple[str, str]:
        """
        Get formatted prompts for the reviewer node.
        
        Args:
            original_input: The original user input
            test_cases: The test cases generated by the generator node
            rag_context: Context retrieved from RAG (if enabled)
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        rag_section = cls.RAG_CONTEXT_TEMPLATE.format(
            retrieved_documents=rag_context
        ) if rag_context else cls.RAG_EMPTY_CONTEXT
        
        system_prompt = cls.REVIEWER_SYSTEM_PROMPT.format(
            rag_context=rag_section
        )
        
        user_prompt = cls.REVIEWER_USER_PROMPT.format(
            original_input=original_input,
            test_cases=test_cases
        )
        
        return system_prompt, user_prompt
    
    @classmethod
    def get_optimizer_prompts(
        cls,
        original_input: str,
        initial_test_cases: str,
        review_feedback: str,
        output_format: str = "markdown",
        rag_context: str = ""
    ) -> tuple[str, str]:
        """
        Get formatted prompts for the optimizer node.
        
        Args:
            original_input: The original user input
            initial_test_cases: The initial test cases from generator
            review_feedback: The feedback from the reviewer
            output_format: Desired output format (markdown/confluence)
            rag_context: Context retrieved from RAG (if enabled)
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        rag_section = cls.RAG_CONTEXT_TEMPLATE.format(
            retrieved_documents=rag_context
        ) if rag_context else cls.RAG_EMPTY_CONTEXT
        
        system_prompt = cls.OPTIMIZER_SYSTEM_PROMPT.format(
            rag_context=rag_section
        )
        
        format_instruction = "Markdown nested list format" if output_format == "markdown" else "Confluence task list format"
        
        user_prompt = cls.OPTIMIZER_USER_PROMPT.format(
            original_input=original_input,
            initial_test_cases=initial_test_cases,
            review_feedback=review_feedback,
            output_format=format_instruction
        )
        
        return system_prompt, user_prompt

    @classmethod
    def customize_prompt(
        cls,
        prompt_name: str,
        new_template: str
    ) -> None:
        """
        Customize a specific prompt template at runtime.
        
        Args:
            prompt_name: Name of the prompt attribute to customize
            new_template: New template string
        
        Example:
            PromptTemplates.customize_prompt(
                "GENERATOR_SYSTEM_PROMPT",
                "Your custom prompt here..."
            )
        """
        if hasattr(cls, prompt_name):
            setattr(cls, prompt_name, new_template)
        else:
            raise ValueError(f"Unknown prompt: {prompt_name}")
