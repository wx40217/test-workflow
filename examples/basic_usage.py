"""
Basic usage examples for the Test Case Generator.

This file demonstrates various ways to use the test case generator.
"""

import os
import sys

# Add workspace to path
sys.path.insert(0, '/workspace')

from main import generate_test_cases
from src.workflow.graph import TestCaseWorkflow, create_workflow
from src.input_handler.handlers import InputHandler
from config.settings import settings, ModelConfig


def example_1_simple_text_input():
    """
    Example 1: Generate test cases from simple text input.
    """
    print("=" * 60)
    print("Example 1: Simple Text Input")
    print("=" * 60)
    
    requirements = """
    User Login Feature Requirements:
    
    1. Users can login with email and password
    2. Password must be at least 8 characters
    3. After 3 failed attempts, account is locked for 30 minutes
    4. Users can reset password via email
    5. Remember me option keeps user logged in for 30 days
    """
    
    result = generate_test_cases(
        requirements,
        api_key=os.getenv("OPENAI_API_KEY"),
        output_format="markdown",
        verbose=True
    )
    
    print("\nGenerated Test Cases:")
    print(result.final_test_cases)
    
    return result


def example_2_file_input():
    """
    Example 2: Generate test cases from a file.
    """
    print("=" * 60)
    print("Example 2: File Input")
    print("=" * 60)
    
    # Assuming you have a requirements file
    file_path = "requirements.docx"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        print("Please provide a requirements file to test this example.")
        return None
    
    result = generate_test_cases(
        file_path,
        api_key=os.getenv("OPENAI_API_KEY"),
        output_format="markdown",
        verbose=True
    )
    
    print("\nGenerated Test Cases:")
    print(result.final_test_cases)
    
    return result


def example_3_custom_models():
    """
    Example 3: Use custom models for each node.
    """
    print("=" * 60)
    print("Example 3: Custom Models")
    print("=" * 60)
    
    requirements = """
    Shopping Cart Feature:
    - Add items to cart
    - Update quantities
    - Remove items
    - Apply discount codes
    - Calculate total with tax
    """
    
    result = generate_test_cases(
        requirements,
        api_key=os.getenv("OPENAI_API_KEY"),
        generator_model="gpt-4o",
        reviewer_model="gpt-4o",  # Use same model if o1 not available
        optimizer_model="gpt-4o",
        output_format="confluence",  # Use Confluence format
        verbose=True
    )
    
    print("\nGenerated Test Cases (Confluence format):")
    print(result.final_test_cases)
    
    return result


def example_4_with_additional_instructions():
    """
    Example 4: Add additional instructions for customization.
    """
    print("=" * 60)
    print("Example 4: Additional Instructions")
    print("=" * 60)
    
    requirements = """
    Payment Processing:
    - Accept credit cards (Visa, MasterCard, Amex)
    - Support PayPal
    - Handle refunds
    """
    
    additional = """
    Please focus on:
    - Security testing (SQL injection, XSS)
    - Edge cases for amounts (0, negative, very large)
    - International payment scenarios
    - Currency conversion
    """
    
    result = generate_test_cases(
        requirements,
        api_key=os.getenv("OPENAI_API_KEY"),
        additional_instructions=additional,
        output_format="markdown",
        verbose=True
    )
    
    print("\nGenerated Test Cases:")
    print(result.final_test_cases)
    
    return result


def example_5_direct_workflow_usage():
    """
    Example 5: Use the workflow directly for more control.
    """
    print("=" * 60)
    print("Example 5: Direct Workflow Usage")
    print("=" * 60)
    
    # Create custom model configurations
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = "https://api.openai.com/v1"
    
    generator_config = ModelConfig(
        api_key=api_key,
        base_url=base_url,
        model_name="gpt-4o",
        temperature=0.8,  # More creative
        max_tokens=4096
    )
    
    reviewer_config = ModelConfig(
        api_key=api_key,
        base_url=base_url,
        model_name="gpt-4o",
        temperature=0.3,  # More precise
        max_tokens=8192
    )
    
    optimizer_config = ModelConfig(
        api_key=api_key,
        base_url=base_url,
        model_name="gpt-4o",
        temperature=0.5,
        max_tokens=8192
    )
    
    # Create workflow with custom configurations
    workflow = TestCaseWorkflow(
        generator_config=generator_config,
        reviewer_config=reviewer_config,
        optimizer_config=optimizer_config,
        output_format="markdown"
    )
    
    requirements = """
    User Registration:
    - Sign up with email
    - Email verification required
    - Password strength requirements
    - Terms of service acceptance
    """
    
    # Run step by step to see progress
    print("\nRunning workflow step by step:")
    for step, result in workflow.run_step_by_step(requirements):
        if result is None:
            print(f"  Step: {step}...")
        else:
            print(f"  Step: {step} - Completed")
            if step == "generated":
                print("\n--- Initial Test Cases ---")
                print(result[:500] + "..." if len(result) > 500 else result)
            elif step == "reviewed":
                print("\n--- Review Feedback ---")
                print(result[:500] + "..." if len(result) > 500 else result)
    
    # Get final result
    final_result = workflow.run(requirements)
    
    print("\n--- Final Test Cases ---")
    print(final_result.final_test_cases)
    
    return final_result


def example_6_input_handler():
    """
    Example 6: Use the InputHandler directly for file processing.
    """
    print("=" * 60)
    print("Example 6: Input Handler Usage")
    print("=" * 60)
    
    handler = InputHandler()
    
    # Process text
    text_result = handler.process_text("Sample requirements text")
    print(f"Text input type: {text_result.input_type}")
    print(f"Text content: {text_result.text_content[:50]}...")
    
    # You can also process files if they exist
    # result = handler.process_file("requirements.pdf")
    # result = handler.process_directory("docs/")
    # result = handler.process_multiple(["doc1.docx", "doc2.pdf", "image.png"])


def example_7_with_rag():
    """
    Example 7: Use RAG for enhanced context.
    """
    print("=" * 60)
    print("Example 7: With RAG")
    print("=" * 60)
    
    # Knowledge base documents
    knowledge_docs = [
        """
        Test Case Best Practices:
        - Each test case should have clear preconditions
        - Test cases should be independent
        - Include both positive and negative scenarios
        - Document expected results clearly
        """,
        """
        Security Testing Guidelines:
        - Test for SQL injection
        - Test for XSS vulnerabilities
        - Verify authentication edge cases
        - Test session management
        """
    ]
    
    requirements = """
    User Authentication API:
    - POST /login - Login with credentials
    - POST /logout - Logout user
    - POST /refresh-token - Refresh auth token
    - POST /change-password - Change password
    """
    
    result = generate_test_cases(
        requirements,
        api_key=os.getenv("OPENAI_API_KEY"),
        enable_rag=True,
        rag_documents=knowledge_docs,
        verbose=True
    )
    
    print("\nGenerated Test Cases (with RAG context):")
    print(result.final_test_cases)
    
    return result


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY=sk-...")
        sys.exit(1)
    
    # Run example 1 (others require actual API calls)
    print("\nRunning Example 1: Simple Text Input")
    print("Note: This requires a valid API key to actually work.\n")
    
    try:
        example_1_simple_text_input()
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have a valid API key set.")
