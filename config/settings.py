"""
Configuration settings for the Test Case Generator Workflow.

This module contains all configurable parameters including:
- API keys and base URLs
- Model configurations for each node
- RAG settings (for future extension)
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class ModelConfig:
    """
    Model configuration for a single LLM node.
    
    Attributes:
        api_key: API key for the model provider
        base_url: Base URL for the API endpoint
        model_name: Name of the model to use
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens in response
        timeout: Request timeout in seconds
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 120,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }


class Settings(BaseSettings):
    """
    Global settings for the application.
    
    Settings can be configured via:
    1. Environment variables
    2. .env file
    3. Direct assignment
    
    Environment variable naming convention:
    - GENERATOR_API_KEY, GENERATOR_BASE_URL, GENERATOR_MODEL_NAME
    - REVIEWER_API_KEY, REVIEWER_BASE_URL, REVIEWER_MODEL_NAME
    - OPTIMIZER_API_KEY, OPTIMIZER_BASE_URL, OPTIMIZER_MODEL_NAME
    """
    
    # ============================================
    # Node 1: Generator (test case generation)
    # ============================================
    generator_api_key: str = Field(
        default="",
        description="API key for the generator model"
    )
    generator_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for the generator model API"
    )
    generator_model_name: str = Field(
        default="gpt-4o",
        description="Model name for test case generation"
    )
    generator_temperature: float = Field(
        default=0.7,
        description="Temperature for generator model (higher = more creative)"
    )
    generator_max_tokens: int = Field(
        default=4096,
        description="Max tokens for generator response"
    )
    
    # ============================================
    # Node 2: Reviewer (test case review)
    # Uses a more powerful/thoughtful model
    # ============================================
    reviewer_api_key: str = Field(
        default="",
        description="API key for the reviewer model"
    )
    reviewer_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for the reviewer model API"
    )
    reviewer_model_name: str = Field(
        default="o1-preview",
        description="Model name for test case review (should be a reasoning model)"
    )
    reviewer_temperature: float = Field(
        default=1.0,
        description="Temperature for reviewer model"
    )
    reviewer_max_tokens: int = Field(
        default=8192,
        description="Max tokens for reviewer response"
    )
    
    # ============================================
    # Node 3: Optimizer (test case optimization)
    # Uses the same model as Generator
    # ============================================
    optimizer_api_key: str = Field(
        default="",
        description="API key for the optimizer model"
    )
    optimizer_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for the optimizer model API"
    )
    optimizer_model_name: str = Field(
        default="gpt-4o",
        description="Model name for test case optimization"
    )
    optimizer_temperature: float = Field(
        default=0.5,
        description="Temperature for optimizer model (lower for more precise output)"
    )
    optimizer_max_tokens: int = Field(
        default=8192,
        description="Max tokens for optimizer response"
    )
    
    # ============================================
    # RAG Configuration (for future extension)
    # ============================================
    rag_enabled: bool = Field(
        default=False,
        description="Enable RAG functionality"
    )
    rag_collection_name: str = Field(
        default="test_case_knowledge",
        description="Vector store collection name"
    )
    rag_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model for RAG"
    )
    rag_top_k: int = Field(
        default=5,
        description="Number of documents to retrieve"
    )
    
    # ============================================
    # General Settings
    # ============================================
    request_timeout: int = Field(
        default=120,
        description="Default request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries for API calls"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
    
    def get_generator_config(self) -> ModelConfig:
        """Get configuration for the generator node."""
        return ModelConfig(
            api_key=self.generator_api_key,
            base_url=self.generator_base_url,
            model_name=self.generator_model_name,
            temperature=self.generator_temperature,
            max_tokens=self.generator_max_tokens,
            timeout=self.request_timeout,
        )
    
    def get_reviewer_config(self) -> ModelConfig:
        """Get configuration for the reviewer node."""
        return ModelConfig(
            api_key=self.reviewer_api_key,
            base_url=self.reviewer_base_url,
            model_name=self.reviewer_model_name,
            temperature=self.reviewer_temperature,
            max_tokens=self.reviewer_max_tokens,
            timeout=self.request_timeout,
        )
    
    def get_optimizer_config(self) -> ModelConfig:
        """Get configuration for the optimizer node."""
        return ModelConfig(
            api_key=self.optimizer_api_key,
            base_url=self.optimizer_base_url,
            model_name=self.optimizer_model_name,
            temperature=self.optimizer_temperature,
            max_tokens=self.optimizer_max_tokens,
            timeout=self.request_timeout,
        )
    
    def use_same_key_for_all(self, api_key: str, base_url: Optional[str] = None):
        """
        Convenience method to use the same API key (and optionally base URL)
        for all nodes.
        """
        self.generator_api_key = api_key
        self.reviewer_api_key = api_key
        self.optimizer_api_key = api_key
        
        if base_url:
            self.generator_base_url = base_url
            self.reviewer_base_url = base_url
            self.optimizer_base_url = base_url


# Global settings instance
settings = Settings()


def configure_from_env():
    """
    Configure settings from environment variables.
    
    This function is called automatically when the module is imported,
    but can be called again to refresh settings from environment.
    """
    global settings
    settings = Settings()
    
    # If optimizer settings are not explicitly set, use generator settings
    if not settings.optimizer_api_key and settings.generator_api_key:
        settings.optimizer_api_key = settings.generator_api_key
    if settings.optimizer_base_url == "https://api.openai.com/v1" and settings.generator_base_url != "https://api.openai.com/v1":
        settings.optimizer_base_url = settings.generator_base_url
    if settings.optimizer_model_name == "gpt-4o" and settings.generator_model_name != "gpt-4o":
        settings.optimizer_model_name = settings.generator_model_name
    
    return settings


# Auto-configure on import
configure_from_env()
