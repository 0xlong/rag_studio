"""
Configuration module for managing environment variables and API keys.
This module provides a safe way to access environment variables and validates their presence.
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Configuration class that handles environment variables and provides type-safe access to them.
    """
    @staticmethod
    def get_env(key: str, default: Optional[str] = None) -> str:
        """
        Safely get an environment variable.
        
        Args:
            key: The name of the environment variable
            default: Optional default value if the variable is not found
            
        Returns:
            The value of the environment variable or the default value
            
        Raises:
            ValueError: If the environment variable is not found and no default is provided
        """
        value = os.getenv(key, default)
        if value is None:
            raise ValueError(f"Environment variable {key} is not set and no default value provided")
        return value

    # API Keys
    @classmethod
    def get_openai_api_key(cls) -> str:
        """Get OpenAI API key from environment variables."""
        return cls.get_env("OPENAI_API_KEY")

    @classmethod
    def get_pinecone_api_key(cls) -> str:
        """Get Pinecone API key from environment variables."""
        return cls.get_env("PINECONE_API_KEY")

    @classmethod
    def get_pinecone_environment(cls) -> str:
        """Get Pinecone environment from environment variables."""
        return cls.get_env("PINECONE_ENVIRONMENT")

    # Add more API key getters as needed 