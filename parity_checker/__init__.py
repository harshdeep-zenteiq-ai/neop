"""Parity Checker - Agentic Framework for JAX/PyTorch Parity Testing."""

from agents.parity_agent import ParityAgent
from utils.llm_client import QwenLLMClient

__version__ = "1.0.0"
__all__ = ["ParityAgent", "QwenLLMClient"]
