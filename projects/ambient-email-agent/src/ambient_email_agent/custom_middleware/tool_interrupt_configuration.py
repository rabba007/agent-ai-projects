from dataclasses import dataclass
from typing import Callable


@dataclass
class ToolInterruptConfig:
    """Configuration for a single tool's interrupt behavior"""
    payload_builder: Callable  # Function to build custom payload
    response_processor: Callable  # Function to process user response
    description: str
