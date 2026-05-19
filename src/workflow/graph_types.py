"""
轻量结果协议，避免 react_agent 与 graph.py 互相导入。
"""

from dataclasses import dataclass, field


@dataclass
class WorkflowResultProtocol:
    success: bool
    final_test_cases: str
    generated_test_cases: str = ""
    review_feedback: str = ""
    errors: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
