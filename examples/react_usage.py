"""
ReAct 模式使用示例。

运行前需要配置 OPENAI_API_KEY，或改用 GENERATOR_API_KEY / REVIEWER_API_KEY /
OPTIMIZER_API_KEY 等环境变量。
"""

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main import generate_test_cases


def main() -> None:
    requirements = """
    用户登录功能：
    1. 用户可以使用邮箱和密码登录
    2. 连续 3 次密码错误后锁定账户 30 分钟
    3. 支持记住登录状态 30 天
    """

    result = generate_test_cases(
        requirements,
        api_key=os.getenv("OPENAI_API_KEY"),
        agent_mode="react",
        max_agent_steps=8,
        output_format="markdown",
        show_agent_trace=True,
        verbose=True,
    )

    print("\n最终测试用例：")
    print(result.final_test_cases)

    tools_used = result.metadata.get("tools_used", [])
    print("\n工具调用顺序：")
    print(" -> ".join(tools_used) if tools_used else "(无)")


if __name__ == "__main__":
    main()
