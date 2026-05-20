"""
quality-graph 模式使用示例。

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
    会员退款功能：
    1. 用户可以对未发货订单申请退款
    2. 退款成功后需要回滚优惠券和积分
    3. 已开票订单退款时需要触发红冲流程
    4. 风控命中时进入人工审核
    """

    result = generate_test_cases(
        requirements,
        api_key=os.getenv("OPENAI_API_KEY"),
        agent_mode="quality-graph",
        max_review_rounds=3,
        quality_threshold=0.8,
        output_format="markdown",
        show_agent_trace=True,
        verbose=True,
    )

    print("\n最终测试用例：")
    print(result.final_test_cases)

    print("\n质量摘要：")
    print(f"score={result.metadata.get('quality_score')}")
    print(f"passed={result.metadata.get('quality_passed')}")
    print(f"review_rounds={result.metadata.get('review_rounds')}")


if __name__ == "__main__":
    main()
