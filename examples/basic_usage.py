"""
测试用例生成器的基本使用示例。

本文件演示了使用测试用例生成器的各种方式。
"""

import os
import sys

# 添加工作区到路径
sys.path.insert(0, '/workspace')

from main import generate_test_cases
from src.workflow.graph import TestCaseWorkflow, create_workflow
from src.input_handler.handlers import InputHandler
from config.settings import settings, ModelConfig


def example_1_simple_text_input():
    """
    示例1：从简单文本输入生成测试用例。
    """
    print("=" * 60)
    print("示例1：简单文本输入")
    print("=" * 60)
    
    requirements = """
    用户登录功能需求：
    
    1. 用户可以使用邮箱和密码登录
    2. 密码必须至少8个字符
    3. 3次失败尝试后，账户锁定30分钟
    4. 用户可以通过邮箱重置密码
    5. "记住我"选项可保持用户登录30天
    """
    
    result = generate_test_cases(
        requirements,
        api_key=os.getenv("OPENAI_API_KEY"),
        output_format="markdown",
        verbose=True
    )
    
    print("\n生成的测试用例：")
    print(result.final_test_cases)
    
    return result


def example_2_file_input():
    """
    示例2：从文件生成测试用例。
    """
    print("=" * 60)
    print("示例2：文件输入")
    print("=" * 60)
    
    # 假设有一个需求文件
    file_path = "requirements.docx"
    
    if not os.path.exists(file_path):
        print(f"文件未找到: {file_path}")
        print("请提供一个需求文件来测试此示例。")
        return None
    
    result = generate_test_cases(
        file_path,
        api_key=os.getenv("OPENAI_API_KEY"),
        output_format="markdown",
        verbose=True
    )
    
    print("\n生成的测试用例：")
    print(result.final_test_cases)
    
    return result


def example_3_custom_models():
    """
    示例3：为每个节点使用自定义模型。
    """
    print("=" * 60)
    print("示例3：自定义模型")
    print("=" * 60)
    
    requirements = """
    购物车功能：
    - 添加商品到购物车
    - 更新数量
    - 移除商品
    - 应用折扣码
    - 计算含税总价
    """
    
    result = generate_test_cases(
        requirements,
        api_key=os.getenv("OPENAI_API_KEY"),
        generator_model="gpt-4o",
        reviewer_model="gpt-4o",  # 如果o1不可用则使用相同模型
        optimizer_model="gpt-4o",
        output_format="confluence",  # 使用Confluence格式
        verbose=True
    )
    
    print("\n生成的测试用例（Confluence格式）：")
    print(result.final_test_cases)
    
    return result


def example_4_with_additional_instructions():
    """
    示例4：添加额外指示进行自定义。
    """
    print("=" * 60)
    print("示例4：额外指示")
    print("=" * 60)
    
    requirements = """
    支付处理：
    - 接受信用卡（Visa、MasterCard、Amex）
    - 支持PayPal
    - 处理退款
    """
    
    additional = """
    请重点关注：
    - 安全测试（SQL注入、XSS）
    - 金额的边界情况（0、负数、超大金额）
    - 国际支付场景
    - 货币转换
    """
    
    result = generate_test_cases(
        requirements,
        api_key=os.getenv("OPENAI_API_KEY"),
        additional_instructions=additional,
        output_format="markdown",
        verbose=True
    )
    
    print("\n生成的测试用例：")
    print(result.final_test_cases)
    
    return result


def example_5_direct_workflow_usage():
    """
    示例5：直接使用工作流以获得更多控制。
    """
    print("=" * 60)
    print("示例5：直接工作流使用")
    print("=" * 60)
    
    # 创建自定义模型配置
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = "https://api.openai.com/v1"
    
    generator_config = ModelConfig(
        api_key=api_key,
        base_url=base_url,
        model_name="gpt-4o",
        temperature=0.8,  # 更有创造性
        max_tokens=4096
    )
    
    reviewer_config = ModelConfig(
        api_key=api_key,
        base_url=base_url,
        model_name="gpt-4o",
        temperature=0.3,  # 更精确
        max_tokens=8192
    )
    
    optimizer_config = ModelConfig(
        api_key=api_key,
        base_url=base_url,
        model_name="gpt-4o",
        temperature=0.5,
        max_tokens=8192
    )
    
    # 使用自定义配置创建工作流
    workflow = TestCaseWorkflow(
        generator_config=generator_config,
        reviewer_config=reviewer_config,
        optimizer_config=optimizer_config,
        output_format="markdown"
    )
    
    requirements = """
    用户注册：
    - 使用邮箱注册
    - 需要邮箱验证
    - 密码强度要求
    - 服务条款接受
    """
    
    # 逐步运行以查看进度
    print("\n逐步运行工作流：")
    for step, result in workflow.run_step_by_step(requirements):
        if result is None:
            print(f"  步骤: {step}...")
        else:
            print(f"  步骤: {step} - 完成")
            if step == "generated":
                print("\n--- 初始测试用例 ---")
                print(result[:500] + "..." if len(result) > 500 else result)
            elif step == "reviewed":
                print("\n--- 评审反馈 ---")
                print(result[:500] + "..." if len(result) > 500 else result)
    
    # 获取最终结果
    final_result = workflow.run(requirements)
    
    print("\n--- 最终测试用例 ---")
    print(final_result.final_test_cases)
    
    return final_result


def example_6_input_handler():
    """
    示例6：直接使用InputHandler进行文件处理。
    """
    print("=" * 60)
    print("示例6：输入处理器使用")
    print("=" * 60)
    
    handler = InputHandler()
    
    # 处理文本
    text_result = handler.process_text("示例需求文本")
    print(f"文本输入类型: {text_result.input_type}")
    print(f"文本内容: {text_result.text_content[:50]}...")
    
    # 如果文件存在也可以处理
    # result = handler.process_file("requirements.pdf")
    # result = handler.process_directory("docs/")
    # result = handler.process_multiple(["doc1.docx", "doc2.pdf", "image.png"])


def example_7_with_rag():
    """
    示例7：使用RAG增强上下文。
    """
    print("=" * 60)
    print("示例7：使用RAG")
    print("=" * 60)
    
    # 知识库文档
    knowledge_docs = [
        """
        测试用例最佳实践：
        - 每个测试用例应有清晰的前置条件
        - 测试用例应该相互独立
        - 包含正向和负向测试场景
        - 清晰记录预期结果
        """,
        """
        安全测试指南：
        - 测试SQL注入
        - 测试XSS漏洞
        - 验证认证边界情况
        - 测试会话管理
        """
    ]
    
    requirements = """
    用户认证API：
    - POST /login - 使用凭证登录
    - POST /logout - 用户登出
    - POST /refresh-token - 刷新认证令牌
    - POST /change-password - 修改密码
    """
    
    result = generate_test_cases(
        requirements,
        api_key=os.getenv("OPENAI_API_KEY"),
        enable_rag=True,
        rag_documents=knowledge_docs,
        verbose=True
    )
    
    print("\n生成的测试用例（带RAG上下文）：")
    print(result.final_test_cases)
    
    return result


if __name__ == "__main__":
    # 检查API密钥
    if not os.getenv("OPENAI_API_KEY"):
        print("请设置OPENAI_API_KEY环境变量")
        print("示例: export OPENAI_API_KEY=sk-...")
        sys.exit(1)
    
    # 运行示例1（其他示例需要实际API调用）
    print("\n运行示例1：简单文本输入")
    print("注意：这需要有效的API密钥才能实际运行。\n")
    
    try:
        example_1_simple_text_input()
    except Exception as e:
        print(f"错误: {e}")
        print("\n请确保已设置有效的API密钥。")
