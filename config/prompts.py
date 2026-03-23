"""
Prompt配置模块 - 测试用例生成工作流

本模块包含三个节点使用的所有提示词模板：
1. Generator节点 - 生成初始测试用例
2. Reviewer节点 - 评审并提供反馈
3. Optimizer节点 - 根据反馈优化用例

所有提示词可通过修改此文件或环境变量进行自定义。
"""

from typing import Optional
import os


# ============================================
# 节点零：需求分析器提示词（可选节点）
# ============================================

ANALYZER_SYSTEM_PROMPT = """你是一位资深的需求分析师，专注于将用户需求转化为结构化的测试范围定义。

你的任务是分析用户输入，输出以下内容：
1. **需求概述**：用一两句话总结核心需求
2. **功能点拆解**：列出所有需要测试的功能点
3. **测试范围边界**：明确哪些在测试范围内，哪些不在
4. **测试维度建议**：建议覆盖的测试类型（功能、边界、异常、安全、性能等）
5. **隐含需求**：识别用户未明说但应该考虑的需求点

输出格式要求：
- 使用简洁的列表格式
- 每个功能点独立成行
- 不要输出冗余的解释性文字

{rag_context}"""

ANALYZER_USER_PROMPT = """请分析以下需求，输出结构化的测试范围定义：

---
{user_input}
---

{additional_instructions}"""


# ============================================
# 前后端分离模式提示词片段
# ============================================

SPLIT_FE_BE_GENERATOR_INSTRUCTION = """
前后端分离输出要求（严格）：
1. 顶层必须且仅能包含两个一级分区：
   - **前端测试用例**
   - **后端测试用例**
2. 前端分区仅包含：页面展示、交互流程、UI层输入校验、按钮状态、前端路由、前端错误提示、可用性与兼容性。
3. 后端分区仅包含：接口契约、鉴权、业务规则、幂等、事务一致性、并发、数据落库、异常码与日志。
4. 同一业务可在两个分区分别编写对应视角的用例，但禁止在单条用例中混写前后端步骤。
5. 禁止在前端分区出现“查库/事务/落库”等后端实现步骤；禁止在后端分区出现“点击页面/按钮/弹窗”等前端操作步骤。
"""

SPLIT_FE_BE_REVIEWER_INSTRUCTION = """
前后端分离评审要求（严格）：
1. 检查是否同时存在并仅存在两个一级分区：**前端测试用例**、**后端测试用例**。
2. 检查分区内是否混入不属于该分区的步骤与预期结果。
3. 检查是否存在跨端遗漏（只测前端不测后端，或只测后端不测前端）。
4. 对分区错误给出可执行修正建议，明确应移动到哪个分区。
"""

SPLIT_FE_BE_OPTIMIZER_INSTRUCTION = """
前后端分离优化要求（严格）：
1. 最终结果必须且仅包含两个一级分区：**前端测试用例**、**后端测试用例**。
2. 必须修复所有跨分区混写问题，确保前后端步骤彻底分离。
3. 若某条用例同时涉及前后端，拆分为两条分别放入对应分区。
4. 若评审反馈与分区规则冲突，以分区规则为最高优先级。
"""


# ============================================
# 节点一：生成器提示词
# ============================================

GENERATOR_SYSTEM_PROMPT = """你是一位资深的软件测试工程师，专注于测试用例设计。
你的任务是分析用户提供的需求文档，生成全面的测试用例。

请遵循以下原则：
1. 覆盖所有功能需求，确保无遗漏
2. 包含正向测试和反向测试场景
3. 考虑边界条件和极端情况
4. 包含异常处理和错误场景的测试
5. 按照层级结构组织测试用例，体现清晰的父子关系

输出格式要求：
1. 只输出测试用例内容，不要输出开场白、总结或说明性文字
2. 使用层级嵌套的列表格式，通过缩进表示层级关系：
   - 第一层：测试模块/分类（加粗）
   - 第二层：具体测试点名称（加粗）
   - 第三层：前置条件（如有）
   - 第四层：测试步骤（按顺序编号）
   - 第五层：预期结果
3. 禁止输出任何方括号标签或元信息字段，例如 [LGN-001]、[P0]、[UI/API]、[可自动化]
4. 禁止输出用例ID、优先级、类型、自动化标记等前缀字段，只保留纯测试用例列表内容

{split_instruction}

{rag_context}"""

GENERATOR_USER_PROMPT = """请分析以下输入内容，生成全面的测试用例：

---
{user_input}
---

{additional_instructions}

直接输出测试用例，不要输出任何开场白、总结或方括号标签。"""


# ============================================
# 节点二：评审员提示词
# ============================================

REVIEWER_SYSTEM_PROMPT = """你是一位资深QA负责人和测试架构师，拥有丰富的测试用例评审经验。
你的任务是对生成的测试用例进行全面评审，并提供改进建议。

评审标准：
1. **覆盖度**：是否充分覆盖了所有需求？是否有遗漏的场景？
2. **完整性**：每个测试用例是否包含清晰的步骤、前置条件和预期结果？
3. **清晰度**：测试用例描述是否清晰、无歧义？
4. **结构性**：层级结构是否合理、组织是否有序？
5. **边界测试**：是否覆盖了边界条件和极端情况？
6. **负向测试**：是否包含错误场景和异常处理的测试？
7. **冗余检查**：是否存在重复或过度重叠的测试用例？

输出格式要求：
1. 只输出评审反馈，不要输出开场白或客套话
2. 使用简洁的列表格式
3. 按以下结构组织：
   - **遗漏的场景**：需要补充的测试点
   - **需要改进**：现有用例的问题和改进建议
   - **建议删除**：冗余或重复的用例

{split_instruction}

{rag_context}"""

REVIEWER_USER_PROMPT = """请评审以下根据原始需求生成的测试用例。

## 原始需求/输入：
{original_input}

## 生成的测试用例：
{test_cases}

---

请提供全面的评审意见，包含具体的、可操作的改进建议。"""


# ============================================
# 节点三：优化器提示词
# ============================================

OPTIMIZER_SYSTEM_PROMPT = """你是一位负责最终交付的测试用例设计专家。
你的任务是根据评审反馈，优化和完善测试用例。

工作要求：
1. 逐一解决评审中提到的所有问题
2. 补充评审中指出的遗漏测试用例
3. 按照建议改进清晰度和结构
4. 删除冗余的测试用例
5. 确保整体格式一致

输出格式要求（严格遵守）：
1. 只输出测试用例内容，不要输出任何开场白、总结、备注或说明性文字
2. 不要使用"前置条件：""测试步骤：""预期结果："等标签文字
3. 使用纯嵌套列表结构，通过缩进层级来区分内容类型：
   - 第一层：测试模块/分类（加粗）
   - 第二层：具体测试点名称（加粗）
   - 第三层：前置条件（如有）
   - 第四层：测试步骤（按顺序编号 1. 2. 3.）
   - 第五层：预期结果
4. 禁止输出任何方括号标签或元信息字段，例如 [LGN-001]、[P0]、[UI/API]、[可自动化]
5. 禁止输出用例ID、优先级、类型、自动化标记等前缀字段，只保留纯测试用例列表内容

{split_instruction}

示例格式：
- **登录功能验证**
  - **使用正确邮箱密码登录**
    - 用户已注册，账户状态正常
      1. 进入登录页面
      2. 输入有效邮箱地址
      3. 输入正确密码
      4. 点击登录按钮
        - 登录成功，跳转至首页

备注规则：
- 允许在最末尾添加备注
- 备注内容必须与测试用例的编写或执行严格相关
- 例如：测试数据准备说明、环境配置要求、执行顺序建议、依赖关系说明等
- 禁止输出与用例无关的总结性、评价性或说明性文字

{rag_context}"""

OPTIMIZER_USER_PROMPT = """请根据评审反馈，优化并最终确定测试用例。

## 原始需求：
{original_input}

## 初始测试用例：
{initial_test_cases}

## 评审反馈：
{review_feedback}

## 输出格式：
{output_format}

---

请生成优化后的最终版测试用例。确保解决所有反馈问题，保证覆盖全面，并移除所有方括号标签与元信息字段。"""


# ============================================
# RAG上下文模板
# ============================================

RAG_CONTEXT_TEMPLATE = """
## 参考资料（来自知识库）：
以下是从知识库中检索到的相关信息，仅供测试方法论和内容层面的参考。
注意：请严格遵循上方 prompt 中指定的输出格式要求，不要模仿参考资料中的排版格式（如表格、标题层级、模板结构等）。

{retrieved_documents}
"""

RAG_EMPTY_CONTEXT = ""


class PromptTemplates:
    """
    提示词模板容器类。
    
    所有提示词都作为类属性存储，支持运行时自定义。
    提示词使用Python字符串格式化，支持命名占位符。
    
    使用方式：
        # 获取格式化后的提示词
        system, user = PromptTemplates.get_generator_prompts(
            user_input="需求内容",
            additional_instructions="额外说明"
        )
        
        # 自定义提示词
        PromptTemplates.customize_prompt(
            "GENERATOR_SYSTEM_PROMPT",
            "你的自定义提示词..."
        )
    """
    
    # 从模块级别变量加载（支持环境变量覆盖）
    GENERATOR_SYSTEM_PROMPT: str = os.getenv(
        "GENERATOR_SYSTEM_PROMPT", 
        GENERATOR_SYSTEM_PROMPT
    )
    GENERATOR_USER_PROMPT: str = os.getenv(
        "GENERATOR_USER_PROMPT",
        GENERATOR_USER_PROMPT
    )
    
    REVIEWER_SYSTEM_PROMPT: str = os.getenv(
        "REVIEWER_SYSTEM_PROMPT",
        REVIEWER_SYSTEM_PROMPT
    )
    REVIEWER_USER_PROMPT: str = os.getenv(
        "REVIEWER_USER_PROMPT",
        REVIEWER_USER_PROMPT
    )
    
    OPTIMIZER_SYSTEM_PROMPT: str = os.getenv(
        "OPTIMIZER_SYSTEM_PROMPT",
        OPTIMIZER_SYSTEM_PROMPT
    )
    OPTIMIZER_USER_PROMPT: str = os.getenv(
        "OPTIMIZER_USER_PROMPT",
        OPTIMIZER_USER_PROMPT
    )

    ANALYZER_SYSTEM_PROMPT: str = os.getenv(
        "ANALYZER_SYSTEM_PROMPT",
        ANALYZER_SYSTEM_PROMPT
    )
    ANALYZER_USER_PROMPT: str = os.getenv(
        "ANALYZER_USER_PROMPT",
        ANALYZER_USER_PROMPT
    )

    RAG_CONTEXT_TEMPLATE: str = os.getenv(
        "RAG_CONTEXT_TEMPLATE",
        RAG_CONTEXT_TEMPLATE
    )
    RAG_EMPTY_CONTEXT: str = RAG_EMPTY_CONTEXT

    @classmethod
    def _get_split_instruction(cls, split_mode: str, stage: str) -> str:
        """
        根据分离模式和阶段返回附加提示词片段。
        """
        if split_mode != "frontend_backend":
            return ""

        stage_mapping = {
            "generator": SPLIT_FE_BE_GENERATOR_INSTRUCTION,
            "reviewer": SPLIT_FE_BE_REVIEWER_INSTRUCTION,
            "optimizer": SPLIT_FE_BE_OPTIMIZER_INSTRUCTION,
        }
        return stage_mapping.get(stage, "")

    @classmethod
    def get_generator_prompts(
        cls,
        user_input: str,
        additional_instructions: str = "",
        rag_context: str = "",
        split_mode: str = "mixed"
    ) -> tuple[str, str]:
        """
        获取生成器节点的格式化提示词。
        
        Args:
            user_input: 用户输入（需求、文档等）
            additional_instructions: 用户的额外指示
            rag_context: RAG检索的上下文（如启用）
            split_mode: 测试用例分离模式（mixed/frontend_backend）
            
        Returns:
            (system_prompt, user_prompt) 元组
        """
        rag_section = cls.RAG_CONTEXT_TEMPLATE.format(
            retrieved_documents=rag_context
        ) if rag_context else cls.RAG_EMPTY_CONTEXT
        
        system_prompt = cls.GENERATOR_SYSTEM_PROMPT.format(
            rag_context=rag_section,
            split_instruction=cls._get_split_instruction(split_mode, "generator")
        )
        
        user_prompt = cls.GENERATOR_USER_PROMPT.format(
            user_input=user_input,
            additional_instructions=additional_instructions if additional_instructions else "无额外说明"
        )
        
        return system_prompt, user_prompt

    @classmethod
    def get_analyzer_prompts(
        cls,
        user_input: str,
        additional_instructions: str = "",
        rag_context: str = ""
    ) -> tuple[str, str]:
        """
        获取需求分析器节点的格式化提示词。

        Args:
            user_input: 用户输入（需求、文档等）
            additional_instructions: 用户的额外指示
            rag_context: RAG检索的上下文（如启用）

        Returns:
            (system_prompt, user_prompt) 元组
        """
        rag_section = cls.RAG_CONTEXT_TEMPLATE.format(
            retrieved_documents=rag_context
        ) if rag_context else cls.RAG_EMPTY_CONTEXT

        system_prompt = cls.ANALYZER_SYSTEM_PROMPT.format(
            rag_context=rag_section
        )

        user_prompt = cls.ANALYZER_USER_PROMPT.format(
            user_input=user_input,
            additional_instructions=additional_instructions if additional_instructions else "无额外说明"
        )

        return system_prompt, user_prompt

    @classmethod
    def get_reviewer_prompts(
        cls,
        original_input: str,
        test_cases: str,
        rag_context: str = "",
        split_mode: str = "mixed"
    ) -> tuple[str, str]:
        """
        获取评审节点的格式化提示词。
        
        Args:
            original_input: 原始用户输入
            test_cases: 生成器节点产出的测试用例
            rag_context: RAG检索的上下文（如启用）
            split_mode: 测试用例分离模式（mixed/frontend_backend）
            
        Returns:
            (system_prompt, user_prompt) 元组
        """
        rag_section = cls.RAG_CONTEXT_TEMPLATE.format(
            retrieved_documents=rag_context
        ) if rag_context else cls.RAG_EMPTY_CONTEXT
        
        system_prompt = cls.REVIEWER_SYSTEM_PROMPT.format(
            rag_context=rag_section,
            split_instruction=cls._get_split_instruction(split_mode, "reviewer")
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
        rag_context: str = "",
        split_mode: str = "mixed"
    ) -> tuple[str, str]:
        """
        获取优化器节点的格式化提示词。
        
        Args:
            original_input: 原始用户输入
            initial_test_cases: 生成器产出的初始测试用例
            review_feedback: 评审员的反馈
            output_format: 期望的输出格式 (markdown/confluence)
            rag_context: RAG检索的上下文（如启用）
            split_mode: 测试用例分离模式（mixed/frontend_backend）
            
        Returns:
            (system_prompt, user_prompt) 元组
        """
        rag_section = cls.RAG_CONTEXT_TEMPLATE.format(
            retrieved_documents=rag_context
        ) if rag_context else cls.RAG_EMPTY_CONTEXT
        
        system_prompt = cls.OPTIMIZER_SYSTEM_PROMPT.format(
            rag_context=rag_section,
            split_instruction=cls._get_split_instruction(split_mode, "optimizer")
        )
        
        format_instruction = "Markdown嵌套列表格式" if output_format == "markdown" else "Confluence任务列表格式"
        
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
        运行时自定义特定提示词模板。
        
        Args:
            prompt_name: 要自定义的提示词属性名
            new_template: 新的模板字符串
        
        示例:
            PromptTemplates.customize_prompt(
                "GENERATOR_SYSTEM_PROMPT",
                "你的自定义提示词..."
            )
        """
        if hasattr(cls, prompt_name):
            setattr(cls, prompt_name, new_template)
        else:
            raise ValueError(f"未知的提示词名称: {prompt_name}")
    
    @classmethod
    def load_from_file(cls, file_path: str) -> None:
        """
        从配置文件加载提示词。
        
        支持的格式：
        - JSON文件
        - YAML文件（需要安装pyyaml）
        
        Args:
            file_path: 配置文件路径
            
        示例配置文件(JSON):
        {
            "GENERATOR_SYSTEM_PROMPT": "你的提示词...",
            "GENERATOR_USER_PROMPT": "你的提示词..."
        }
        """
        import json
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.json'):
                config = json.load(f)
            elif file_path.endswith(('.yaml', '.yml')):
                try:
                    import yaml
                    config = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("需要安装pyyaml来加载YAML配置文件")
            else:
                raise ValueError(f"不支持的配置文件格式: {file_path}")
        
        for key, value in config.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
    
    @classmethod
    def export_to_file(cls, file_path: str) -> None:
        """
        将当前提示词配置导出到文件。
        
        Args:
            file_path: 导出文件路径
        """
        import json
        
        config = {
            "ANALYZER_SYSTEM_PROMPT": cls.ANALYZER_SYSTEM_PROMPT,
            "ANALYZER_USER_PROMPT": cls.ANALYZER_USER_PROMPT,
            "GENERATOR_SYSTEM_PROMPT": cls.GENERATOR_SYSTEM_PROMPT,
            "GENERATOR_USER_PROMPT": cls.GENERATOR_USER_PROMPT,
            "REVIEWER_SYSTEM_PROMPT": cls.REVIEWER_SYSTEM_PROMPT,
            "REVIEWER_USER_PROMPT": cls.REVIEWER_USER_PROMPT,
            "OPTIMIZER_SYSTEM_PROMPT": cls.OPTIMIZER_SYSTEM_PROMPT,
            "OPTIMIZER_USER_PROMPT": cls.OPTIMIZER_USER_PROMPT,
            "RAG_CONTEXT_TEMPLATE": cls.RAG_CONTEXT_TEMPLATE,
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            if file_path.endswith('.json'):
                json.dump(config, f, ensure_ascii=False, indent=2)
            elif file_path.endswith(('.yaml', '.yml')):
                try:
                    import yaml
                    yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
                except ImportError:
                    raise ImportError("需要安装pyyaml来导出YAML配置文件")
            else:
                # 默认JSON格式
                json.dump(config, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def get_all_prompts(cls) -> dict:
        """
        获取所有提示词配置。
        
        Returns:
            包含所有提示词的字典
        """
        return {
            "GENERATOR_SYSTEM_PROMPT": cls.GENERATOR_SYSTEM_PROMPT,
            "GENERATOR_USER_PROMPT": cls.GENERATOR_USER_PROMPT,
            "REVIEWER_SYSTEM_PROMPT": cls.REVIEWER_SYSTEM_PROMPT,
            "REVIEWER_USER_PROMPT": cls.REVIEWER_USER_PROMPT,
            "OPTIMIZER_SYSTEM_PROMPT": cls.OPTIMIZER_SYSTEM_PROMPT,
            "OPTIMIZER_USER_PROMPT": cls.OPTIMIZER_USER_PROMPT,
            "RAG_CONTEXT_TEMPLATE": cls.RAG_CONTEXT_TEMPLATE,
        }
