# CLAUDE.md

本文件为 AI 助手提供项目上下文信息，帮助更好地理解和协助开发此项目。

## 项目概述

这是一个基于 LangGraph 的测试用例生成工作流项目。通过三个 LLM 节点（生成器 -> 评审员 -> 优化器）协作，从用户输入的需求文档生成高质量的测试用例。

## 技术栈

- **Python 3.10+**
- **LangChain / LangGraph** - LLM 编排和工作流
- **Pydantic / Pydantic-Settings** - 配置管理和数据验证
- **python-dotenv** - 环境变量加载

## 项目结构

```
/workspace/
├── main.py                      # 主程序入口，CLI和编程接口
├── requirements.txt             # Python依赖
├── .env.example                 # 环境变量配置示例
├── config/
│   ├── settings.py              # 配置管理（自动加载.env）
│   ├── prompts.py               # 提示词模板（中文）
│   └── prompts_config.json      # 提示词JSON配置
├── src/
│   ├── workflow/
│   │   ├── nodes.py             # 三个LLM节点：Generator/Reviewer/Optimizer
│   │   └── graph.py             # LangGraph工作流定义
│   ├── input_handler/
│   │   └── handlers.py          # 多格式输入处理（docx/pdf/xlsx/pptx/image）
│   ├── output_formatter/
│   │   └── formatters.py        # 输出格式化（markdown/confluence）
│   └── rag/
│       └── interface.py         # RAG接口（预留扩展）
└── examples/
    └── basic_usage.py           # 使用示例
```

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 运行测试用例生成
python main.py --input "需求描述" --verbose

# 从文件生成
python main.py --file requirements.docx

# 交互模式
python main.py --interactive

# 查看帮助
python main.py --help
```

## 开发规范

### 代码风格

- 所有注释和文档字符串使用**中文**
- 提示词（prompts）使用**中文**
- 变量名和函数名使用英文
- 遵循 PEP 8 规范

### 文件修改注意事项

1. **config/settings.py** - 修改配置项时需同步更新 `.env.example`
2. **config/prompts.py** - 修改提示词时需同步更新 `prompts_config.json`
3. **src/workflow/graph.py** - 修改工作流时注意 `WorkflowState` 类型定义

### Git 规范

- 不要主动 commit 和 push（除非用户明确要求）
- 如果在 master 分支，产生代码改动前新建分支
- commit message 使用中文或英文均可

## 核心类和函数

### 主要入口

```python
# main.py
generate_test_cases(
    input_content: str,      # 输入内容（文本/文件路径）
    api_key: str,            # API密钥
    output_format: str,      # 输出格式：markdown/confluence
    ...
) -> WorkflowResult
```

### 工作流

```python
# src/workflow/graph.py
class TestCaseWorkflow:
    def run(input_source, ...) -> WorkflowResult  # 完整运行
    def run_step_by_step(...)  # 逐步运行，可获取中间结果
```

### 节点

```python
# src/workflow/nodes.py
class GeneratorNode(BaseNode)   # 生成初始测试用例
class ReviewerNode(BaseNode)    # 评审并提供反馈
class OptimizerNode(BaseNode)   # 根据反馈优化
```

### 配置

```python
# config/settings.py
settings = Settings()  # 全局配置实例，自动从.env加载
settings.get_generator_config() -> ModelConfig
settings.get_reviewer_config() -> ModelConfig
settings.get_optimizer_config() -> ModelConfig
```

### 提示词

```python
# config/prompts.py
PromptTemplates.get_generator_prompts(user_input, ...) -> (system, user)
PromptTemplates.get_reviewer_prompts(original_input, test_cases, ...) -> (system, user)
PromptTemplates.get_optimizer_prompts(original_input, initial_cases, feedback, ...) -> (system, user)
PromptTemplates.customize_prompt(name, template)  # 运行时自定义
PromptTemplates.load_from_file(path)  # 从文件加载
```

## 扩展点

### 添加新输入格式

在 `src/input_handler/handlers.py` 中：
1. 创建继承 `BaseHandler` 的新类
2. 实现 `can_handle()` 和 `process()` 方法
3. 在 `InputHandler.__init__` 中注册

### 添加新输出格式

在 `src/output_formatter/formatters.py` 中：
1. 在 `OutputFormat` 枚举中添加新格式
2. 实现 `to_xxx()` 转换方法
3. 在 `format()` 方法中添加分支

### 添加新工作流节点

1. 在 `src/workflow/nodes.py` 中创建新节点类
2. 在 `src/workflow/graph.py` 中添加节点和边
3. 更新 `WorkflowState` 添加新的状态字段

### 实现RAG

在 `src/rag/interface.py` 中已预留接口：
- `RAGInterface` - 主接口类
- `BaseVectorStore` - 向量存储抽象类
- `ChromaVectorStore` - Chroma实现（需安装chromadb）
- `InMemoryVectorStore` - 内存实现（用于测试）

## 环境变量

关键环境变量（在 `.env` 中配置）：

| 变量 | 说明 |
|------|------|
| `GENERATOR_API_KEY` | 生成器API密钥 |
| `GENERATOR_BASE_URL` | 生成器API地址 |
| `GENERATOR_MODEL_NAME` | 生成器模型名 |
| `REVIEWER_API_KEY` | 评审员API密钥 |
| `REVIEWER_MODEL_NAME` | 评审员模型名（建议用思考模型如o1） |
| `OPTIMIZER_API_KEY` | 优化器API密钥 |
| `OPTIMIZER_MODEL_NAME` | 优化器模型名 |

## 调试技巧

1. 使用 `--verbose` 参数查看详细执行过程
2. 使用 `run_step_by_step()` 方法获取中间结果
3. 检查 `WorkflowResult.errors` 获取错误信息
4. 配置较低的 `temperature` 获得更稳定的输出

## 已知限制

1. RAG 功能目前为预留接口，完整实现需要安装额外依赖
2. 图片输入需要使用支持视觉的模型（如 gpt-4o）
3. 评审节点使用 o1 等思考模型时，temperature 参数可能被忽略
