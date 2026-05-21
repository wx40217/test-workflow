# 测试用例生成器工作流

基于 LangGraph 的智能测试用例生成工作流，通过多个 LLM 节点协作完成测试用例的生成、评审和优化。

## 目录

- [功能特性](#功能特性)
- [项目结构](#项目结构)
- [安装配置](#安装配置)
- [使用方法](#使用方法)
- [配置说明](#配置说明)
- [开发指南](#开发指南)
- [扩展开发](#扩展开发)

## 功能特性

- **多节点工作流**: 分析(可选) -> 生成 -> 评审 -> 优化，确保测试用例质量
- **四种执行模式**: `workflow` 线性工作流、`react` 工具调用 Agent、`quality-graph` 质量闭环状态图、`multi-agent-quality-graph` 多 Agent 质量图
- **智能需求分析**: 可选的需求分析节点，自动识别复杂需求并进行结构化分析
- **实时进度显示**: 执行过程中显示详细进度和耗时统计
- **多种输入支持**: 文本、Word、PDF、Excel、PowerPoint、图片
- **灵活的模型配置**: 每个节点可独立配置不同的模型
- **多种输出格式**: Markdown 嵌套列表、Confluence 任务列表
- **自动保存**: 支持自动保存到 outputs 目录
- **RAG 支持**: 预留知识库检索增强接口
- **质量门禁**: `quality-graph` 会生成结构化质量报告、质量分、修订计划和确定性校验结果
- **可配置提示词**: 支持自定义各节点的提示词

## 项目结构

```
.
├── main.py                      # 主程序入口
├── requirements.txt             # Python 依赖
├── .env.example                 # 环境变量配置示例
├── inputs/                      # 需求文件目录（txt/md/docx/pdf/图片等）
├── templates/                   # 测试模板目录（C端/B端等）
├── outputs/                     # 输出目录（生成的测试用例）
├── config/
│   ├── __init__.py
│   ├── settings.py              # 配置管理
│   ├── prompts.py               # 提示词模板
│   └── prompts_config.json      # 提示词配置文件
├── src/
│   ├── __init__.py
│   ├── workflow/
│   │   ├── __init__.py
│   │   ├── nodes.py             # LLM 节点定义（含分析器节点）
│   │   ├── graph.py             # 默认 workflow 编排
│   │   ├── react_agent.py       # ReAct 工具调用 Agent
│   │   ├── tools.py             # ReAct 白名单工具
│   │   ├── quality.py           # 确定性质量报告和路由判断
│   │   └── quality_graph.py     # quality-graph 质量闭环状态图
│   ├── input_handler/
│   │   ├── __init__.py
│   │   └── handlers.py          # 输入处理器
│   ├── output_formatter/
│   │   ├── __init__.py
│   │   └── formatters.py        # 输出格式化
│   └── rag/
│       ├── __init__.py
│       └── interface.py         # RAG 接口
└── examples/
    ├── basic_usage.py           # 基础 workflow 示例
    ├── react_usage.py           # react 模式示例
    └── quality_graph_usage.py   # quality-graph 模式示例
```

### 目录说明

| 目录 | 说明 |
|------|------|
| `inputs/` | 存放需求文件，支持 txt、md、docx、pdf、png、jpg 等格式 |
| `templates/` | 存放测试模板文件，如 C端测试模板、B端测试模板等 |
| `outputs/` | 生成的测试用例输出目录，使用 `--auto-save` 时自动保存到此目录 |

## 安装配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制示例配置文件：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的配置：

```env
# 推荐最简配置（三个节点继承同一 provider）
MODEL_PROVIDER=deepseek
MODEL_API_KEY=sk-your-api-key
MODEL_NAME=deepseek-v4-flash

# OpenAI-compatible 厂商需要额外配置
# MODEL_BASE_URL=https://your-compatible-endpoint/v1
```

### 3. 验证安装

```bash
python main.py --help
```

## 使用方法

### 方式一：命令行使用

**基本用法 - 文本输入：**

```bash
python main.py --input "用户登录功能：支持邮箱密码登录，3次失败锁定账户"

# 临时指定 DeepSeek V4
python main.py --input "用户登录功能：支持邮箱密码登录，3次失败锁定账户" \
  --provider deepseek \
  --generator-model deepseek-v4-flash
```

执行时会显示实时进度：

```
==================================================
  测试用例生成工作流
==================================================

[分析] - 未启用
[生成] 正在生成初始测试用例...
[生成] ✓ 完成
[评审] 正在评审测试用例...
[评审] ✓ 完成
[优化] 正在优化测试用例...
[优化] ✓ 完成

==================================================
  ✓ 生成完成 | 耗时: 45.2秒
  测试点数量: 18
==================================================
```

**文件输入：**

```bash
python main.py --file requirements.docx
# 或从 inputs 目录读取
python main.py --file inputs/login_requirement.md
```

**多文件输入：**

```bash
python main.py --files doc1.pdf doc2.docx screenshot.png
```

**自动保存到 outputs 目录：**

```bash
python main.py --input "用户登录功能" --auto-save
# 输出文件: outputs/20240204_153022_用户登录功能.md
```

**指定输出文件：**

```bash
python main.py --input "..." --output test_cases.md
```

**静默模式（不显示进度）：**

```bash
python main.py --input "..." --quiet
# 或
python main.py --input "..." -q
```

**指定输出格式：**

```bash
# Markdown 格式（默认）
python main.py --input "..." --format markdown

# Confluence 任务列表格式
python main.py --input "..." --format confluence
```

**使用自定义模型：**

```bash
python main.py --input "..." \
  --generator-model gpt-4o \
  --reviewer-model o1-preview \
  --optimizer-model gpt-4o
```

**交互模式：**

```bash
python main.py --interactive
```

### 四种执行模式

| 模式 | 适用场景 | 执行特点 | 关键参数 |
|------|----------|----------|----------|
| `workflow` | 默认稳定路径、常规需求、批量生成 | 固定顺序：分析（可选）-> 生成 -> 评审 -> 优化 | `--agent-mode workflow` |
| `react` | 需要模型按需选择分析、RAG、生成、评审、结构校验工具 | LangChain tool calling，受 `max_agent_steps` 限制 | `--agent-mode react --max-agent-steps 8` |
| `quality-graph` | 高质量交付、复杂需求、需要可解释质量门禁 | LangGraph 显式节点：生成、评审、质量评分、决策、修订、校验、收敛 | `--agent-mode quality-graph --max-review-rounds 3 --quality-threshold 0.8` |
| `multi-agent-quality-graph` | 需要职责分离、候选池、可观察共享状态和确定性质量门 | Orchestrator 协调 Planner、Retrieval、Generator、Reviewer、Optimizer、Validator、Finalizer，基于候选池和质量证据收敛 | `--agent-mode multi-agent-quality-graph --max-agent-rounds 2 --candidate-pool-size 5` |

```bash
# 默认线性工作流
python main.py --input "用户登录功能：邮箱密码登录，3次失败锁定" --agent-mode workflow

# ReAct 工具调用 Agent
python main.py --input "用户登录功能：邮箱密码登录，3次失败锁定" \
  --agent-mode react \
  --max-agent-steps 8 \
  --show-agent-trace

# 质量闭环状态图
python main.py --input "用户登录功能：邮箱密码登录，3次失败锁定" \
  --agent-mode quality-graph \
  --max-review-rounds 3 \
  --quality-threshold 0.8 \
  --show-agent-trace

# 多 Agent 质量图
python main.py --input "用户登录功能：邮箱密码登录，3次失败锁定" \
  --agent-mode multi-agent-quality-graph \
  --max-agent-rounds 2 \
  --candidate-pool-size 5 \
  --quality-threshold 0.8 \
  --show-agent-trace
```

### 方式二：编程调用

**基本调用：**

```python
from main import generate_test_cases

result = generate_test_cases(
    "用户登录功能需求...",
    api_key="sk-...",
    output_format="markdown"
)

print(result.final_test_cases)
```

**使用工作流对象：**

```python
from src.workflow.graph import TestCaseWorkflow, create_workflow
from config.settings import ModelConfig

# 方式1：使用工厂函数快速创建
workflow = create_workflow(
    api_key="sk-...",
    generator_model="gpt-4o",
    reviewer_model="o1-preview",
    optimizer_model="gpt-4o"
)

result = workflow.run("你的需求描述")

# 方式2：自定义配置
generator_config = ModelConfig(
    api_key="sk-...",
    base_url="https://api.openai.com/v1",
    model_name="gpt-4o",
    temperature=0.8,
    max_tokens=4096
)

workflow = TestCaseWorkflow(
    generator_config=generator_config,
    # ... 其他配置
)
```

**选择 Agent 模式：**

```python
from main import generate_test_cases

react_result = generate_test_cases(
    "用户登录功能：邮箱密码登录，3次失败锁定账户",
    api_key="sk-...",
    agent_mode="react",
    max_agent_steps=8,
    show_agent_trace=True,
)

quality_result = generate_test_cases(
    "退款功能：未发货退款、优惠券回滚、风控人工审核",
    api_key="sk-...",
    agent_mode="quality-graph",
    max_review_rounds=3,
    quality_threshold=0.8,
    show_agent_trace=True,
)

print(quality_result.metadata["quality_score"])
```

**独立示例：**

```bash
python examples/react_usage.py
python examples/quality_graph_usage.py
```

**逐步执行并获取中间结果：**

```python
workflow = create_workflow(api_key="sk-...")

for step, result in workflow.run_step_by_step("需求描述"):
    if step == "generated":
        print("初始测试用例:", result)
    elif step == "reviewed":
        print("评审反馈:", result)
    elif step == "completed":
        print("最终用例:", result)
```

### 方式三：启用 RAG 增强

```python
from main import generate_test_cases

# 知识库文档
knowledge_docs = [
    "测试用例编写规范：每个用例需包含前置条件...",
    "安全测试指南：需要测试 SQL 注入、XSS..."
]

result = generate_test_cases(
    "用户认证 API 需求...",
    api_key="sk-...",
    enable_rag=True,
    rag_documents=knowledge_docs
)
```

## 配置说明

### 环境变量配置

#### 工作流配置

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `ENABLE_ANALYZER` | 是否启用需求分析节点 | false |
| `ANALYZER_COMPLEXITY_THRESHOLD` | 复杂度阈值（1-5），满足几个指标时触发分析 | 2 |
| `AGENT_MODE` | 执行模式：`workflow`、`react`、`quality-graph` 或 `multi-agent-quality-graph` | workflow |
| `MAX_AGENT_STEPS` | ReAct Agent 最大工具调用步数 | 10 |
| `MAX_REVIEW_ROUNDS` | quality-graph 最大评审/修订轮次 | 1 |
| `MAX_AGENT_ROUNDS` | multi-agent-quality-graph 最大修订轮次 | 2 |
| `QUALITY_THRESHOLD` | quality-graph / multi-agent-quality-graph 质量通过阈值 | 0.75 |
| `CANDIDATE_POOL_SIZE` | multi-agent-quality-graph 候选池最大保留数量 | 5 |
| `STOP_ON_NO_IMPROVEMENT_ROUNDS` | multi-agent-quality-graph 连续无质量提升时提前停止轮数 | 2 |
| `SHOW_AGENT_TRACE` | 是否在详细输出中展示 Agent trace | false |

**复杂度指标说明：**
- 需求描述超过 200 字符
- 逗号数量超过 5 个（多个子需求）
- 包含复合逻辑词（并且、同时、以及、或者等）
- 包含问号（有不确定性）
- 换行数超过 3 个

#### 模型 Provider 配置

推荐优先只配置全局 `MODEL_*`，各节点未覆盖时会继承同一套 provider 预设。

```env
MODEL_PROVIDER=deepseek
MODEL_API_KEY=sk-...
MODEL_NAME=deepseek-v4-flash
```

支持的 provider：

| Provider | LangChain ChatModel | 默认行为 |
|----------|---------------------|----------|
| `openai` | `langchain_openai.ChatOpenAI` | 默认 base URL 为 `https://api.openai.com/v1`，启用 Responses API |
| `deepseek` | `langchain_deepseek.ChatDeepSeek` | 默认 base URL 为 `https://api.deepseek.com`，默认模型 `deepseek-v4-flash`，不使用 OpenAI Responses API |
| `openai-compatible` | `langchain_openai.ChatOpenAI` | 必须配置 `MODEL_BASE_URL`，不使用 Responses API |
| `anthropic` | `langchain_anthropic.ChatAnthropic` | 使用 Claude/Anthropic 协议，不传 OpenAI 专属参数 |

DeepSeek 说明：DeepSeek 官方已发布 V4，示例使用 `deepseek-v4-flash` / `deepseek-v4-pro`。旧 `deepseek-chat` / `deepseek-reasoner` 不再作为推荐值。V4 思考模式通过 `extra_body.thinking` 和 `reasoning_effort=high|max` 透传；ReAct 模式依赖 tool calling，如 provider 或模型能力不足会返回明确错误。

RAG embedding 暂不跟随 chat provider，仍保留现有 OpenAI embedding 配置。

#### 节点配置

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `MODEL_PROVIDER` | 全局模型 provider：`openai`、`deepseek`、`openai-compatible`、`anthropic` | openai |
| `MODEL_API_KEY` | 全局模型 API 密钥 | - |
| `MODEL_BASE_URL` | 全局 API 基础 URL；`openai-compatible` 必填 | provider 默认 |
| `MODEL_NAME` | 全局模型名称 | provider 默认 |
| `MODEL_SUPPORTS_TOOLS` | 显式声明模型是否支持 tool calling | provider 推断 |
| `ANALYZER_API_KEY` | 分析器节点的 API 密钥（未配置时继承全局配置） | - |
| `ANALYZER_BASE_URL` | 分析器 API 基础 URL | 继承全局配置 |
| `ANALYZER_MODEL_NAME` | 分析器使用的模型 | 继承全局配置 |
| `ANALYZER_TEMPERATURE` | 分析器采样温度 | 0.3 |
| `ANALYZER_MAX_TOKENS` | 分析器最大 token 数 | 4096 |
| `GENERATOR_API_KEY` | 生成器节点的 API 密钥 | 继承全局配置 |
| `GENERATOR_BASE_URL` | 生成器 API 基础 URL | 继承全局配置 |
| `GENERATOR_MODEL_NAME` | 生成器使用的模型 | 继承全局配置 |
| `GENERATOR_TEMPERATURE` | 生成器采样温度 | 0.7 |
| `GENERATOR_MAX_TOKENS` | 生成器最大 token 数 | 8192 |
| `REVIEWER_API_KEY` | 评审员节点的 API 密钥 | 继承全局配置 |
| `REVIEWER_BASE_URL` | 评审员 API 基础 URL | 继承全局配置 |
| `REVIEWER_MODEL_NAME` | 评审员使用的模型 | 继承全局配置 |
| `REVIEWER_TEMPERATURE` | 评审员采样温度 | 1.0 |
| `REVIEWER_MAX_TOKENS` | 评审员最大 token 数 | 8192 |
| `OPTIMIZER_API_KEY` | 优化器节点的 API 密钥 | 继承全局配置 |
| `OPTIMIZER_BASE_URL` | 优化器 API 基础 URL | 继承全局配置 |
| `OPTIMIZER_MODEL_NAME` | 优化器使用的模型 | 继承全局配置 |
| `OPTIMIZER_TEMPERATURE` | 优化器采样温度 | 0.5 |
| `OPTIMIZER_MAX_TOKENS` | 优化器最大 token 数 | 8192 |
| `TEST_CASE_SPLIT_MODE` | 用例分离模式：`mixed` 或 `frontend_backend` | mixed |
| `TEST_CASE_SPLIT_STRICT` | 分离模式下是否严格校验并自动修复结构 | true |
| `USE_RESPONSES_API` | 是否使用 OpenAI Responses API（仅 `openai` 默认启用） | provider 默认 |
| `REQUEST_TIMEOUT` | 请求超时时间（秒） | 120 |
| `RAG_ENABLED` | 是否启用 RAG | false |

### 提示词配置

**方式1：修改配置文件**

编辑 `config/prompts_config.json` 文件自定义提示词。

**方式2：环境变量覆盖**

```env
GENERATOR_SYSTEM_PROMPT=你的自定义系统提示词...
```

**方式3：代码中动态修改**

```python
from config.prompts import PromptTemplates

PromptTemplates.customize_prompt(
    "GENERATOR_SYSTEM_PROMPT",
    "你的自定义提示词..."
)
```

**方式4：从文件加载**

```python
PromptTemplates.load_from_file("my_prompts.json")
```

## 开发指南

### 开发环境设置

```bash
# 克隆仓库
git clone <repo-url>
cd test-workflow

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入配置
```

### 代码结构说明

**配置层 (`config/`)**
- `settings.py`: 使用 pydantic-settings 管理配置，自动从环境变量和 .env 文件加载
- `prompts.py`: 提示词模板管理，支持动态自定义

**工作流层 (`src/workflow/`)**
- `nodes.py`: 定义 LLM 节点（Analyzer、Generator、Reviewer、Optimizer）
- `graph.py`: 默认 workflow 编排和模式分发
- `react_agent.py` / `tools.py`: ReAct 模式和受限工具白名单
- `quality.py` / `quality_graph.py`: 确定性质量报告、路由决策和质量闭环状态图

**输入处理层 (`src/input_handler/`)**
- `handlers.py`: 处理各种输入类型，提取文本和图片

**输出格式化层 (`src/output_formatter/`)**
- `formatters.py`: 将测试用例转换为不同格式

**RAG 层 (`src/rag/`)**
- `interface.py`: RAG 接口，支持向量存储和检索

### 运行测试

```bash
# 运行示例
python examples/basic_usage.py
python examples/react_usage.py
python examples/quality_graph_usage.py

# 测试命令行
python main.py --input "测试需求" --verbose

# 运行单元测试
python -m unittest
```

## Benchmark

以下 5 条需求用于比较三种模式的预期收益和成本。表中结论基于当前实现的能力边界：`workflow` 固定三节点，`react` 依赖模型 tool calling 自主调度，`quality-graph` 具备确定性质量报告、修订循环和结构校验。

| # | 需求类型 | 推荐模式 | workflow 预期 | react 预期 | quality-graph 预期 | 结论 |
|---|----------|----------|---------------|------------|--------------------|------|
| 1 | 简单登录：邮箱密码登录、失败锁定 | `workflow` | 成本低，路径稳定 | 工具调度收益有限 | 质量报告可用但偏重 | 不需要多 Agent |
| 2 | 退款链路：退款、券/积分回滚、开票红冲、风控 | `quality-graph` | 容易一次性遗漏跨域场景 | 可按需分析和检索 | 可通过覆盖缺口触发修订 | 优先升级质量闭环 |
| 3 | 前后端分离输出：Confluence 表格要求前端/后端用例拆分 | `quality-graph` | 依赖模型自觉遵守格式 | 可调用校验/修复工具 | 内置确定性结构校验和修复路径 | 暂不需要更多 Agent，先强化校验 |
| 4 | 大型多模块需求：会员、订单、售后、通知混合 | `react` 或 `quality-graph` | 上下文和覆盖压力较大 | 能按需调用分析/RAG | 能循环收敛但仍是单图内修订 | 可作为多 Agent 候选 |
| 5 | 合规/安全测试：登录、权限、审计、风控证据 | `quality-graph` | 缺少可解释质量门 | 可按需检索安全资料 | 可记录质量分、缺口和决策路径 | 先用质量闭环，必要时再拆专家 Agent |

**是否继续升级多 Agent：** 目前不建议立即引入独立多 Agent 编排。当前 benchmark 只有第 4 类大型多模块需求明确显示潜在收益；第 2、3、5 类主要缺口可以由 `quality-graph` 的质量门、修订循环、结构校验和 RAG 补强覆盖。下一步应先积累真实运行结果，包括每条需求的质量分、修订轮次、工具调用数、耗时和人工返工点；只有当大型多模块需求持续出现跨域遗漏或单图修订振荡时，再升级为多 Agent。

## 扩展开发

### 添加新的输入类型处理器

1. 在 `src/input_handler/handlers.py` 中创建新的处理器类：

```python
class NewFormatHandler(BaseHandler):
    """新格式的处理器。"""
    
    def can_handle(self, input_source: Union[str, Path]) -> bool:
        path = Path(input_source)
        return path.suffix.lower() == '.newformat'
    
    def process(self, input_source: Union[str, Path, bytes]) -> ProcessedInput:
        # 实现处理逻辑
        pass
```

2. 在 `InputHandler.__init__` 中注册：

```python
self.handlers = [
    NewFormatHandler(),  # 添加新处理器
    DocxHandler(),
    # ...
]
```

### 添加新的输出格式

1. 在 `src/output_formatter/formatters.py` 中添加：

```python
class OutputFormat(Enum):
    MARKDOWN = "markdown"
    CONFLUENCE = "confluence"
    NEW_FORMAT = "new_format"  # 新格式

class OutputFormatter:
    def to_new_format(self, content: str) -> str:
        """转换为新格式。"""
        # 实现转换逻辑
        pass
    
    def format(self, content: str, output_format: OutputFormat) -> str:
        if output_format == OutputFormat.NEW_FORMAT:
            return self.to_new_format(content)
        # ...
```

### 添加新的工作流节点

1. 在 `src/workflow/nodes.py` 中添加节点：

```python
class NewNode(BaseNode):
    """新节点。"""
    
    def invoke(self, **kwargs) -> str:
        # 实现节点逻辑
        pass
```

2. 在 `src/workflow/graph.py` 中修改工作流图：

```python
def _build_graph(self) -> StateGraph:
    workflow = StateGraph(WorkflowState)
    
    # 添加新节点
    workflow.add_node("new_step", self._new_node)
    
    # 修改边
    workflow.add_edge("optimize", "new_step")
    workflow.add_edge("new_step", END)
    
    return workflow.compile()
```

### 实现完整的 RAG 功能

1. 安装向量数据库：

```bash
pip install chromadb
```

2. 配置 RAG：

```python
from src.rag.interface import RAGInterface, RAGConfig

rag_config = RAGConfig(
    enabled=True,
    vector_store_type="chroma",
    embedding_api_key="sk-...",
    collection_name="test_case_knowledge",
    top_k=5
)

rag = RAGInterface(rag_config)

# 添加知识库文档
rag.add_documents([
    "测试用例编写规范...",
    "历史测试用例示例..."
])

# 或从文件添加
rag.add_from_file("knowledge_base.pdf")
```

3. 在工作流中使用：

```python
workflow = TestCaseWorkflow(
    # ... 其他配置
    rag_interface=rag
)
```

### 自定义向量存储

实现 `BaseVectorStore` 接口：

```python
from src.rag.interface import BaseVectorStore, RetrievedDocument

class MyVectorStore(BaseVectorStore):
    def add_documents(self, documents, metadatas=None, ids=None):
        # 实现添加文档
        pass
    
    def search(self, query, top_k=5, filter_dict=None):
        # 实现搜索
        pass
    
    def delete(self, ids):
        # 实现删除
        pass
    
    def clear(self):
        # 实现清空
        pass
```

## 常见问题

**Q: 如何使用国内的 API 服务？**

修改 `.env` 中的 `BASE_URL`：

```env
GENERATOR_BASE_URL=https://your-api-provider.com/v1
```

**Q: 评审节点出错怎么办？**

如果评审模型（如 o1-preview）不可用，可以使用相同的模型：

```env
REVIEWER_MODEL_NAME=gpt-4o
```

**Q: 如何提高生成质量？**

1. 使用更详细的输入描述
2. 通过 `--instructions` 参数添加额外指示
3. 自定义提示词
4. 启用 RAG 添加领域知识

**Q: 处理大文档时超时怎么办？**

增加超时时间：

```env
REQUEST_TIMEOUT=300
```

## 路线图 (Roadmap)

### 已实现

- [x] 三节点工作流（生成 → 评审 → 优化）
- [x] 可选的需求分析节点（条件性启用）
- [x] 多种输入格式支持（文本、文件、图片）
- [x] 输出截断检测和警告
- [x] 实时进度显示
- [x] 自动保存到 outputs 目录
- [x] ReAct 工具调用 Agent 模式
- [x] quality-graph 质量闭环模式
- [x] quality.py 和 React tools 的 mock 单元测试

### 规划中

- [ ] **大需求智能拆分**（配置项已预留：`ENABLE_REQUIREMENT_SPLIT`）
  - 适用场景：需求字符数 > 1000，包含多个独立功能点
  - 实现思路：
    1. 需求分析节点识别功能点并判断关联性
    2. 高关联功能点 → 整体生成
    3. 低关联功能点 → 拆分并行生成 → 合并去重
  - 注意事项：跨功能的测试场景（如退款影响开票）需要特殊处理

- [ ] **C端/B端测试模板支持**
  - 根据需求类型自动加载对应的测试模板
  - 支持混合场景（同时涉及C端和B端）

- [x] **多轮评审优化**（quality-graph 通过 `MAX_REVIEW_ROUNDS` 控制）
  - 评审 → 优化循环，直到评审通过或达到最大轮次
  - 需要设置明确的退出条件避免振荡

- [ ] **历史用例学习（RAG增强）**
  - 从历史测试用例中学习风格和覆盖模式
  - 自动参考相似需求的测试用例



增加超时时间：

```env
REQUEST_TIMEOUT=300
```

## License

MIT License
