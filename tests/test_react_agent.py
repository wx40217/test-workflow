import unittest

from langchain_core.messages import AIMessage

from config.settings import ModelConfig
from src.workflow.graph import TestCaseWorkflow
from src.workflow.nodes import NodeOutput
from src.workflow.react_agent import TestCaseReactAgent
from src.workflow.tools import ReactToolState, ReactWorkflowTools


class FakeNode:
    def __init__(self, outputs, split_mode="mixed"):
        self.outputs = list(outputs)
        self.config = ModelConfig(
            api_key="test",
            base_url="http://example.invalid",
            model_name="fake",
            test_case_split_mode=split_mode,
        )
        self.rag_interface = None
        self.calls = []

    def invoke(self, **kwargs):
        self.calls.append(kwargs)
        if not self.outputs:
            return NodeOutput("")
        value = self.outputs.pop(0)
        return value if isinstance(value, NodeOutput) else NodeOutput(value)


class FakeToolCallLLM:
    def __init__(self, tool_names):
        self.tool_names = list(tool_names)
        self.index = 0

    def bind_tools(self, tools):
        self.bound_tool_names = [tool.name for tool in tools]
        return self

    def invoke(self, messages):
        if self.index >= len(self.tool_names):
            return AIMessage(content="完成")
        name = self.tool_names[self.index]
        self.index += 1
        args = {"query": "登录安全"} if name == "retrieve_rag_context" else {}
        return AIMessage(
            content="",
            tool_calls=[{
                "name": name,
                "args": args,
                "id": f"call-{self.index}",
            }],
        )


class FakeMultiToolCallLLM:
    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return AIMessage(
            content="",
            tool_calls=[
                {"name": "generate_test_cases", "args": {}, "id": "call-1"},
                {"name": "review_test_cases", "args": {}, "id": "call-2"},
            ],
        )


class FakeNoToolBindingLLM:
    def invoke(self, messages):
        return AIMessage(content="不会被调用")


class FakeRAG:
    def is_enabled(self):
        return True

    def retrieve(self, query):
        return ["登录安全测试参考"]


def make_workflow(split_mode="mixed"):
    config = ModelConfig(
        api_key="test",
        base_url="http://example.invalid",
        model_name="fake",
        test_case_split_mode=split_mode,
    )
    return TestCaseWorkflow(
        generator_config=config,
        reviewer_config=config,
        optimizer_config=config,
        analyzer_config=config,
        agent_mode="react",
        max_agent_steps=10,
    )


class ReactAgentTests(unittest.TestCase):
    def test_react_agent_calls_three_tools_and_returns_result(self):
        workflow = make_workflow()
        workflow.generator = FakeNode(["## 用例\n**登录成功**"])
        workflow.reviewer = FakeNode(["无阻塞问题，通过"])
        workflow.optimizer = FakeNode(["## 最终用例\n**登录成功**"])

        llm = FakeToolCallLLM([
            "generate_test_cases",
            "review_test_cases",
            "optimize_test_cases",
        ])
        result = TestCaseReactAgent(workflow, llm=llm, max_steps=5).run(
            user_input="邮箱密码登录",
            output_format="markdown",
        )

        self.assertTrue(result.success)
        self.assertIn("登录成功", result.final_test_cases)
        self.assertEqual(result.metadata["agent_mode"], "react")
        self.assertEqual(
            result.metadata["tools_used"],
            ["generate_test_cases", "review_test_cases", "optimize_test_cases"],
        )
        self.assertTrue(result.metadata["validation_passed"])

    def test_react_agent_can_call_full_tool_whitelist(self):
        workflow = make_workflow(split_mode="frontend_backend")
        workflow.enable_analyzer = True
        workflow.analyzer_complexity_threshold = 0
        workflow.rag_interface = FakeRAG()
        workflow.analyzer = FakeNode(["分析结果"])
        workflow.generator = FakeNode(["初始用例"])
        workflow.reviewer = FakeNode(["无阻塞问题，通过"])
        workflow.optimizer = FakeNode([
            "缺少表格的最终内容",
            (
                "<table><tr><th>功能点</th><th>前端用例</th><th>后端用例</th></tr>"
                "<tr><td>登录</td><td>页面输入邮箱密码并点击登录</td>"
                "<td>校验密码错误次数并锁定账户</td></tr></table>"
            ),
        ], split_mode="frontend_backend")

        llm = FakeToolCallLLM([
            "analyze_requirements",
            "retrieve_rag_context",
            "generate_test_cases",
            "review_test_cases",
            "optimize_test_cases",
            "validate_output_structure",
            "repair_output_structure",
        ])
        result = TestCaseReactAgent(workflow, llm=llm, max_steps=10).run(
            user_input="邮箱密码登录，3次失败锁定账户",
            output_format="confluence",
        )

        self.assertTrue(result.success)
        self.assertEqual(result.metadata["tools_used"], [
            "analyze_requirements",
            "retrieve_rag_context",
            "generate_test_cases",
            "review_test_cases",
            "optimize_test_cases",
            "validate_output_structure",
            "repair_output_structure",
        ])
        generator_instructions = workflow.generator.calls[0]["additional_instructions"]
        self.assertIn("登录安全测试参考", generator_instructions)
        self.assertTrue(result.metadata["validation_passed"])

    def test_frontend_backend_validation_failure_can_be_repaired(self):
        workflow = make_workflow(split_mode="frontend_backend")
        workflow.generator = FakeNode(["初始用例"])
        workflow.reviewer = FakeNode(["无阻塞问题，通过"])
        workflow.optimizer = FakeNode([
            "缺少表格的最终内容",
            (
                "<table><tr><th>功能点</th><th>前端用例</th><th>后端用例</th></tr>"
                "<tr><td>登录</td><td>页面输入邮箱密码并点击登录</td>"
                "<td>校验密码错误次数并锁定账户</td></tr></table>"
            ),
        ], split_mode="frontend_backend")

        llm = FakeToolCallLLM([
            "generate_test_cases",
            "review_test_cases",
            "optimize_test_cases",
            "validate_output_structure",
            "repair_output_structure",
        ])
        result = TestCaseReactAgent(workflow, llm=llm, max_steps=8).run(
            user_input="邮箱密码登录，3次失败锁定账户",
            output_format="confluence",
        )

        self.assertTrue(result.success)
        self.assertIn("<table>", result.final_test_cases)
        self.assertTrue(result.metadata["validation_passed"])
        self.assertIn("repair_output_structure", result.metadata["tools_used"])

    def test_max_steps_terminates_without_fake_success(self):
        workflow = make_workflow()
        workflow.generator = FakeNode(["## 用例\n**登录成功**"])

        llm = FakeToolCallLLM(["generate_test_cases", "generate_test_cases"])
        result = TestCaseReactAgent(workflow, llm=llm, max_steps=1).run(
            user_input="邮箱密码登录",
            output_format="markdown",
        )

        self.assertFalse(result.success)
        self.assertIn("最大步数限制", "\n".join(result.errors))
        self.assertEqual(result.metadata["tools_used"], ["generate_test_cases"])

    def test_max_steps_limits_multiple_tool_calls_in_one_model_turn(self):
        workflow = make_workflow()
        workflow.generator = FakeNode(["## 用例\n**登录成功**"])

        result = TestCaseReactAgent(workflow, llm=FakeMultiToolCallLLM(), max_steps=1).run(
            user_input="邮箱密码登录",
            output_format="markdown",
        )

        self.assertFalse(result.success)
        self.assertEqual(len(result.metadata["agent_steps"]), 1)
        self.assertEqual(result.metadata["tools_used"], ["generate_test_cases"])
        self.assertIn("最大步数限制", "\n".join(result.errors))

    def test_auto_validation_does_not_exceed_max_steps(self):
        workflow = make_workflow(split_mode="frontend_backend")
        workflow.optimizer = FakeNode(["缺少表格"], split_mode="frontend_backend")

        result = TestCaseReactAgent(
            workflow,
            llm=FakeToolCallLLM(["generate_test_cases", "optimize_test_cases"]),
            max_steps=2,
        ).run(
            user_input="邮箱密码登录",
            output_format="confluence",
        )

        self.assertFalse(result.success)
        self.assertLessEqual(len(result.metadata["agent_steps"]), 2)
        self.assertNotIn("validate_output_structure", result.metadata["tools_used"])
        self.assertIn("最大步数限制", "\n".join(result.errors))

    def test_react_dependency_error_returns_workflow_result(self):
        workflow = make_workflow()
        workflow.generator._llm = FakeNoToolBindingLLM()

        result = workflow.run("邮箱密码登录", output_format="markdown")

        self.assertFalse(result.success)
        self.assertEqual(result.metadata["agent_mode"], "react")
        self.assertEqual(result.metadata["tools_used"], [])
        self.assertIn("不支持 bind_tools", "\n".join(result.errors))


class ReactWorkflowToolsMockTests(unittest.TestCase):
    def test_generate_review_optimize_tools_update_shared_state(self):
        workflow = make_workflow()
        workflow.generator = FakeNode(["## 用例\n**登录成功**"])
        workflow.reviewer = FakeNode(["无阻塞问题，通过"])
        workflow.optimizer = FakeNode(["## 最终用例\n**登录成功**"])
        state = ReactToolState(user_input="邮箱密码登录", output_format="markdown")
        tools = ReactWorkflowTools(workflow, state)

        generate_observation = tools.generate_test_cases()
        review_observation = tools.review_test_cases()
        optimize_observation = tools.optimize_test_cases()

        self.assertIn("status: ok", generate_observation)
        self.assertIn("status: ok", review_observation)
        self.assertIn("status: ok", optimize_observation)
        self.assertEqual(state.generated_test_cases, "## 用例\n**登录成功**")
        self.assertEqual(state.review_feedback, "无阻塞问题，通过")
        self.assertEqual(state.final_test_cases, "## 最终用例\n**登录成功**")
        self.assertFalse(state.review_has_blocking_issues)
        self.assertEqual(workflow.generator.calls[0]["user_input"], "邮箱密码登录")

    def test_tools_report_precondition_errors_without_calling_nodes(self):
        workflow = make_workflow()
        workflow.reviewer = FakeNode(["不应被调用"])
        workflow.optimizer = FakeNode(["不应被调用"])
        state = ReactToolState(user_input="邮箱密码登录")
        tools = ReactWorkflowTools(workflow, state)

        review_observation = tools.review_test_cases()
        optimize_observation = tools.optimize_test_cases()

        self.assertIn("status: error", review_observation)
        self.assertIn("缺少 generated_test_cases", review_observation)
        self.assertIn("status: error", optimize_observation)
        self.assertIn("缺少 generated_test_cases", optimize_observation)
        self.assertEqual(workflow.reviewer.calls, [])
        self.assertEqual(workflow.optimizer.calls, [])

    def test_retrieve_rag_context_and_generate_merge_rag_instructions(self):
        workflow = make_workflow()
        workflow.rag_interface = FakeRAG()
        workflow.generator = FakeNode(["## 用例\n**登录安全**"])
        state = ReactToolState(
            user_input="邮箱密码登录",
            additional_instructions="输出边界值",
        )
        tools = ReactWorkflowTools(workflow, state)

        rag_observation = tools.retrieve_rag_context("登录安全")
        tools.generate_test_cases()

        self.assertIn("登录安全测试参考", rag_observation)
        self.assertIn("登录安全测试参考", state.rag_context)
        generator_instructions = workflow.generator.calls[0]["additional_instructions"]
        self.assertIn("输出边界值", generator_instructions)
        self.assertIn("RAG 检索上下文", generator_instructions)
        self.assertIn("登录安全测试参考", generator_instructions)

    def test_validate_and_repair_frontend_backend_structure(self):
        workflow = make_workflow(split_mode="frontend_backend")
        workflow.optimizer = FakeNode([
            (
                "<table><tr><th>功能点</th><th>前端用例</th><th>后端用例</th></tr>"
                "<tr><td>登录</td><td>页面输入邮箱密码并点击登录</td>"
                "<td>校验密码错误次数并锁定账户</td></tr></table>"
            )
        ], split_mode="frontend_backend")
        state = ReactToolState(
            user_input="邮箱密码登录",
            generated_test_cases="初始用例",
            review_feedback="无阻塞问题，通过",
            final_test_cases="缺少表格的最终内容",
            output_format="confluence",
            validation_passed=False,
        )
        tools = ReactWorkflowTools(workflow, state)

        validation_observation = tools.validate_output_structure()
        repair_observation = tools.repair_output_structure()

        self.assertIn("status: error", validation_observation)
        self.assertFalse("缺少表格的最终内容" in state.validation_issues)
        self.assertIn("status: ok", repair_observation)
        self.assertTrue(state.validation_passed)
        self.assertIn("<table>", state.final_test_cases)


if __name__ == "__main__":
    unittest.main()
