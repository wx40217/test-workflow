import unittest

from config.settings import ModelConfig, settings
from src.workflow.graph import TestCaseWorkflow, create_workflow
from src.workflow.nodes import NodeOutput
from src.workflow.quality import build_quality_report, decide_next_step
from src.workflow.quality_graph import QualityGraphWorkflow


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


class FailingNode(FakeNode):
    def __init__(self, message="依赖能力不可用", split_mode="mixed"):
        super().__init__([], split_mode=split_mode)
        self.message = message

    def invoke(self, **kwargs):
        self.calls.append(kwargs)
        raise RuntimeError(self.message)


class FakeRAG:
    def __init__(self):
        self.queries = []

    def is_enabled(self):
        return True

    def retrieve(self, query):
        self.queries.append(query)
        return ["登录安全测试参考"]


def make_workflow(agent_mode="quality-graph", split_mode="mixed", max_review_rounds=2):
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
        agent_mode=agent_mode,
        max_review_rounds=max_review_rounds,
        quality_threshold=0.75,
    )


class QualityReportTests(unittest.TestCase):
    def test_quality_report_contains_structured_issue_categories(self):
        report = build_quality_report(
            "当然，以下是用例",
            user_input="用户登录功能：支持邮箱密码登录，3次失败锁定账户",
            reviewer_feedback="遗漏：未覆盖3次失败锁定账户",
            output_format="markdown",
            node_warnings=["生成器输出被截断（达到 max_tokens 限制）"],
            quality_threshold=0.75,
        )

        self.assertFalse(report.passed)
        self.assertTrue(report.blocking)
        self.assertTrue(report.coverage_issues)
        self.assertTrue(report.format_issues)
        self.assertTrue(report.truncation_issues)
        self.assertLess(report.score, 0.75)
        self.assertIn("deductions", report.score_breakdown)
        self.assertGreater(report.score_breakdown["deductions"]["coverage"]["points"], 0)
        self.assertGreater(report.score_breakdown["deductions"]["format"]["points"], 0)
        self.assertGreater(report.score_breakdown["deductions"]["truncation"]["points"], 0)

    def test_quality_report_flags_bracket_labels_and_metadata_fields(self):
        report = build_quality_report(
            "## 用例\n**登录成功**\n- 邮箱密码登录。\n[Trace]\nmetadata: internal",
            user_input="用户登录功能：支持邮箱密码登录",
            reviewer_feedback="无阻塞问题，通过",
            output_format="markdown",
            quality_threshold=0.75,
        )

        self.assertFalse(report.passed)
        self.assertTrue(report.blocking)
        self.assertTrue(any("metadata" in issue or "方括号" in issue for issue in report.format_issues))

    def test_decide_routes_to_revise_and_max_rounds(self):
        report = build_quality_report(
            "## 用例\n**登录成功**",
            user_input="用户登录功能：支持邮箱密码登录，3次失败锁定账户",
            reviewer_feedback="遗漏：未覆盖3次失败锁定账户",
            output_format="markdown",
            quality_threshold=0.75,
        )

        decision = decide_next_step(report, review_round=0, max_review_rounds=2)
        self.assertEqual(decision.route, "revise")
        self.assertTrue(decision.revision_plan)

        stopped = decide_next_step(report, review_round=2, max_review_rounds=2)
        self.assertEqual(stopped.route, "max_rounds")
        self.assertIn("最大评审轮次", stopped.warnings[0])

    def test_decide_can_route_need_info_with_assumption(self):
        report = build_quality_report(
            "## 用例\n**登录成功**\n- 校验邮箱密码登录。",
            user_input="用户登录功能",
            reviewer_feedback="需补充：锁定策略不明确，证据不足",
            output_format="markdown",
            quality_threshold=0.75,
        )

        decision = decide_next_step(report, review_round=0, max_review_rounds=2)
        self.assertEqual(decision.route, "need_info")
        self.assertTrue(decision.warnings)


class QualityGraphWorkflowTests(unittest.TestCase):
    def test_default_agent_mode_remains_workflow(self):
        self.assertEqual(settings.agent_mode, "workflow")

        config = ModelConfig(
            api_key="test",
            base_url="http://example.invalid",
            model_name="fake",
        )
        workflow = TestCaseWorkflow(
            generator_config=config,
            reviewer_config=config,
            optimizer_config=config,
            analyzer_config=config,
        )

        self.assertEqual(workflow.agent_mode, "workflow")

    def test_agent_mode_rejects_unknown_values(self):
        config = ModelConfig(
            api_key="test",
            base_url="http://example.invalid",
            model_name="fake",
        )

        with self.assertRaisesRegex(ValueError, "workflow、react 或 quality-graph"):
            TestCaseWorkflow(
                generator_config=config,
                reviewer_config=config,
                optimizer_config=config,
                analyzer_config=config,
                agent_mode="unknown",
            )

    def test_create_workflow_accepts_quality_graph_options(self):
        workflow = create_workflow(
            api_key="test",
            base_url="http://example.invalid",
            generator_model="fake",
            reviewer_model="fake",
            optimizer_model="fake",
            agent_mode="quality-graph",
            max_review_rounds=4,
            quality_threshold=0.66,
        )

        self.assertEqual(workflow.agent_mode, "quality-graph")
        self.assertEqual(workflow.max_review_rounds, 4)
        self.assertEqual(workflow.quality_threshold, 0.66)

    def test_quality_graph_revises_then_validates_and_finalizes(self):
        workflow = make_workflow(max_review_rounds=2)
        workflow.generator = FakeNode(["## 用例\n**登录成功**\n- 邮箱密码登录。"])
        workflow.reviewer = FakeNode([
            "遗漏：未覆盖3次失败锁定账户",
            "无阻塞问题，通过",
        ])
        workflow.optimizer = FakeNode([
            "## 用例\n**登录成功**\n- 邮箱密码登录。\n**失败锁定**\n- 连续3次失败后锁定账户。"
        ])

        result = workflow.run("用户登录功能：支持邮箱密码登录，3次失败锁定账户")

        self.assertTrue(result.success)
        self.assertIn("失败锁定", result.final_test_cases)
        self.assertEqual(result.metadata["agent_mode"], "quality-graph")
        self.assertEqual(result.metadata["review_rounds"], 1)
        self.assertTrue(result.metadata["quality_passed"])
        self.assertGreaterEqual(result.metadata["quality_score"], 0.75)
        self.assertIn("score_breakdown", result.metadata["quality_report"])
        self.assertTrue(result.metadata["validation_passed"])
        trace_nodes = [item["node"] for item in result.metadata["agent_trace"]]
        self.assertIn("evaluate", trace_nodes)
        self.assertIn("revise", trace_nodes)
        self.assertIn("validate", trace_nodes)
        self.assertNotIn("quality_report", result.final_test_cases)

    def test_quality_graph_need_info_continues_with_assumption_and_records_next_info(self):
        workflow = make_workflow(max_review_rounds=2)
        workflow.generator = FakeNode(["## 用例\n**登录成功**\n- 邮箱密码登录。"])
        workflow.reviewer = FakeNode([
            "需补充：3次失败后的锁定时长不明确，证据不足",
            "无阻塞问题，通过",
        ])
        workflow.optimizer = FakeNode([
            "## 用例\n**登录成功**\n- 邮箱密码登录。\n**失败锁定**\n- 假设锁定时长按系统默认策略处理，连续3次失败后锁定账户。"
        ])

        result = workflow.run("用户登录功能：支持邮箱密码登录，3次失败锁定账户")

        self.assertTrue(result.success)
        self.assertIn("失败锁定", result.final_test_cases)
        self.assertTrue(result.metadata["next_required_information"])
        self.assertIn("常见业务假设", result.metadata["next_required_information"][0])
        self.assertIn("证据不足", "\n".join(result.errors))
        optimizer_feedback = workflow.optimizer.calls[0]["review_feedback"]
        self.assertIn("继续修订所依据的假设或待补信息", optimizer_feedback)

    def test_truncated_generation_fails_quality_gate_and_revises(self):
        workflow = make_workflow(max_review_rounds=2)
        workflow.generator = FakeNode([
            NodeOutput(
                "## 用例\n**登录成功**\n- 邮箱密码登录。",
                is_truncated=True,
                truncation_warning="生成器输出被截断（达到 max_tokens 限制）",
            )
        ])
        workflow.reviewer = FakeNode([
            "无阻塞问题，通过",
            "无阻塞问题，通过",
        ])
        workflow.optimizer = FakeNode([
            "## 用例\n**登录成功**\n- 邮箱密码登录。\n**失败锁定**\n- 连续3次失败后锁定账户。"
        ])

        result = workflow.run("用户登录功能：支持邮箱密码登录，3次失败锁定账户")

        self.assertTrue(result.success)
        self.assertIn("失败锁定", result.final_test_cases)
        self.assertIn("生成器输出被截断", "\n".join(result.errors))
        trace_nodes = [item["node"] for item in result.metadata["agent_trace"]]
        self.assertIn("revise", trace_nodes)
        optimizer_feedback = workflow.optimizer.calls[0]["review_feedback"]
        self.assertIn("修复截断问题", optimizer_feedback)

    def test_internal_metadata_in_candidate_routes_to_revise_and_final_is_clean(self):
        workflow = make_workflow(max_review_rounds=2)
        workflow.generator = FakeNode([
            "## 用例\n**登录成功**\n- 邮箱密码登录。\n[Trace]\nmetadata: internal"
        ])
        workflow.reviewer = FakeNode([
            "无阻塞问题，通过",
            "无阻塞问题，通过",
        ])
        workflow.optimizer = FakeNode([
            "## 用例\n**登录成功**\n- 邮箱密码登录。\n**失败锁定**\n- 连续3次失败后锁定账户。"
        ])

        result = workflow.run("用户登录功能：支持邮箱密码登录，3次失败锁定账户")

        self.assertTrue(result.success)
        self.assertIn("失败锁定", result.final_test_cases)
        self.assertNotIn("[Trace]", result.final_test_cases)
        self.assertNotIn("metadata", result.final_test_cases)
        trace_nodes = [item["node"] for item in result.metadata["agent_trace"]]
        self.assertIn("revise", trace_nodes)
        optimizer_feedback = workflow.optimizer.calls[0]["review_feedback"]
        self.assertIn("修复格式问题", optimizer_feedback)

    def test_max_rounds_returns_best_candidate_with_warning(self):
        workflow = make_workflow(max_review_rounds=1)
        good_initial = "## 用例\n**登录成功**\n- 邮箱密码登录。"
        worse_revision = "坏"
        workflow.generator = FakeNode([good_initial])
        workflow.reviewer = FakeNode([
            "遗漏：未覆盖3次失败锁定账户",
            "遗漏：未覆盖邮箱密码登录和3次失败锁定账户",
        ])
        workflow.optimizer = FakeNode([worse_revision])

        result = workflow.run("用户登录功能：支持邮箱密码登录，3次失败锁定账户")

        self.assertIn("登录成功", result.final_test_cases)
        self.assertNotEqual(result.final_test_cases, worse_revision)
        self.assertEqual(result.metadata["decision"]["route"], "max_rounds")
        self.assertTrue(result.metadata["validation_reports"])
        last_validation = result.metadata["validation_reports"][-1]
        self.assertEqual(last_validation["candidate_source"], "best_candidate")
        self.assertIn("登录成功", last_validation["candidate_preview"])
        self.assertNotIn(worse_revision, last_validation["candidate_preview"])
        trace_nodes = [item["node"] for item in result.metadata["agent_trace"]]
        self.assertIn("validate", trace_nodes)
        self.assertIn("最大评审轮次", "\n".join(result.errors))

    def test_dependency_failure_returns_best_candidate_with_warning_and_next_info(self):
        workflow = make_workflow(max_review_rounds=1)
        initial = "## 用例\n**登录成功**\n- 邮箱密码登录。"
        workflow.generator = FakeNode([initial])
        workflow.reviewer = FakeNode([
            "遗漏：未覆盖3次失败锁定账户",
            "遗漏：未覆盖3次失败锁定账户",
        ])
        workflow.optimizer = FailingNode("优化模型不可用")

        result = workflow.run("用户登录功能：支持邮箱密码登录，3次失败锁定账户")

        self.assertTrue(result.success)
        self.assertEqual(result.final_test_cases, initial)
        self.assertIn("修订失败", "\n".join(result.errors))
        self.assertIn("优化模型不可用", "\n".join(result.errors))
        self.assertTrue(result.metadata["next_required_information"])
        self.assertIn("修订/优化模型能力", result.metadata["next_required_information"][0])

    def test_frontend_backend_validation_failure_can_be_fixed(self):
        workflow = make_workflow(split_mode="frontend_backend", max_review_rounds=2)
        workflow.generator = FakeNode(["## 用例\n**登录**\n- 邮箱密码登录。"], split_mode="frontend_backend")
        workflow.reviewer = FakeNode([
            "无阻塞问题，通过",
            "无阻塞问题，通过",
        ], split_mode="frontend_backend")
        workflow.optimizer = FakeNode([
            (
                "<table><tr><th>功能点</th><th>前端用例</th><th>后端用例</th></tr>"
                "<tr><td>登录</td><td>页面输入邮箱密码并点击登录</td>"
                "<td>校验密码错误次数并锁定账户</td></tr></table>"
            )
        ], split_mode="frontend_backend")

        result = workflow.run("用户登录功能：支持邮箱密码登录，3次失败锁定账户", output_format="confluence")

        self.assertTrue(result.success)
        self.assertIn("<table>", result.final_test_cases)
        self.assertTrue(result.metadata["validation_passed"])
        self.assertGreaterEqual(result.metadata["review_rounds"], 1)

    def test_frontend_backend_validation_failure_at_max_rounds_returns_best_candidate(self):
        workflow = make_workflow(split_mode="frontend_backend", max_review_rounds=1)
        initial = (
            "<table><tr><th>功能点</th><th>前端用例</th><th>后端用例</th></tr>"
            "<tr><td>登录</td><td>页面输入邮箱密码并点击登录</td>"
            "<td>校验密码错误次数并锁定账户</td></tr>"
        )
        workflow.generator = FakeNode([initial], split_mode="frontend_backend")
        workflow.reviewer = FakeNode([
            "遗漏：需补充异常提示",
            "遗漏：需补充异常提示",
        ], split_mode="frontend_backend")
        workflow.optimizer = FakeNode(["没有表格的坏修订"], split_mode="frontend_backend")

        result = workflow.run("用户登录功能：支持邮箱密码登录，3次失败锁定账户", output_format="confluence")

        self.assertTrue(result.success)
        self.assertEqual(result.final_test_cases, initial)
        self.assertFalse(result.metadata["validation_passed"])
        self.assertTrue(result.metadata["validation_reports"])
        last_validation = result.metadata["validation_reports"][-1]
        self.assertEqual(last_validation["candidate_source"], "best_candidate")
        self.assertTrue(last_validation["issues"])
        self.assertIn("确定性校验失败", "\n".join(result.errors))

    def test_default_workflow_mode_is_unchanged(self):
        workflow = make_workflow(agent_mode="workflow")
        workflow.generator = FakeNode(["## 初始\n**登录成功**"])
        workflow.reviewer = FakeNode(["无阻塞问题，通过"])
        workflow.optimizer = FakeNode(["## 最终\n**登录成功**"])

        result = workflow.run("用户登录功能：支持邮箱密码登录")

        self.assertTrue(result.success)
        self.assertEqual(result.metadata["agent_mode"], "workflow")
        self.assertNotIn("quality_report", result.metadata)

    def test_quality_graph_uses_rag_as_controlled_step(self):
        workflow = make_workflow(max_review_rounds=1)
        workflow.rag_interface = FakeRAG()
        workflow.generator = FakeNode(["## 用例\n**登录成功**\n- 邮箱密码登录并校验失败锁定。"])
        workflow.reviewer = FakeNode(["无阻塞问题，通过"])
        workflow.optimizer = FakeNode([])

        result = workflow.run("用户登录功能：支持邮箱密码登录")

        self.assertTrue(result.success)
        self.assertEqual(len(workflow.rag_interface.queries), 1)
        generator_instructions = workflow.generator.calls[0]["additional_instructions"]
        self.assertIn("登录安全测试参考", generator_instructions)
        trace_nodes = [item["node"] for item in result.metadata["agent_trace"]]
        self.assertIn("retrieve_rag_context", trace_nodes)

    def test_quality_graph_compiles_with_memory_checkpointer(self):
        workflow = make_workflow(max_review_rounds=1)
        agent = QualityGraphWorkflow(workflow)

        self.assertIsNotNone(agent.checkpointer)
        self.assertIs(agent._graph.checkpointer, agent.checkpointer)


if __name__ == "__main__":
    unittest.main()
