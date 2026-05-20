import unittest

from config.settings import ModelConfig, settings
from src.workflow.candidate_pool import CandidatePool
from src.workflow.graph import TestCaseWorkflow, create_workflow
from src.workflow.multi_agent_graph import MultiAgentQualityGraphWorkflow
from src.workflow.nodes import NodeOutput


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


class FakeRAG:
    def __init__(self):
        self.queries = []

    def is_enabled(self):
        return True

    def retrieve(self, query):
        self.queries.append(query)
        return ["登录安全测试参考", "账户锁定边界测试参考"]


def make_workflow(agent_mode="multi-agent-quality-graph", split_mode="mixed", max_agent_rounds=2):
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
        max_agent_rounds=max_agent_rounds,
        quality_threshold=0.75,
        candidate_pool_size=4,
        stop_on_no_improvement_rounds=2,
    )


class CandidatePoolTests(unittest.TestCase):
    def test_candidate_pool_keeps_quality_evidence_and_best_candidate(self):
        pool = CandidatePool(max_size=3)
        first = pool.add_candidate("## 用例\n**登录成功**", source_agent="Generator Agent", round=0, created_at_step="generator")
        second = pool.add_candidate(
            "## 用例\n**登录成功**\n**失败锁定**\n- 连续3次失败后锁定账户。",
            source_agent="Optimizer Agent",
            round=1,
            created_at_step="optimizer",
        )

        pool.update_quality(first.id, quality_score=0.5, is_valid=True)
        pool.update_quality(second.id, quality_score=0.9, is_valid=True, review_summary={"coverage_gaps": []})

        self.assertEqual(pool.best().id, second.id)
        self.assertEqual(pool.best().review_summary["coverage_gaps"], [])
        self.assertEqual(len(pool.to_list()), 2)


class MultiAgentQualityGraphTests(unittest.TestCase):
    def test_mode_is_accepted_without_changing_default(self):
        self.assertEqual(settings.agent_mode, "workflow")

        workflow = make_workflow()

        self.assertEqual(workflow.agent_mode, "multi-agent-quality-graph")
        self.assertEqual(workflow.max_agent_rounds, 2)

    def test_unknown_mode_error_mentions_multi_agent_mode(self):
        config = ModelConfig(api_key="test", base_url="http://example.invalid", model_name="fake")

        with self.assertRaisesRegex(ValueError, "multi-agent-quality-graph"):
            TestCaseWorkflow(
                generator_config=config,
                reviewer_config=config,
                optimizer_config=config,
                analyzer_config=config,
                agent_mode="unknown",
            )

    def test_create_workflow_accepts_multi_agent_options(self):
        workflow = create_workflow(
            api_key="test",
            base_url="http://example.invalid",
            generator_model="fake",
            reviewer_model="fake",
            optimizer_model="fake",
            agent_mode="multi-agent-quality-graph",
            max_agent_rounds=3,
            candidate_pool_size=7,
            stop_on_no_improvement_rounds=4,
            quality_threshold=0.81,
        )

        self.assertEqual(workflow.agent_mode, "multi-agent-quality-graph")
        self.assertEqual(workflow.max_agent_rounds, 3)
        self.assertEqual(workflow.candidate_pool_size, 7)
        self.assertEqual(workflow.stop_on_no_improvement_rounds, 4)
        self.assertEqual(workflow.quality_threshold, 0.81)

    def test_first_candidate_can_pass_directly_to_finalizer(self):
        workflow = make_workflow(max_agent_rounds=2)
        workflow.generator = FakeNode([
            "## 用例\n**登录成功**\n- 邮箱密码登录。\n**失败锁定**\n- 连续3次失败后锁定账户。"
        ])
        workflow.reviewer = FakeNode(["无阻塞问题，通过"])
        workflow.optimizer = FakeNode([])

        result = workflow.run("用户登录功能：支持邮箱密码登录，3次失败锁定账户")

        self.assertTrue(result.success)
        self.assertTrue(result.metadata["quality_passed"])
        self.assertTrue(result.metadata["validation_passed"])
        self.assertEqual(result.metadata["agent_rounds"], 0)
        self.assertEqual(result.metadata["candidate_count"], 1)
        self.assertIn("Planner Agent", result.metadata["active_agents"])
        self.assertIn("Orchestrator", result.metadata["active_agents"])
        self.assertIn("Quality Gate", result.metadata["active_agents"])
        self.assertIn("requirement_analysis", result.metadata)
        self.assertIn("retrieval_context", result.metadata)
        self.assertNotIn("agent_trace", result.final_test_cases)
        self.assertNotIn("quality_report", result.final_test_cases)
        trace_agents = [item["agent"] for item in result.metadata["agent_trace"]]
        self.assertIn("Reviewer Agent", trace_agents)
        self.assertIn("Validator", trace_agents)
        self.assertIn("Finalizer Agent", trace_agents)

    def test_reviewer_gap_triggers_optimizer_and_second_round_passes(self):
        workflow = make_workflow(max_agent_rounds=2)
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
        self.assertTrue(result.metadata["quality_passed"])
        self.assertEqual(result.metadata["agent_rounds"], 1)
        self.assertEqual(result.metadata["candidate_count"], 2)
        self.assertIn("失败锁定", result.final_test_cases)
        first_review = result.metadata["review_reports"][0]
        self.assertTrue(first_review["blocking_issues"])
        optimizer_feedback = workflow.optimizer.calls[0]["review_feedback"]
        self.assertIn("多 Agent 修订计划", optimizer_feedback)
        self.assertIn("Validator 失败证据", optimizer_feedback)
        first_decision = result.metadata["orchestrator_decisions"][0]
        self.assertTrue(any("补齐覆盖度问题" in item for item in first_decision["revision_plan"]))

    def test_validator_failure_cannot_be_overridden_by_reviewer(self):
        workflow = make_workflow(max_agent_rounds=1)
        workflow.generator = FakeNode([
            "当然，以下是用例\n## 用例\n**登录成功**\n- 邮箱密码登录并覆盖3次失败锁定账户。"
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
        self.assertTrue(result.metadata["quality_passed"])
        self.assertTrue(result.metadata["validation_reports"][0]["issues"])
        self.assertTrue(result.metadata["validation_passed"])
        first_decision = result.metadata["orchestrator_decisions"][0]
        self.assertIn("输出包含开场白", first_decision["revision_plan"][-1])
        self.assertIn("失败锁定", result.final_test_cases)

    def test_validator_records_basic_structure_failure_as_deterministic_evidence(self):
        workflow = make_workflow(max_agent_rounds=1)
        workflow.generator = FakeNode(["坏"])
        workflow.reviewer = FakeNode(["无阻塞问题，通过", "无阻塞问题，通过"])
        workflow.optimizer = FakeNode([
            "## 用例\n**登录成功**\n- 邮箱密码登录。\n**失败锁定**\n- 连续3次失败后锁定账户。"
        ])

        result = workflow.run("用户登录功能：支持邮箱密码登录，3次失败锁定账户")

        first_validation = result.metadata["validation_reports"][0]
        self.assertFalse(first_validation["passed"])
        self.assertTrue(any("过短" in issue or "Markdown" in issue for issue in first_validation["issues"]))
        self.assertTrue(result.metadata["validation_passed"])

    def test_frontend_backend_validator_drives_fix(self):
        workflow = make_workflow(split_mode="frontend_backend", max_agent_rounds=2)
        workflow.generator = FakeNode(["## 用例\n**登录**\n- 邮箱密码登录。"], split_mode="frontend_backend")
        workflow.reviewer = FakeNode(["无阻塞问题，通过", "无阻塞问题，通过"], split_mode="frontend_backend")
        workflow.optimizer = FakeNode([
            (
                "<table><tr><th>功能点</th><th>前端用例</th><th>后端用例</th></tr>"
                "<tr><td>登录</td><td>页面输入邮箱密码并点击登录</td>"
                "<td>校验密码错误次数并锁定账户</td></tr></table>"
            )
        ], split_mode="frontend_backend")

        result = workflow.run("用户登录功能：支持邮箱密码登录，3次失败锁定账户", output_format="confluence")

        self.assertTrue(result.success)
        self.assertTrue(result.metadata["quality_passed"])
        self.assertTrue(result.metadata["validation_reports"][0]["issues"])
        self.assertTrue(result.metadata["validation_passed"])
        self.assertIn("<table>", result.final_test_cases)

    def test_max_rounds_returns_best_candidate_with_failure_evidence(self):
        workflow = make_workflow(max_agent_rounds=1)
        initial = "## 用例\n**登录成功**\n- 邮箱密码登录。"
        workflow.generator = FakeNode([initial])
        workflow.reviewer = FakeNode([
            "遗漏：未覆盖3次失败锁定账户",
            "遗漏：未覆盖3次失败锁定账户",
        ])
        workflow.optimizer = FakeNode(["坏"])

        result = workflow.run("用户登录功能：支持邮箱密码登录，3次失败锁定账户")

        self.assertTrue(result.success)
        self.assertFalse(result.metadata["quality_passed"])
        self.assertEqual(result.metadata["agent_rounds"], 1)
        self.assertIn("登录成功", result.final_test_cases)
        self.assertEqual(result.metadata["orchestrator_decision"]["route"], "finalizer")
        self.assertIn("最大多 Agent 修订轮次", "\n".join(result.errors))
        self.assertTrue(result.metadata["validation_reports"])
        self.assertGreaterEqual(result.metadata["candidate_count"], 2)

    def test_no_improvement_stops_early_and_returns_best_candidate(self):
        workflow = make_workflow(max_agent_rounds=4)
        workflow.stop_on_no_improvement_rounds = 1
        initial = "## 用例\n**登录成功**\n- 邮箱密码登录。"
        workflow.generator = FakeNode([initial])
        workflow.reviewer = FakeNode([
            "遗漏：未覆盖3次失败锁定账户",
            "遗漏：未覆盖3次失败锁定账户",
        ])
        workflow.optimizer = FakeNode(["坏"])

        result = workflow.run("用户登录功能：支持邮箱密码登录，3次失败锁定账户")

        self.assertTrue(result.success)
        self.assertFalse(result.metadata["quality_passed"])
        self.assertEqual(result.metadata["no_improvement_rounds"], 1)
        self.assertIn("候选质量连续多轮未提升", "\n".join(result.errors))
        self.assertIn("登录成功", result.final_test_cases)

    def test_retrieval_agent_records_context_without_polluting_final_output(self):
        workflow = make_workflow(max_agent_rounds=1)
        workflow.rag_interface = FakeRAG()
        workflow.generator = FakeNode([
            "## 用例\n**登录成功**\n- 邮箱密码登录。\n**失败锁定**\n- 连续3次失败后锁定账户。"
        ])
        workflow.reviewer = FakeNode(["无阻塞问题，通过"])
        workflow.optimizer = FakeNode([])

        result = workflow.run("用户登录功能：支持邮箱密码登录，3次失败锁定账户")

        self.assertTrue(result.success)
        self.assertEqual(len(workflow.rag_interface.queries), 1)
        self.assertIn("登录安全测试参考", workflow.generator.calls[0]["additional_instructions"])
        self.assertEqual(result.metadata["retrieval_context"]["source_summary"], "检索到 2 条参考资料。")
        self.assertNotIn("登录安全测试参考", result.final_test_cases)

    def test_multi_agent_graph_compiles_with_memory_checkpointer(self):
        workflow = make_workflow(max_agent_rounds=1)
        agent = MultiAgentQualityGraphWorkflow(workflow)

        self.assertIsNotNone(agent.checkpointer)
        self.assertIs(agent._graph.checkpointer, agent.checkpointer)


if __name__ == "__main__":
    unittest.main()
