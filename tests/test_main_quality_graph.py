import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import main
from src.workflow.graph import WorkflowResult


class FakeQualityWorkflow:
    def __init__(self):
        self.calls = []
        self.detailed = False
        self.analyzer = type("Node", (), {"stream_to_console": False})()
        self.generator = type("Node", (), {"stream_to_console": False})()
        self.reviewer = type("Node", (), {"stream_to_console": False})()
        self.optimizer = type("Node", (), {"stream_to_console": False})()

    def run(self, input_content, additional_instructions="", output_format="markdown"):
        self.calls.append({
            "input_content": input_content,
            "additional_instructions": additional_instructions,
            "output_format": output_format,
        })
        return WorkflowResult(
            success=True,
            final_test_cases="## 用例\n**登录成功**\n- 邮箱密码登录。\n**失败锁定**\n- 连续3次失败后锁定账户。",
            generated_test_cases="## 草稿",
            review_feedback="无阻塞问题，通过",
            errors=[],
            metadata={
                "agent_mode": "quality-graph",
                "review_rounds": 1,
                "quality_passed": True,
                "quality_score": 0.9,
                "validation_reports": [{"passed": True, "issues": []}],
                "validation_passed": True,
                "agent_trace": [{"node": "finalize", "detail": "done"}],
            },
        )


class MainQualityGraphTests(unittest.TestCase):
    def test_generate_test_cases_passes_quality_graph_options_to_factory(self):
        captured = {}
        fake_workflow = FakeQualityWorkflow()

        def fake_create_workflow(**kwargs):
            captured.update(kwargs)
            return fake_workflow

        with patch("main.create_workflow", side_effect=fake_create_workflow):
            result = main.generate_test_cases(
                "用户登录功能：支持邮箱密码登录，3次失败锁定账户",
                api_key="test-key",
                verbose=False,
                agent_mode="quality-graph",
                max_review_rounds=3,
                quality_threshold=0.82,
                show_agent_trace=True,
            )

        self.assertTrue(result.success)
        self.assertEqual(captured["agent_mode"], "quality-graph")
        self.assertEqual(captured["max_review_rounds"], 3)
        self.assertEqual(captured["quality_threshold"], 0.82)
        self.assertEqual(fake_workflow.calls[0]["output_format"], "markdown")
        self.assertNotIn("agent_trace", result.final_test_cases)
        self.assertNotIn("quality_report", result.final_test_cases)

    def test_cli_quiet_quality_graph_does_not_print_trace_or_metadata(self):
        fake_result = WorkflowResult(
            success=True,
            final_test_cases="## 用例\n**登录成功**",
            metadata={
                "agent_mode": "quality-graph",
                "agent_trace": [{"node": "finalize"}],
                "quality_score": 0.9,
            },
        )
        argv = [
            "main.py",
            "--input",
            "用户登录功能：支持邮箱密码登录，3次失败锁定账户",
            "--agent-mode",
            "quality-graph",
            "--max-review-rounds",
            "3",
            "--quality-threshold",
            "0.82",
            "--show-agent-trace",
            "--quiet",
        ]
        captured_kwargs = {}

        def fake_generate_test_cases(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return fake_result

        stdout = io.StringIO()
        with patch("sys.argv", argv), patch("main.generate_test_cases", side_effect=fake_generate_test_cases):
            with redirect_stdout(stdout):
                with self.assertRaises(SystemExit) as exit_ctx:
                    main.main()

        self.assertEqual(exit_ctx.exception.code, 0)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(captured_kwargs["agent_mode"], "quality-graph")
        self.assertEqual(captured_kwargs["max_review_rounds"], 3)
        self.assertEqual(captured_kwargs["quality_threshold"], 0.82)
        self.assertTrue(captured_kwargs["show_agent_trace"])
        self.assertFalse(captured_kwargs["verbose"])

    def test_cli_default_quality_graph_prints_final_cases_without_metadata(self):
        fake_result = WorkflowResult(
            success=True,
            final_test_cases="## 用例\n**登录成功**",
            metadata={
                "agent_mode": "quality-graph",
                "agent_trace": [{"node": "finalize"}],
                "quality_score": 0.9,
                "quality_report": {"score": 0.9},
            },
        )
        argv = [
            "main.py",
            "--input",
            "用户登录功能：支持邮箱密码登录，3次失败锁定账户",
            "--agent-mode",
            "quality-graph",
        ]

        with patch("sys.argv", argv), patch("main.generate_test_cases", return_value=fake_result):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                with self.assertRaises(SystemExit) as exit_ctx:
                    main.main()

        self.assertEqual(exit_ctx.exception.code, 0)
        output = stdout.getvalue()
        self.assertIn("最终测试用例", output)
        self.assertIn("**登录成功**", output)
        self.assertNotIn("agent_trace", output)
        self.assertNotIn("quality_score", output)
        self.assertNotIn("quality_report", output)
        self.assertNotIn("Quality Graph Trace", output)

    def test_show_agent_trace_prints_quality_graph_trace_when_verbose(self):
        fake_workflow = FakeQualityWorkflow()

        with patch("main.create_workflow", return_value=fake_workflow):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = main.generate_test_cases(
                    "用户登录功能：支持邮箱密码登录，3次失败锁定账户",
                    api_key="test-key",
                    verbose=True,
                    agent_mode="quality-graph",
                    show_agent_trace=True,
                )

        self.assertTrue(result.success)
        output = stdout.getvalue()
        self.assertIn("Quality Graph Trace", output)
        self.assertIn("quality_score", output)
        self.assertIn("finalize", output)


if __name__ == "__main__":
    unittest.main()
