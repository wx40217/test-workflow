import unittest

from src.workflow.quality import (
    QualityReport,
    build_quality_report,
    build_revision_plan,
    decide_next_step,
    validate_candidate_structure,
)


class QualityModuleTests(unittest.TestCase):
    def test_empty_candidate_fails_with_blocking_format_issue(self):
        report = build_quality_report(
            "",
            user_input="用户登录功能：支持邮箱密码登录",
            reviewer_feedback="无阻塞问题，通过",
        )

        self.assertFalse(report.passed)
        self.assertTrue(report.blocking)
        self.assertIn("输出为空。", report.format_issues)
        self.assertLess(report.score, 0.75)

    def test_requirement_keywords_create_coverage_issues(self):
        report = build_quality_report(
            "## 用例\n**登录成功**\n- 使用手机号登录。",
            user_input="用户登录功能：支持邮箱密码登录，3次失败锁定账户",
            reviewer_feedback="无阻塞问题，通过",
        )

        self.assertFalse(report.passed)
        self.assertTrue(any("邮箱" in issue for issue in report.coverage_issues))
        self.assertTrue(any("密码" in issue for issue in report.coverage_issues))
        self.assertTrue(any("3次失败锁定" in issue for issue in report.coverage_issues))

    def test_confluence_frontend_backend_validation_is_blocking(self):
        report = build_quality_report(
            "功能点 | 前端用例\n登录 | 页面输入邮箱密码",
            user_input="用户登录功能：支持邮箱密码登录",
            reviewer_feedback="无阻塞问题，通过",
            output_format="confluence",
            frontend_backend_mode=True,
        )

        self.assertFalse(report.passed)
        self.assertTrue(report.blocking)
        self.assertTrue(report.frontend_backend_issues)
        self.assertGreater(report.score_breakdown["deductions"]["frontend_backend"]["points"], 0)

    def test_revision_plan_preserves_issue_categories(self):
        report = QualityReport(
            coverage_issues=["缺少锁定场景"],
            format_issues=["Markdown 输出缺少列表"],
            frontend_backend_issues=["缺少后端用例列"],
            truncation_issues=["检测到模型输出可能被截断。"],
            redundancy_issues=["存在 2 行重复内容。"],
        )

        plan = build_revision_plan(report)

        self.assertEqual(plan, [
            "补齐覆盖度问题：缺少锁定场景",
            "修复格式问题：Markdown 输出缺少列表",
            "修复前后端分离结构问题：缺少后端用例列",
            "修复截断问题：检测到模型输出可能被截断。",
            "压缩重复或冗余内容：存在 2 行重复内容。",
        ])

    def test_decide_finalize_revise_need_info_and_max_rounds(self):
        passing = QualityReport(passed=True, score=0.9)
        self.assertEqual(
            decide_next_step(passing, review_round=0, max_review_rounds=2).route,
            "finalize",
        )

        failed = QualityReport(passed=False, coverage_issues=["缺少锁定场景"], score=0.7)
        self.assertEqual(
            decide_next_step(failed, review_round=0, max_review_rounds=2).route,
            "revise",
        )
        self.assertEqual(
            decide_next_step(failed, review_round=2, max_review_rounds=2).route,
            "max_rounds",
        )

        need_info = QualityReport(passed=False, coverage_issues=["需补充：锁定时长不明确"])
        decision = decide_next_step(need_info, review_round=0, max_review_rounds=2)
        self.assertEqual(decision.route, "need_info")
        self.assertTrue(decision.warnings)

    def test_validation_failed_forces_revision_plan_before_finalize(self):
        report = QualityReport(passed=True, score=0.9)

        decision = decide_next_step(
            report,
            review_round=0,
            max_review_rounds=2,
            validation_failed=True,
        )

        self.assertEqual(decision.route, "revise")
        self.assertIn("确定性结构校验失败项", decision.revision_plan[0])

    def test_validate_candidate_structure_filters_internal_metadata(self):
        validation = validate_candidate_structure(
            "## 用例\n**登录成功**\nmetadata: internal"
        )

        self.assertFalse(validation["passed"])
        self.assertTrue(any("metadata" in issue for issue in validation["issues"]))


if __name__ == "__main__":
    unittest.main()
