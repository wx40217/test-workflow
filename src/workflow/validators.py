"""
前后端分离结构校验器。

用于在 frontend_backend 模式下校验输出结构是否符合 HTML 表格要求：
1. 必须包含 <table> 标签
2. 表头三列：功能点、前端用例、后端用例
3. 每行按功能点组织，<td> 数量为 3
4. 前后端列不混写明显跨端步骤
"""

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class SplitValidationResult:
    """前后端分离结构校验结果。"""
    is_valid: bool
    issues: List[str] = field(default_factory=list)
    repair_hint: str = ""


def validate_fe_be_structure(text: str) -> SplitValidationResult:
    """
    校验前后端分离 HTML 表格结构。

    参数:
        text: 测试用例文本

    返回:
        SplitValidationResult
    """
    issues: list[str] = []
    flags = re.DOTALL | re.IGNORECASE

    # 1. 检查 <table> / </table> 存在
    if not re.search(r"<table[\s>]", text, flags):
        issues.append("缺少 <table> 标签。")
    if "</table>" not in text.lower():
        issues.append("缺少 </table> 标签。")

    if issues:
        return SplitValidationResult(
            is_valid=False,
            issues=issues,
            repair_hint=_build_repair_hint(issues),
        )

    # 2. 提取 <th> 检查表头
    th_cells = re.findall(r"<th[^>]*>(.*?)</th>", text, flags)
    th_texts = [c.strip() for c in th_cells]

    expected_headers = ["功能点", "前端", "后端"]
    for expected in expected_headers:
        if not any(expected in th for th in th_texts):
            issues.append(f"表头缺少「{expected}」列。")

    # 3. 提取数据行（排除含 <th> 的表头行）
    all_rows = re.findall(r"<tr[^>]*>(.*?)</tr>", text, flags)
    data_rows = [row for row in all_rows if "<th" not in row.lower()]

    if not data_rows:
        issues.append("表格缺少数据行（<tr> 中无 <td> 内容）。")
    else:
        # 4. 逐行检查
        for idx, row in enumerate(data_rows, start=1):
            tds = re.findall(r"<td[^>]*>(.*?)</td>", row, flags)

            if len(tds) != 3:
                issues.append(f"第 {idx} 行 <td> 数量为 {len(tds)}，应为 3。")
                continue

            feature_name = tds[0].strip()
            if not feature_name:
                issues.append(f"第 {idx} 行功能点列为空。")

            # 5. 混写检查
            fe_cell = tds[1]
            be_cell = tds[2]
            cross_issues = _check_cross_boundary(fe_cell, be_cell)
            for ci in cross_issues:
                feature_label = feature_name if feature_name else f"第 {idx} 行"
                issues.append(f"[{feature_label}] {ci}")

    repair_hint = _build_repair_hint(issues)
    return SplitValidationResult(
        is_valid=len(issues) == 0,
        issues=issues,
        repair_hint=repair_hint,
    )


def _check_cross_boundary(fe_block: str, be_block: str) -> list[str]:
    """检查前后端列是否存在明显混写。"""
    issues: list[str] = []

    # 前端列中不应出现的后端实现词
    fe_forbidden = [
        "数据库", "落库", "事务", "幂等", "SQL", "数据表", "消息队列", "Redis", "缓存击穿"
    ]
    # 后端列中不应出现的前端交互词
    be_forbidden = [
        "点击", "页面", "按钮", "弹窗", "输入框", "下拉", "浏览器", "前端页面", "UI交互"
    ]

    fe_hit = [kw for kw in fe_forbidden if kw in fe_block]
    be_hit = [kw for kw in be_forbidden if kw in be_block]

    if fe_hit:
        issues.append(f"前端列疑似混入后端实现内容（关键词：{', '.join(fe_hit[:4])}）。")
    if be_hit:
        issues.append(f"后端列疑似混入前端交互内容（关键词：{', '.join(be_hit[:4])}）。")

    return issues


def _build_repair_hint(issues: list[str]) -> str:
    """构造修复提示。"""
    if not issues:
        return ""

    return (
        "请严格按以下 HTML 表格结构修复输出：\n"
        "1. 使用 <table> 标签，表头三列：功能点 | 前端用例 | 后端用例。\n"
        "2. 每个功能点占一行 <tr>，包含三个 <td>。\n"
        "3. 前端列仅保留 UI/交互/页面行为相关用例。\n"
        "4. 后端列仅保留接口/业务规则/数据一致性相关用例。\n"
        "5. 将混写用例拆分并移动到正确列。\n"
        "6. 若某功能点仅涉及单端，另一列填「无」。\n"
        "7. 保持原有覆盖度，不要删减有效测试场景。"
    )
