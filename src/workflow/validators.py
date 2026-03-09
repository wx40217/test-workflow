"""
前后端分离结构校验器。

用于在 frontend_backend 模式下校验输出结构是否符合要求：
1. 必须且仅包含两个一级分区：前端测试用例、后端测试用例
2. 前后端分区不混写明显跨端步骤
"""

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
    校验前后端分离结构。

    参数:
        text: 测试用例文本

    返回:
        SplitValidationResult
    """
    lines = text.splitlines()
    issues: list[str] = []

    fe_positions = [i for i, line in enumerate(lines) if "前端测试用例" in line]
    be_positions = [i for i, line in enumerate(lines) if "后端测试用例" in line]

    if not fe_positions:
        issues.append("缺少“前端测试用例”分区。")
    if not be_positions:
        issues.append("缺少“后端测试用例”分区。")

    if len(fe_positions) > 1:
        issues.append("“前端测试用例”分区出现多次，应仅保留一个。")
    if len(be_positions) > 1:
        issues.append("“后端测试用例”分区出现多次，应仅保留一个。")

    fe_idx = fe_positions[0] if fe_positions else -1
    be_idx = be_positions[0] if be_positions else -1
    if fe_idx != -1 and be_idx != -1 and fe_idx > be_idx:
        issues.append("分区顺序错误：应先“前端测试用例”，后“后端测试用例”。")

    # 检查分区内是否有内容
    if fe_idx != -1 and be_idx != -1:
        if not _has_content(lines[fe_idx + 1:be_idx]):
            issues.append("“前端测试用例”分区为空。")
        if not _has_content(lines[be_idx + 1:]):
            issues.append("“后端测试用例”分区为空。")

    # 混写检查（仅在两个分区都存在时执行）
    if fe_idx != -1 and be_idx != -1 and fe_idx < be_idx:
        fe_block = "\n".join(lines[fe_idx + 1:be_idx])
        be_block = "\n".join(lines[be_idx + 1:])
        issues.extend(_check_cross_boundary(fe_block, be_block))

    repair_hint = _build_repair_hint(issues)
    return SplitValidationResult(
        is_valid=len(issues) == 0,
        issues=issues,
        repair_hint=repair_hint
    )


def _has_content(lines: list[str]) -> bool:
    """判断分区是否有有效内容。"""
    for line in lines:
        stripped = line.strip()
        if stripped and "测试用例" not in stripped:
            return True
    return False


def _check_cross_boundary(fe_block: str, be_block: str) -> list[str]:
    """检查前后端分区是否存在明显混写。"""
    issues: list[str] = []

    # 前端分区中不应出现的后端实现词
    fe_forbidden = [
        "数据库", "落库", "事务", "幂等", "SQL", "数据表", "消息队列", "Redis", "缓存击穿"
    ]
    # 后端分区中不应出现的前端交互词
    be_forbidden = [
        "点击", "页面", "按钮", "弹窗", "输入框", "下拉", "浏览器", "前端页面", "UI交互"
    ]

    fe_hit = [kw for kw in fe_forbidden if kw in fe_block]
    be_hit = [kw for kw in be_forbidden if kw in be_block]

    if fe_hit:
        issues.append(f"前端分区疑似混入后端实现内容（关键词：{', '.join(fe_hit[:4])}）。")
    if be_hit:
        issues.append(f"后端分区疑似混入前端交互内容（关键词：{', '.join(be_hit[:4])}）。")

    return issues


def _build_repair_hint(issues: list[str]) -> str:
    """构造修复提示。"""
    if not issues:
        return ""

    return (
        "请严格按以下结构修复输出：\n"
        "1. 顶层必须且仅包含两个分区：**前端测试用例**、**后端测试用例**。\n"
        "2. 前端分区仅保留 UI/交互/页面行为相关用例。\n"
        "3. 后端分区仅保留接口/业务规则/数据一致性相关用例。\n"
        "4. 将混写用例拆分并移动到正确分区。\n"
        "5. 保持原有覆盖度，不要删减有效测试场景。"
    )
