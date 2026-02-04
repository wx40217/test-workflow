"""
测试用例输出格式化器模块。

支持多种输出格式：
- Markdown: 标准markdown嵌套列表格式
- Confluence: Confluence wiki任务列表格式
"""

import re
from enum import Enum
from typing import Optional


class OutputFormat(Enum):
    """支持的输出格式。"""
    MARKDOWN = "markdown"
    CONFLUENCE = "confluence"


class OutputFormatter:
    """
    测试用例输出的格式化器。
    
    将测试用例转换为适合各种文档系统的不同格式。
    """
    
    def format(
        self,
        content: str,
        output_format: OutputFormat = OutputFormat.MARKDOWN
    ) -> str:
        """
        根据指定格式格式化内容。
        
        参数:
            content: 要格式化的测试用例内容
            output_format: 目标输出格式
            
        返回:
            格式化后的内容
        """
        if output_format == OutputFormat.CONFLUENCE:
            return self.to_confluence(content)
        else:
            return self.to_markdown(content)
    
    def to_markdown(self, content: str) -> str:
        """
        确保内容是正确的markdown格式。
        
        参数:
            content: 要格式化的内容
            
        返回:
            markdown格式化的内容
        """
        # 如果已经是markdown格式，直接返回
        if self._is_markdown_list(content):
            return content
        
        # 尝试从其他格式转换
        return self._normalize_markdown(content)
    
    def to_confluence(self, content: str) -> str:
        """
        将内容转换为Confluence任务列表格式。
        
        Confluence任务列表格式：
        * [ ] 顶层任务
        ** [ ] 子任务
        *** [ ] 子子任务
        
        参数:
            content: 要转换的内容
            
        返回:
            Confluence格式化的内容
        """
        lines = content.split('\n')
        result_lines = []
        
        for line in lines:
            stripped = line.rstrip()
            if not stripped:
                result_lines.append('')
                continue
            
            # 计算缩进级别
            leading_spaces = len(line) - len(line.lstrip())
            
            # 检测列表标记
            content_part = line.lstrip()
            
            # 检查markdown列表标记
            markdown_match = re.match(r'^[-*+]\s*(\[.\])?\s*(.*)$', content_part)
            if markdown_match:
                checkbox = markdown_match.group(1) or '[ ]'
                text = markdown_match.group(2)
                
                # 转换为confluence级别
                level = self._calculate_level(leading_spaces)
                stars = '*' * (level + 1)
                
                # 使用复选框格式
                if checkbox in ['[x]', '[X]']:
                    result_lines.append(f"{stars} [x] {text}")
                else:
                    result_lines.append(f"{stars} [ ] {text}")
            
            # 检查数字列表
            elif re.match(r'^\d+\.\s+(.*)$', content_part):
                match = re.match(r'^\d+\.\s+(.*)$', content_part)
                text = match.group(1)
                level = self._calculate_level(leading_spaces)
                stars = '*' * (level + 1)
                result_lines.append(f"{stars} [ ] {text}")
            
            # 非列表内容（标题等）
            else:
                # 检查是否是标题
                if content_part.startswith('#'):
                    header_match = re.match(r'^(#+)\s*(.*)$', content_part)
                    if header_match:
                        level = len(header_match.group(1))
                        text = header_match.group(2)
                        # 转换为Confluence标题
                        result_lines.append(f"h{level}. {text}")
                    else:
                        result_lines.append(stripped)
                else:
                    result_lines.append(stripped)
        
        return '\n'.join(result_lines)
    
    def _calculate_level(self, spaces: int, indent_size: int = 2) -> int:
        """
        根据缩进计算嵌套级别。
        
        参数:
            spaces: 前导空格数
            indent_size: 每个缩进级别的空格数
            
        返回:
            嵌套级别（从0开始）
        """
        return spaces // indent_size
    
    def _is_markdown_list(self, content: str) -> bool:
        """
        检查内容是否已经是markdown列表格式。
        
        参数:
            content: 要检查的内容
            
        返回:
            如果内容包含markdown列表标记则返回True
        """
        lines = content.split('\n')
        list_lines = 0
        for line in lines:
            stripped = line.strip()
            if stripped and re.match(r'^[-*+]\s+', stripped):
                list_lines += 1
        
        # 如果至少30%的非空行是列表项，则认为是markdown
        non_empty = sum(1 for line in lines if line.strip())
        return non_empty > 0 and (list_lines / non_empty) > 0.3
    
    def _normalize_markdown(self, content: str) -> str:
        """
        将内容规范化为正确的markdown格式。
        
        参数:
            content: 要规范化的内容
            
        返回:
            规范化的markdown内容
        """
        lines = content.split('\n')
        result_lines = []
        
        for line in lines:
            stripped = line.rstrip()
            if not stripped:
                result_lines.append('')
                continue
            
            # 保留markdown标题
            if stripped.lstrip().startswith('#'):
                result_lines.append(stripped)
                continue
            
            # 保留现有列表格式
            if re.match(r'^[\s]*[-*+]\s+', stripped):
                result_lines.append(stripped)
                continue
            
            # 保留数字列表
            if re.match(r'^[\s]*\d+\.\s+', stripped):
                result_lines.append(stripped)
                continue
            
            # 其他内容保持原样
            result_lines.append(stripped)
        
        return '\n'.join(result_lines)
    
    def extract_test_cases(self, content: str) -> list[dict]:
        """
        从格式化内容中提取结构化的测试用例数据。
        
        参数:
            content: 格式化的测试用例内容
            
        返回:
            包含title、level和children的测试用例字典列表
        """
        lines = content.split('\n')
        test_cases = []
        current_stack = []  # 用于跟踪层级的栈
        
        for line in lines:
            stripped = line.rstrip()
            if not stripped:
                continue
            
            # 根据缩进计算级别
            leading_spaces = len(line) - len(line.lstrip())
            level = self._calculate_level(leading_spaces)
            
            # 提取文本内容
            content_part = line.lstrip()
            
            # 移除列表标记
            text_match = re.match(r'^[-*+]\s*(\[.\])?\s*(.*)$', content_part)
            if text_match:
                text = text_match.group(2)
            elif re.match(r'^\d+\.\s+(.*)$', content_part):
                text = re.match(r'^\d+\.\s+(.*)$', content_part).group(1)
            else:
                # 跳过非列表项或作为顶层处理
                if content_part.startswith('#'):
                    continue
                text = content_part
                level = 0
            
            test_case = {
                'title': text,
                'level': level,
                'children': []
            }
            
            # 构建层级结构
            while current_stack and current_stack[-1]['level'] >= level:
                current_stack.pop()
            
            if current_stack:
                current_stack[-1]['children'].append(test_case)
            else:
                test_cases.append(test_case)
            
            current_stack.append(test_case)
        
        return test_cases
    
    def to_json_structure(self, content: str) -> list[dict]:
        """
        将测试用例转换为JSON可序列化的结构。
        
        参数:
            content: 测试用例内容
            
        返回:
            测试用例字典列表
        """
        return self.extract_test_cases(content)
    
    def from_json_structure(
        self,
        test_cases: list[dict],
        output_format: OutputFormat = OutputFormat.MARKDOWN,
        indent_size: int = 2
    ) -> str:
        """
        将JSON结构转换回格式化文本。
        
        参数:
            test_cases: 测试用例字典列表
            output_format: 目标输出格式
            indent_size: 每个缩进级别的空格数
            
        返回:
            格式化的测试用例文本
        """
        lines = []
        
        def process_case(case: dict, level: int = 0):
            indent = ' ' * (level * indent_size)
            title = case.get('title', '')
            
            if output_format == OutputFormat.CONFLUENCE:
                stars = '*' * (level + 1)
                lines.append(f"{stars} [ ] {title}")
            else:
                lines.append(f"{indent}- {title}")
            
            for child in case.get('children', []):
                process_case(child, level + 1)
        
        for case in test_cases:
            process_case(case)
        
        return '\n'.join(lines)
