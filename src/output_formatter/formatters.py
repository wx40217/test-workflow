"""
Output formatters for test cases.

Supports multiple output formats:
- Markdown: Standard markdown nested list format
- Confluence: Confluence wiki task list format
"""

import re
from enum import Enum
from typing import Optional


class OutputFormat(Enum):
    """Supported output formats."""
    MARKDOWN = "markdown"
    CONFLUENCE = "confluence"


class OutputFormatter:
    """
    Formatter for test case output.
    
    Converts test cases to different formats suitable for
    various documentation systems.
    """
    
    def format(
        self,
        content: str,
        output_format: OutputFormat = OutputFormat.MARKDOWN
    ) -> str:
        """
        Format the content according to the specified format.
        
        Args:
            content: The test cases content to format
            output_format: Target output format
            
        Returns:
            Formatted content
        """
        if output_format == OutputFormat.CONFLUENCE:
            return self.to_confluence(content)
        else:
            return self.to_markdown(content)
    
    def to_markdown(self, content: str) -> str:
        """
        Ensure content is in proper markdown format.
        
        Args:
            content: The content to format
            
        Returns:
            Markdown formatted content
        """
        # If already in markdown format, return as-is
        if self._is_markdown_list(content):
            return content
        
        # Try to convert from other formats
        return self._normalize_markdown(content)
    
    def to_confluence(self, content: str) -> str:
        """
        Convert content to Confluence task list format.
        
        Confluence task list format:
        * [ ] Top level task
        ** [ ] Sub task
        *** [ ] Sub-sub task
        
        Args:
            content: The content to convert
            
        Returns:
            Confluence formatted content
        """
        lines = content.split('\n')
        result_lines = []
        
        for line in lines:
            stripped = line.rstrip()
            if not stripped:
                result_lines.append('')
                continue
            
            # Calculate indentation level
            leading_spaces = len(line) - len(line.lstrip())
            
            # Detect list markers
            content_part = line.lstrip()
            
            # Check for markdown list markers
            markdown_match = re.match(r'^[-*+]\s*(\[.\])?\s*(.*)$', content_part)
            if markdown_match:
                checkbox = markdown_match.group(1) or '[ ]'
                text = markdown_match.group(2)
                
                # Convert to confluence level
                level = self._calculate_level(leading_spaces)
                stars = '*' * (level + 1)
                
                # Use checkbox format
                if checkbox in ['[x]', '[X]']:
                    result_lines.append(f"{stars} [x] {text}")
                else:
                    result_lines.append(f"{stars} [ ] {text}")
            
            # Check for numbered list
            elif re.match(r'^\d+\.\s+(.*)$', content_part):
                match = re.match(r'^\d+\.\s+(.*)$', content_part)
                text = match.group(1)
                level = self._calculate_level(leading_spaces)
                stars = '*' * (level + 1)
                result_lines.append(f"{stars} [ ] {text}")
            
            # Non-list content (headers, etc.)
            else:
                # Check if it's a header
                if content_part.startswith('#'):
                    header_match = re.match(r'^(#+)\s*(.*)$', content_part)
                    if header_match:
                        level = len(header_match.group(1))
                        text = header_match.group(2)
                        # Convert to Confluence heading
                        result_lines.append(f"h{level}. {text}")
                    else:
                        result_lines.append(stripped)
                else:
                    result_lines.append(stripped)
        
        return '\n'.join(result_lines)
    
    def _calculate_level(self, spaces: int, indent_size: int = 2) -> int:
        """
        Calculate nesting level from indentation.
        
        Args:
            spaces: Number of leading spaces
            indent_size: Number of spaces per indent level
            
        Returns:
            Nesting level (0-based)
        """
        return spaces // indent_size
    
    def _is_markdown_list(self, content: str) -> bool:
        """
        Check if content is already in markdown list format.
        
        Args:
            content: Content to check
            
        Returns:
            True if content contains markdown list markers
        """
        lines = content.split('\n')
        list_lines = 0
        for line in lines:
            stripped = line.strip()
            if stripped and re.match(r'^[-*+]\s+', stripped):
                list_lines += 1
        
        # Consider it markdown if at least 30% of non-empty lines are list items
        non_empty = sum(1 for line in lines if line.strip())
        return non_empty > 0 and (list_lines / non_empty) > 0.3
    
    def _normalize_markdown(self, content: str) -> str:
        """
        Normalize content to proper markdown format.
        
        Args:
            content: Content to normalize
            
        Returns:
            Normalized markdown content
        """
        lines = content.split('\n')
        result_lines = []
        
        for line in lines:
            stripped = line.rstrip()
            if not stripped:
                result_lines.append('')
                continue
            
            # Keep markdown headers
            if stripped.lstrip().startswith('#'):
                result_lines.append(stripped)
                continue
            
            # Keep existing list format
            if re.match(r'^[\s]*[-*+]\s+', stripped):
                result_lines.append(stripped)
                continue
            
            # Keep numbered lists
            if re.match(r'^[\s]*\d+\.\s+', stripped):
                result_lines.append(stripped)
                continue
            
            # Keep other content as-is
            result_lines.append(stripped)
        
        return '\n'.join(result_lines)
    
    def extract_test_cases(self, content: str) -> list[dict]:
        """
        Extract structured test case data from formatted content.
        
        Args:
            content: Formatted test case content
            
        Returns:
            List of test case dictionaries with title, level, and children
        """
        lines = content.split('\n')
        test_cases = []
        current_stack = []  # Stack to track hierarchy
        
        for line in lines:
            stripped = line.rstrip()
            if not stripped:
                continue
            
            # Calculate level from indentation
            leading_spaces = len(line) - len(line.lstrip())
            level = self._calculate_level(leading_spaces)
            
            # Extract text content
            content_part = line.lstrip()
            
            # Remove list markers
            text_match = re.match(r'^[-*+]\s*(\[.\])?\s*(.*)$', content_part)
            if text_match:
                text = text_match.group(2)
            elif re.match(r'^\d+\.\s+(.*)$', content_part):
                text = re.match(r'^\d+\.\s+(.*)$', content_part).group(1)
            else:
                # Skip non-list items or treat as top-level
                if content_part.startswith('#'):
                    continue
                text = content_part
                level = 0
            
            test_case = {
                'title': text,
                'level': level,
                'children': []
            }
            
            # Build hierarchy
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
        Convert test cases to a JSON-serializable structure.
        
        Args:
            content: Test case content
            
        Returns:
            List of test case dictionaries
        """
        return self.extract_test_cases(content)
    
    def from_json_structure(
        self,
        test_cases: list[dict],
        output_format: OutputFormat = OutputFormat.MARKDOWN,
        indent_size: int = 2
    ) -> str:
        """
        Convert JSON structure back to formatted text.
        
        Args:
            test_cases: List of test case dictionaries
            output_format: Target output format
            indent_size: Number of spaces per indent level
            
        Returns:
            Formatted test case text
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
