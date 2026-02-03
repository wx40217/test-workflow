"""
Input handlers for processing various input types.

Supported input types:
- Plain text
- Word documents (.docx)
- PDF files (.pdf)
- Excel files (.xlsx)
- PowerPoint files (.pptx)
- Images (with base64 encoding for vision models)
- Multiple files/documents
"""

import base64
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union

try:
    import chardet
except ImportError:
    chardet = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from openpyxl import load_workbook
except ImportError:
    load_workbook = None

try:
    from pptx import Presentation
except ImportError:
    Presentation = None

try:
    from PIL import Image
except ImportError:
    Image = None


class InputType(Enum):
    """Enumeration of supported input types."""
    TEXT = "text"
    DOCX = "docx"
    PDF = "pdf"
    XLSX = "xlsx"
    PPTX = "pptx"
    IMAGE = "image"
    UNKNOWN = "unknown"


@dataclass
class ProcessedInput:
    """
    Container for processed input data.
    
    Attributes:
        input_type: Type of the original input
        text_content: Extracted text content
        images: List of base64-encoded images (for vision models)
        metadata: Additional metadata about the input
        source: Original source (file path or 'direct_text')
    """
    input_type: InputType
    text_content: str
    images: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    source: str = "direct_text"
    
    def has_images(self) -> bool:
        """Check if the input contains images."""
        return len(self.images) > 0
    
    def get_combined_content(self) -> str:
        """Get all text content combined."""
        return self.text_content


@dataclass
class MultiInput:
    """
    Container for multiple processed inputs.
    
    Used when the user provides multiple files or mixed input types.
    """
    inputs: list[ProcessedInput] = field(default_factory=list)
    
    def add_input(self, processed_input: ProcessedInput) -> None:
        """Add a processed input to the collection."""
        self.inputs.append(processed_input)
    
    def get_combined_text(self) -> str:
        """Get all text content from all inputs combined."""
        texts = []
        for i, inp in enumerate(self.inputs):
            header = f"\n--- Document {i + 1} ({inp.source}) ---\n"
            texts.append(header + inp.text_content)
        return "\n".join(texts)
    
    def get_all_images(self) -> list[dict]:
        """Get all images from all inputs."""
        images = []
        for inp in self.inputs:
            images.extend(inp.images)
        return images
    
    def has_images(self) -> bool:
        """Check if any input contains images."""
        return any(inp.has_images() for inp in self.inputs)


class BaseHandler(ABC):
    """Abstract base class for input handlers."""
    
    @abstractmethod
    def can_handle(self, input_source: Union[str, Path]) -> bool:
        """Check if this handler can process the given input."""
        pass
    
    @abstractmethod
    def process(self, input_source: Union[str, Path, bytes]) -> ProcessedInput:
        """Process the input and return processed data."""
        pass


class TextHandler(BaseHandler):
    """Handler for plain text input."""
    
    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.rst', '.csv', '.json', '.xml', '.yaml', '.yml'}
    
    def can_handle(self, input_source: Union[str, Path]) -> bool:
        if isinstance(input_source, str) and not os.path.exists(input_source):
            # Treat as direct text input
            return True
        
        path = Path(input_source)
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    def process(self, input_source: Union[str, Path, bytes]) -> ProcessedInput:
        if isinstance(input_source, bytes):
            # Detect encoding
            if chardet:
                detected = chardet.detect(input_source)
                encoding = detected.get('encoding', 'utf-8')
            else:
                encoding = 'utf-8'
            text = input_source.decode(encoding, errors='replace')
            return ProcessedInput(
                input_type=InputType.TEXT,
                text_content=text,
                source="bytes_input"
            )
        
        if isinstance(input_source, str) and not os.path.exists(input_source):
            # Direct text input
            return ProcessedInput(
                input_type=InputType.TEXT,
                text_content=input_source,
                source="direct_text"
            )
        
        # File input
        path = Path(input_source)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            # Try with detected encoding
            with open(path, 'rb') as f:
                raw = f.read()
            if chardet:
                detected = chardet.detect(raw)
                encoding = detected.get('encoding', 'utf-8')
            else:
                encoding = 'utf-8'
            text = raw.decode(encoding, errors='replace')
        
        return ProcessedInput(
            input_type=InputType.TEXT,
            text_content=text,
            source=str(path),
            metadata={"file_name": path.name, "file_size": path.stat().st_size}
        )


class DocxHandler(BaseHandler):
    """Handler for Word documents (.docx)."""
    
    def can_handle(self, input_source: Union[str, Path]) -> bool:
        if DocxDocument is None:
            return False
        path = Path(input_source)
        return path.suffix.lower() == '.docx'
    
    def process(self, input_source: Union[str, Path, bytes]) -> ProcessedInput:
        if DocxDocument is None:
            raise ImportError("python-docx is required for processing .docx files")
        
        path = Path(input_source)
        doc = DocxDocument(path)
        
        # Extract text from paragraphs
        paragraphs = []
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text)
        
        # Extract text from tables
        table_texts = []
        for table in doc.tables:
            table_content = []
            for row in table.rows:
                row_content = [cell.text.strip() for cell in row.cells]
                table_content.append(" | ".join(row_content))
            table_texts.append("\n".join(table_content))
        
        text_content = "\n\n".join(paragraphs)
        if table_texts:
            text_content += "\n\n--- Tables ---\n" + "\n\n".join(table_texts)
        
        return ProcessedInput(
            input_type=InputType.DOCX,
            text_content=text_content,
            source=str(path),
            metadata={
                "file_name": path.name,
                "paragraph_count": len(paragraphs),
                "table_count": len(doc.tables)
            }
        )


class PdfHandler(BaseHandler):
    """Handler for PDF files (.pdf)."""
    
    def can_handle(self, input_source: Union[str, Path]) -> bool:
        if PdfReader is None:
            return False
        path = Path(input_source)
        return path.suffix.lower() == '.pdf'
    
    def process(self, input_source: Union[str, Path, bytes]) -> ProcessedInput:
        if PdfReader is None:
            raise ImportError("pypdf is required for processing PDF files")
        
        path = Path(input_source)
        reader = PdfReader(path)
        
        pages_text = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                pages_text.append(f"--- Page {i + 1} ---\n{page_text}")
        
        text_content = "\n\n".join(pages_text)
        
        return ProcessedInput(
            input_type=InputType.PDF,
            text_content=text_content,
            source=str(path),
            metadata={
                "file_name": path.name,
                "page_count": len(reader.pages)
            }
        )


class ExcelHandler(BaseHandler):
    """Handler for Excel files (.xlsx)."""
    
    def can_handle(self, input_source: Union[str, Path]) -> bool:
        if load_workbook is None:
            return False
        path = Path(input_source)
        return path.suffix.lower() in {'.xlsx', '.xls'}
    
    def process(self, input_source: Union[str, Path, bytes]) -> ProcessedInput:
        if load_workbook is None:
            raise ImportError("openpyxl is required for processing Excel files")
        
        path = Path(input_source)
        wb = load_workbook(path, data_only=True)
        
        sheets_text = []
        for sheet_name in wb.sheetnames:
            sheet = wb[sheet_name]
            rows = []
            for row in sheet.iter_rows(values_only=True):
                row_values = [str(cell) if cell is not None else "" for cell in row]
                if any(v.strip() for v in row_values):
                    rows.append(" | ".join(row_values))
            
            if rows:
                sheets_text.append(f"--- Sheet: {sheet_name} ---\n" + "\n".join(rows))
        
        text_content = "\n\n".join(sheets_text)
        
        return ProcessedInput(
            input_type=InputType.XLSX,
            text_content=text_content,
            source=str(path),
            metadata={
                "file_name": path.name,
                "sheet_count": len(wb.sheetnames),
                "sheet_names": wb.sheetnames
            }
        )


class PowerPointHandler(BaseHandler):
    """Handler for PowerPoint files (.pptx)."""
    
    def can_handle(self, input_source: Union[str, Path]) -> bool:
        if Presentation is None:
            return False
        path = Path(input_source)
        return path.suffix.lower() == '.pptx'
    
    def process(self, input_source: Union[str, Path, bytes]) -> ProcessedInput:
        if Presentation is None:
            raise ImportError("python-pptx is required for processing PowerPoint files")
        
        path = Path(input_source)
        prs = Presentation(path)
        
        slides_text = []
        for i, slide in enumerate(prs.slides):
            slide_content = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_content.append(shape.text)
            
            if slide_content:
                slides_text.append(f"--- Slide {i + 1} ---\n" + "\n".join(slide_content))
        
        text_content = "\n\n".join(slides_text)
        
        return ProcessedInput(
            input_type=InputType.PPTX,
            text_content=text_content,
            source=str(path),
            metadata={
                "file_name": path.name,
                "slide_count": len(prs.slides)
            }
        )


class ImageHandler(BaseHandler):
    """
    Handler for image files.
    
    Converts images to base64 for use with vision-capable models.
    """
    
    SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}
    
    def can_handle(self, input_source: Union[str, Path]) -> bool:
        path = Path(input_source)
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    def process(self, input_source: Union[str, Path, bytes]) -> ProcessedInput:
        path = Path(input_source)
        
        # Read and encode image
        with open(path, 'rb') as f:
            image_data = f.read()
        
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Determine MIME type
        extension = path.suffix.lower()
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp'
        }
        mime_type = mime_types.get(extension, 'image/png')
        
        # Get image dimensions if PIL is available
        metadata = {"file_name": path.name}
        if Image:
            try:
                with Image.open(path) as img:
                    metadata["width"] = img.width
                    metadata["height"] = img.height
                    metadata["format"] = img.format
            except Exception:
                pass
        
        image_info = {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_image}"
            },
            "source": str(path)
        }
        
        return ProcessedInput(
            input_type=InputType.IMAGE,
            text_content=f"[Image: {path.name}]",
            images=[image_info],
            source=str(path),
            metadata=metadata
        )


class InputHandler:
    """
    Main input handler that delegates to specific handlers based on input type.
    
    Usage:
        handler = InputHandler()
        
        # Process single text input
        result = handler.process_text("Some requirements text")
        
        # Process single file
        result = handler.process_file("requirements.docx")
        
        # Process multiple inputs
        result = handler.process_multiple([
            "requirements.docx",
            "screenshots/",
            "Additional notes here"
        ])
    """
    
    def __init__(self):
        self.handlers: list[BaseHandler] = [
            DocxHandler(),
            PdfHandler(),
            ExcelHandler(),
            PowerPointHandler(),
            ImageHandler(),
            TextHandler(),  # TextHandler should be last as it's the fallback
        ]
    
    def _get_handler(self, input_source: Union[str, Path]) -> BaseHandler:
        """Get the appropriate handler for the input source."""
        for handler in self.handlers:
            if handler.can_handle(input_source):
                return handler
        return self.handlers[-1]  # Default to TextHandler
    
    def process_text(self, text: str) -> ProcessedInput:
        """
        Process direct text input.
        
        Args:
            text: The text content to process
            
        Returns:
            ProcessedInput containing the text
        """
        return ProcessedInput(
            input_type=InputType.TEXT,
            text_content=text,
            source="direct_text"
        )
    
    def process_file(self, file_path: Union[str, Path]) -> ProcessedInput:
        """
        Process a single file.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            ProcessedInput containing the extracted content
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        handler = self._get_handler(path)
        return handler.process(path)
    
    def process_directory(
        self,
        dir_path: Union[str, Path],
        recursive: bool = True
    ) -> MultiInput:
        """
        Process all supported files in a directory.
        
        Args:
            dir_path: Path to the directory
            recursive: Whether to process subdirectories
            
        Returns:
            MultiInput containing all processed files
        """
        path = Path(dir_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")
        
        multi_input = MultiInput()
        
        # Get all supported extensions
        supported_extensions = set()
        for handler in self.handlers:
            if hasattr(handler, 'SUPPORTED_EXTENSIONS'):
                supported_extensions.update(handler.SUPPORTED_EXTENSIONS)
        supported_extensions.update({'.docx', '.pdf', '.xlsx', '.pptx'})
        
        # Find and process files
        pattern = '**/*' if recursive else '*'
        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    processed = self.process_file(file_path)
                    multi_input.add_input(processed)
                except Exception as e:
                    # Log error but continue processing other files
                    print(f"Warning: Failed to process {file_path}: {e}")
        
        return multi_input
    
    def process_multiple(
        self,
        inputs: list[Union[str, Path]]
    ) -> MultiInput:
        """
        Process multiple inputs (files, directories, or text).
        
        Args:
            inputs: List of file paths, directory paths, or text strings
            
        Returns:
            MultiInput containing all processed inputs
        """
        multi_input = MultiInput()
        
        for input_item in inputs:
            path = Path(input_item) if not isinstance(input_item, Path) else input_item
            
            if path.exists():
                if path.is_file():
                    processed = self.process_file(path)
                    multi_input.add_input(processed)
                elif path.is_dir():
                    dir_result = self.process_directory(path)
                    for inp in dir_result.inputs:
                        multi_input.add_input(inp)
            else:
                # Treat as direct text input
                processed = self.process_text(str(input_item))
                multi_input.add_input(processed)
        
        return multi_input
    
    def process(
        self,
        input_source: Union[str, Path, list[Union[str, Path]]]
    ) -> Union[ProcessedInput, MultiInput]:
        """
        Universal processing method that handles any input type.
        
        Args:
            input_source: Single input (text/file/directory) or list of inputs
            
        Returns:
            ProcessedInput for single input, MultiInput for multiple inputs
        """
        if isinstance(input_source, list):
            return self.process_multiple(input_source)
        
        path = Path(input_source) if isinstance(input_source, str) else input_source
        
        if isinstance(input_source, str) and not path.exists():
            # Direct text input
            return self.process_text(input_source)
        
        if path.is_file():
            return self.process_file(path)
        elif path.is_dir():
            return self.process_directory(path)
        else:
            return self.process_text(str(input_source))
