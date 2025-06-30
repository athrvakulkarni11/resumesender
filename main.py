from typing import Annotated
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
import markdownify
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, TextContent
from pydantic import BaseModel, AnyUrl, Field
import readabilipy
from pathlib import Path
import os
import docx2txt
import PyPDF2
import io

# TODO: Replace these with your actual values
TOKEN = "cc45166fd38a"  # Replace with your application key
MY_NUMBER = "918010353482"  # Replace with your phone number (country code + number, no +)

# TODO: Set the path to your resume file
RESUME_PATH = "resume_athrva.pdf"  # Change this to your resume file path


class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None


class SimpleBearerAuthProvider(BearerAuthProvider):
    """
    A simple BearerAuthProvider that does not require any specific configuration.
    It allows any valid bearer token to access the MCP server.
    For a more complete implementation that can authenticate dynamically generated tokens,
    please use `BearerAuthProvider` with your public key or JWKS URI.
    """

    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(
            public_key=k.public_key, jwks_uri=None, issuer=None, audience=None
        )
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="unknown",
                scopes=[],
                expires_at=None,  # No expiration for simplicity
            )
        return None


def convert_resume_to_markdown(file_path: str) -> str:
    """
    Convert resume file to markdown format.
    Supports PDF, DOCX, and TXT files.
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Resume file not found: {file_path}")
    
    file_extension = file_path.suffix.lower()
    
    try:
        if file_extension == '.pdf':
            return convert_pdf_to_markdown(file_path)
        elif file_extension == '.docx':
            return convert_docx_to_markdown(file_path)
        elif file_extension == '.txt':
            return convert_txt_to_markdown(file_path)
        elif file_extension == '.md':
            # Already markdown
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    except Exception as e:
        raise McpError(
            ErrorData(
                code=INTERNAL_ERROR,
                message=f"Failed to convert resume to markdown: {str(e)}"
            )
        )


def convert_pdf_to_markdown(file_path: Path) -> str:
    """Convert PDF file to markdown."""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        
        # Basic markdown formatting
        lines = text.split('\n')
        markdown_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Simple heuristics for formatting
            if line.isupper() and len(line) > 3:
                # Likely a section header
                markdown_lines.append(f"## {line.title()}")
            elif any(keyword in line.lower() for keyword in ['email:', 'phone:', 'linkedin:', 'github:']):
                # Contact information
                markdown_lines.append(f"**{line}**")
            else:
                markdown_lines.append(line)
        
        return '\n\n'.join(markdown_lines)
    
    except Exception as e:
        raise McpError(
            ErrorData(
                code=INTERNAL_ERROR,
                message=f"Failed to read PDF file: {str(e)}"
            )
        )


def convert_docx_to_markdown(file_path: Path) -> str:
    """Convert DOCX file to markdown."""
    try:
        text = docx2txt.process(str(file_path))
        
        # Basic markdown formatting
        lines = text.split('\n')
        markdown_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Simple heuristics for formatting
            if line.isupper() and len(line) > 3:
                # Likely a section header
                markdown_lines.append(f"## {line.title()}")
            elif any(keyword in line.lower() for keyword in ['email:', 'phone:', 'linkedin:', 'github:']):
                # Contact information
                markdown_lines.append(f"**{line}**")
            else:
                markdown_lines.append(line)
        
        return '\n\n'.join(markdown_lines)
    
    except Exception as e:
        raise McpError(
            ErrorData(
                code=INTERNAL_ERROR,
                message=f"Failed to read DOCX file: {str(e)}"
            )
        )


def convert_txt_to_markdown(file_path: Path) -> str:
    """Convert TXT file to markdown."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Basic markdown formatting
        lines = text.split('\n')
        markdown_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Simple heuristics for formatting
            if line.isupper() and len(line) > 3:
                # Likely a section header
                markdown_lines.append(f"## {line.title()}")
            elif any(keyword in line.lower() for keyword in ['email:', 'phone:', 'linkedin:', 'github:']):
                # Contact information
                markdown_lines.append(f"**{line}**")
            else:
                markdown_lines.append(line)
        
        return '\n\n'.join(markdown_lines)
    
    except Exception as e:
        raise McpError(
            ErrorData(
                code=INTERNAL_ERROR,
                message=f"Failed to read TXT file: {str(e)}"
            )
        )


class Fetch:
    IGNORE_ROBOTS_TXT = True
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(
        cls,
        url: str,
        user_agent: str,
        force_raw: bool = False,
    ) -> tuple[str, str]:
        """
        Fetch the URL and return the content in a form ready for the LLM, as well as a prefix string with status information.
        """
        from httpx import AsyncClient, HTTPError

        async with AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    follow_redirects=True,
                    headers={"User-Agent": user_agent},
                    timeout=30,
                )
            except HTTPError as e:
                raise McpError(
                    ErrorData(
                        code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"
                    )
                )
            if response.status_code >= 400:
                raise McpError(
                    ErrorData(
                        code=INTERNAL_ERROR,
                        message=f"Failed to fetch {url} - status code {response.status_code}",
                    )
                )

            page_raw = response.text

        content_type = response.headers.get("content-type", "")
        is_page_html = (
            "<html" in page_raw[:100] or "text/html" in content_type or not content_type
        )

        if is_page_html and not force_raw:
            return cls.extract_content_from_html(page_raw), ""

        return (
            page_raw,
            f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
        )

    @staticmethod
    def extract_content_from_html(html: str) -> str:
        """Extract and convert HTML content to Markdown format.

        Args:
            html: Raw HTML content to process

        Returns:
            Simplified markdown version of the content
        """
        ret = readabilipy.simple_json.simple_json_from_html_string(
            html, use_readability=True
        )
        if not ret["content"]:
            return "<error>Page failed to be simplified from HTML</error>"
        content = markdownify.markdownify(
            ret["content"],
            heading_style=markdownify.ATX,
        )
        return content


mcp = FastMCP(
    "Resume MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

ResumeToolDescription = RichToolDescription(
    description="Serve your resume in plain markdown.",
    use_when="Puch (or anyone) asks for your resume; this must return raw markdown, no extra formatting.",
    side_effects=None,
)

@mcp.tool(description=ResumeToolDescription.model_dump_json())
async def resume() -> str:
    """
    Return your resume exactly as markdown text.
    
    This function:
    1. Finds and reads your resume file.
    2. Converts the resume to markdown format.
    3. Handles errors gracefully.
    4. Returns the resume as markdown text.
    """
    try:
        if not RESUME_PATH or RESUME_PATH == "path/to/your/resume.pdf":
            return "# Resume Not Configured\n\nPlease set the RESUME_PATH variable to point to your resume file."
        
        markdown_resume = convert_resume_to_markdown(RESUME_PATH)
        return markdown_resume
    
    except FileNotFoundError:
        return f"# Resume File Not Found\n\nThe resume file at `{RESUME_PATH}` was not found. Please check the file path."
    
    except Exception as e:
        return f"# Resume Error\n\nFailed to load resume: {str(e)}"


@mcp.tool
async def validate() -> str:
    """
    NOTE: This tool must be present in an MCP server used by puch.
    Returns the phone number for validation.
    """
    return MY_NUMBER


FetchToolDescription = RichToolDescription(
    description="Fetch a URL and return its content.",
    use_when="Use this tool when the user provides a URL and asks for its content, or when the user wants to fetch a webpage.",
    side_effects="The user will receive the content of the requested URL in a simplified format, or raw HTML if requested.",
)


@mcp.tool(description=FetchToolDescription.model_dump_json())
async def fetch(
    url: Annotated[AnyUrl, Field(description="URL to fetch")],
    max_length: Annotated[
        int,
        Field(
            default=5000,
            description="Maximum number of characters to return.",
            gt=0,
            lt=1000000,
        ),
    ] = 5000,
    start_index: Annotated[
        int,
        Field(
            default=0,
            description="On return output starting at this character index, useful if a previous fetch was truncated and more context is required.",
            ge=0,
        ),
    ] = 0,
    raw: Annotated[
        bool,
        Field(
            default=False,
            description="Get the actual HTML content if the requested page, without simplification.",
        ),
    ] = False,
) -> list[TextContent]:
    """Fetch a URL and return its content."""
    url_str = str(url).strip()
    if not url:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="URL is required"))

    content, prefix = await Fetch.fetch_url(url_str, Fetch.USER_AGENT, force_raw=raw)
    original_length = len(content)
    if start_index >= original_length:
        content = "<error>No more content available.</error>"
    else:
        truncated_content = content[start_index : start_index + max_length]
        if not truncated_content:
            content = "<error>No more content available.</error>"
        else:
            content = truncated_content
            actual_content_length = len(truncated_content)
            remaining_content = original_length - (start_index + actual_content_length)
            # Only add the prompt to continue fetching if there is still remaining content
            if actual_content_length == max_length and remaining_content > 0:
                next_start = start_index + actual_content_length
                content += f"\n\n<error>Content truncated. Call the fetch tool with a start_index of {next_start} to get more content.</error>"
    return [TextContent(type="text", text=f"{prefix}Contents of {url}:\n{content}")]


async def main():
    print(f"Starting MCP Resume Server...")
    print(f"Resume path: {RESUME_PATH}")
    print(f"Phone number: {MY_NUMBER}")
    print(f"Server will run on http://0.0.0.0:8085")
    
    await mcp.run_async(
        "streamable-http",
        host="0.0.0.0",
        port=8085,
    )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())