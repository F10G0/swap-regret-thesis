"""Combine existing plot files without re-running experiments or plotting."""

from io import BytesIO
from pathlib import Path

from PIL import Image
from pypdf import PdfWriter


def merged_figure_pdf(paths: list[Path]) -> BytesIO:
    """Keep source PDF pages intact and preserve the caller's figure order."""
    output = BytesIO()
    with PdfWriter() as writer:
        for path in paths:
            if path.suffix.lower() == ".pdf":
                writer.append(str(path), import_outline=False)
            else:
                # Legacy results may only have a PNG. Composite transparency
                # onto white, matching the figure's appearance in the dashboard.
                with Image.open(path) as image, BytesIO() as page:
                    rgba = image.convert("RGBA")
                    background = Image.new("RGB", rgba.size, "white")
                    background.paste(rgba, mask=rgba.getchannel("A"))
                    background.save(page, format="PDF", resolution=150.0)
                    page.seek(0)
                    writer.append(page, import_outline=False)
        writer.write(output)
    output.seek(0)
    return output
