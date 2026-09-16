import os
from io import BytesIO
from pathlib import Path

from pypdf import PdfWriter

from experiments.plots.pdf_information import experiment_information_pdf


HEATMAP_COLORMAP = "Blues"
FIGURE_FORMATS = ("png", "pdf")


def figure_path(output_path: str | Path, figure_format: str) -> Path:
    if figure_format not in FIGURE_FORMATS:
        raise ValueError(f"unknown figure format: {figure_format}")
    return Path(output_path).with_suffix(f".{figure_format}")


def figure_paths(output_path: str | Path) -> tuple[Path, Path]:
    preview_path = Path(output_path)
    if preview_path.suffix.lower() != ".png":
        raise ValueError("figure output path must use the .png suffix")
    return preview_path, preview_path.with_suffix(".pdf")


def figure_pair_is_current(
    output_path: str | Path,
    input_mtime: int,
) -> bool:
    return all(
        path.is_file() and path.stat().st_mtime_ns >= input_mtime
        for path in figure_paths(output_path)
    )


def publish_figure_pair(
    source_path: str | Path,
    output_path: str | Path,
    overwrite: bool = True,
) -> None:
    sources = figure_paths(source_path)
    destinations = figure_paths(output_path)
    for source, destination in zip(sources, destinations):
        if overwrite or not destination.is_file():
            os.replace(source, destination)


def save_figure_pair(
    figure,
    output_path: str | Path,
    png_dpi: int | None = None,
    information_rows: list[tuple[str, str]] | None = None,
    **kwargs,
) -> tuple[Path, Path]:
    preview_path, pdf_path = figure_paths(output_path)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the physical page size; PDF artists remain vector, PNG is a preview.
    png_options = kwargs | {"dpi": png_dpi or 150}
    if information_rows is None:
        figure.savefig(pdf_path, **kwargs)
    else:
        figure_pdf = BytesIO()
        figure.savefig(figure_pdf, format="pdf", **kwargs)
        figure_pdf.seek(0)
        information_pdf = experiment_information_pdf(information_rows)
        with PdfWriter() as writer:
            writer.append(information_pdf, import_outline=False)
            writer.append(figure_pdf, import_outline=False)
            writer.write(pdf_path)
    figure.savefig(preview_path, **png_options)
    return preview_path, pdf_path
