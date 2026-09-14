from io import BytesIO

from PIL import Image
from pypdf import PdfReader, PdfWriter
import pytest

from tests.web.support import create_test_app, csrf_token


@pytest.fixture
def export_app(tmp_path):
    app, service = create_test_app(tmp_path)
    return app, service


def make_figure(directory, stem, widths=(200,)):
    directory.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (300, 200), (255, 255, 255, 0)).save(directory / f"{stem}.png")
    if widths:
        with PdfWriter() as writer:
            for width in widths:
                writer.add_blank_page(width=width, height=400)
            writer.write(directory / f"{stem}.pdf")
    return f"{stem}.pdf" if widths else f"{stem}.png"


def download(client, filenames, mode="figure_builder"):
    return client.post("/figures/download-filtered.pdf", data={
        "_csrf_token": csrf_token(client), "mode": mode, "filenames": filenames,
    })


def test_export_merges_only_selected_figures_in_submitted_order(export_app):
    app, service = export_app
    directory = service.figure_builder.output_dir
    first = make_figure(directory, "external", (201, 202))
    second = make_figure(directory, "internal", (301,))
    make_figure(directory, "swap", (401,))
    originals = {path: path.read_bytes() for path in directory.iterdir()}
    client = app.test_client()
    response = download(client, [second, first])

    assert response.status_code == 200
    assert response.mimetype == "application/pdf"
    assert "attachment;" in response.headers["Content-Disposition"]
    assert "filtered-regret-figures.pdf" in response.headers["Content-Disposition"]
    assert "no-store" in response.headers["Cache-Control"]
    pages = PdfReader(BytesIO(response.data)).pages
    assert [float(page.mediabox.width) for page in pages] == [301, 201, 202]
    assert {path: path.read_bytes() for path in directory.iterdir()} == originals


def test_export_supports_pdf_and_png_only_results(export_app):
    app, service = export_app
    current = make_figure(service.figure_builder.output_dir, "external", (350,))
    png_only = make_figure(service.figure_builder.output_dir, "swap", ())

    response = download(app.test_client(), [current, png_only])

    assert response.status_code == 200
    pages = PdfReader(BytesIO(response.data)).pages
    assert len(pages) == 2
    assert float(pages[0].mediabox.width) == 350
    assert float(pages[1].mediabox.width) == pytest.approx(300 * 72 / 150)


@pytest.mark.parametrize("filenames,mode,status", [
    ([], "figure_builder", 400),
    (["missing.pdf"], "figure_builder", 404),
    (["../outside.pdf"], "figure_builder", 404),
    (["/tmp/outside.pdf"], "figure_builder", 404),
    (["https://example.com/file.pdf"], "figure_builder", 404),
    (["file.csv"], "figure_builder", 404),
    (["file.pdf"], "unknown", 400),
])
def test_export_rejects_empty_or_invalid_selection(export_app, filenames, mode, status):
    app, service = export_app
    response = download(app.test_client(), filenames, mode)
    assert response.status_code == status
    assert "error" in response.json


def test_export_does_not_silently_skip_missing_figure(export_app):
    app, service = export_app
    filename = make_figure(service.figure_builder.output_dir, "external")
    response = download(app.test_client(), [filename, "missing.pdf"])
    assert response.status_code == 404


def test_export_rejects_symlink_outside_figure_directory(export_app, tmp_path):
    app, service = export_app
    filename = make_figure(service.figure_builder.output_dir, "external", ())
    pdf_path = (service.figure_builder.output_dir / filename).with_suffix(".pdf")
    outside = tmp_path / make_figure(tmp_path, "outside")
    pdf_path.symlink_to(outside)
    response = download(app.test_client(), [pdf_path.name])
    assert response.status_code == 404


def test_export_reports_unreadable_pdf(export_app):
    app, service = export_app
    filename = make_figure(service.figure_builder.output_dir, "external")
    (service.figure_builder.output_dir / filename).write_bytes(b"not a PDF")
    response = download(app.test_client(), [filename])
    assert response.status_code == 422
    assert "Generate" in response.json["error"]


def test_export_requires_csrf_token(export_app):
    app, _ = export_app
    response = app.test_client().post("/figures/download-filtered.pdf", data={"mode": "figure_builder"})
    assert response.status_code == 400
    assert b"CSRF" in response.data
