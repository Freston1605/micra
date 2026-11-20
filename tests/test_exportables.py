"""Report tables and figures: numbered, linked and exportable.

What is watched here is what a reader of the paper would notice:

* that the number the report shows ("Table 2") is the number of the exported
  file (``Table02_...``);
* that the CSV holds what the table shows, not something else;
* that the figure comes out vector and at 300 dpi, not as the report's 120 dpi
  thumbnail;
* that regenerating the report does not leave old files mixed in with new ones;
* that all of this exists in Spanish and in English.

The runner is the report itself, generated from one measured detection, so a
regression that only shows up inside a real document is caught here.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile

import pytest

from tests.test_bilingual_report import _estado_minimo


def _generate(tmp_path, lang="es", export=True, **kw):
    """Render the report in ``lang``, with the article export on by default."""
    import polyx.core.i18n as i18n
    import polyx.gui.report as report

    assert i18n.set_language(lang, persist=False)
    output = tmp_path / f"report_{lang}.html"
    report.generate_report(
        _estado_minimo(tmp_path),
        output,
        sections=list(report.IDS_SECTIONS),
        article_export=True,
        export_folder=export,
        **kw,
    )
    from polyx.gui.report import exportables

    return output, output.read_text(encoding="utf-8"), exportables.export_dir(output)


@pytest.fixture(autouse=True)
def _restore_language():
    """Leave the suite in the language conftest pins."""
    from polyx.core import i18n

    previous = i18n.language()
    yield
    i18n.set_language(previous, persist=False)


def _read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f))


@pytest.mark.parametrize(
    "lang,table,figure", [("es", "Tabla", "Figura"), ("en", "Table", "Figure")]
)
def test_the_folder_has_tables_figures_and_an_index(tmp_path, lang, table, figure):
    _, _, folder = _generate(tmp_path, lang)
    assert folder.name == f"report_{lang}_export"
    assert list((folder / "tables").glob("*.csv")), "no table was exported"
    assert list((folder / "tables").glob("*.tex"))
    assert list((folder / "tables").glob("*.xlsx"))
    assert list((folder / "figures").glob("*.svg"))
    assert (folder / "index.csv").exists()
    # The language prefix and suffix are in the file name, so two reports of the
    # same run in two languages do not overwrite each other.
    assert all(
        p.name.startswith(table) and f"_{lang.upper()}" in p.name
        for p in (folder / "tables").glob("*.csv")
    )
    assert all(p.name.startswith(figure) for p in (folder / "figures").glob("*.svg"))


@pytest.mark.parametrize(
    "lang,table,figure", [("es", "Tabla", "Figura"), ("en", "Table", "Figure")]
)
def test_the_report_numbers_the_same_as_the_files(tmp_path, lang, table, figure):
    _, html, folder = _generate(tmp_path, lang)
    n_tables = len(list((folder / "tables").glob("*.csv")))
    n_figures = len(list((folder / "figures").glob("*.svg")))
    assert n_tables and n_figures
    for n in range(1, n_tables + 1):
        assert f"{table} {n}." in html
    for n in range(1, n_figures + 1):
        assert f"{figure} {n}." in html
    assert f"{table} {n_tables + 1}." not in html
    # The old caption without a number ("Figure. Histogram…") must be gone.
    assert f">{figure}. " not in html


def test_the_csv_is_the_table_in_the_report(tmp_path):
    _, html, folder = _generate(tmp_path, "es")
    first = min((folder / "tables").glob("Tabla01_*.csv"))
    rows = _read_csv(first)
    table = re.search(r"<table class='data'[^>]*>(.*?)</table>", html, re.DOTALL).group(1)
    header = re.findall(r"<th[^>]*>(.*?)</th>", table, re.DOTALL)
    assert rows[0] == [re.sub(r"<[^>]+>", "", c).strip() for c in header]
    assert len(rows) > 1


def test_the_figure_is_vector_and_300_dpi(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    _, _, folder = _generate(tmp_path, "es")
    svg = min((folder / "figures").glob("*.svg"))
    assert b"<svg" in svg.read_bytes()
    assert svg.with_suffix(".pdf").read_bytes().startswith(b"%PDF")
    png = Image.open(svg.with_suffix(".png"))
    assert round(png.info["dpi"][0]) == 300


def test_every_exported_figure_carries_its_data(tmp_path):
    """Whoever redraws a figure needs the measurements, not the drawing."""
    _, _, folder = _generate(tmp_path, "es")
    figures = sorted((folder / "figures").glob("*.svg"))
    assert figures
    for svg in figures:
        stem = svg.stem
        data_files = list((folder / "data").glob(f"{stem}_data*.csv"))
        assert data_files, f"{stem} was exported without its data"
        for path in data_files:
            rows = _read_csv(path)
            assert rows and len(rows[0]) >= 2, f"{path.name} is not a data table"


def test_the_index_lists_every_file_with_its_caption(tmp_path):
    _, _, folder = _generate(tmp_path, "es")
    rows = _read_csv(folder / "index.csv")
    assert rows[0] == ["Tipo", "Número", "Archivos", "Leyenda"]
    for kind, number, files, caption in rows[1:]:
        assert caption.startswith(f"{kind} {number}.")
        for a in files.split(", "):
            if "/" in a:
                assert (folder / a.split(", ")[0]).parent.exists()


def test_the_download_links_are_in_the_html(tmp_path):
    _, html, _ = _generate(tmp_path, "es")
    assert "class='exp'" in html
    assert "download='Tabla01_" in html and ".csv'" in html
    assert "image/svg+xml" in html


def test_without_export_it_only_numbers(tmp_path):
    _, html, folder = _generate(tmp_path, "es", export=False)
    assert not folder.exists()
    assert "class='exp'" not in html
    assert "Tabla 1." in html and "Figura 1." in html


def test_regenerating_leaves_no_stale_files(tmp_path):
    _, _, folder = _generate(tmp_path, "es")
    stale = folder / "tables" / "Tabla99_from_a_previous_run.csv"
    stale.write_text("x", encoding="utf-8")
    foreign = folder / "my_notes.txt"
    foreign.write_text("mine", encoding="utf-8")
    _generate(tmp_path, "es")
    assert not stale.exists()
    assert foreign.exists(), "a file that was not the exporter's was deleted"


def test_the_particle_cards_are_not_numbered_as_tables(tmp_path):
    """A card's measure table is one particle's, not a report table."""
    _, html, folder = _generate(tmp_path, "es")
    cards = re.search(r"<h2 id='cards'>(.*?)(<h2 |</body>)", html, re.DOTALL)
    assert cards is not None, "the cards section is not in the report"
    assert "tabcap" not in cards.group(1)
    # And nothing from that section reached the table folder.
    assert not list((folder / "tables").glob("*Measure*"))


def test_the_xlsx_opens_and_keeps_numbers_as_numbers(tmp_path):
    _, _, folder = _generate(tmp_path, "es")
    xlsx = next((folder / "tables").glob("*.xlsx"))
    with zipfile.ZipFile(xlsx) as z:
        assert z.testzip() is None
        assert "xl/worksheets/sheet1.xml" in z.namelist()
    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.load_workbook(xlsx)
    assert book.sheetnames[0] == "Tabla 1"
    sheet = book.worksheets[0]
    assert sheet.max_row > 1


def test_latex_escapes_the_special_characters():
    from polyx.gui.report import exportables as ex

    tex = ex._latex_table([["a_b", "50 %"], ["x & y", "3.5"]], "Total $5", "tab:1").decode()
    assert r"a\_b" in tex and r"50 \%" in tex and r"x \& y" in tex
    assert r"\caption{Total \$5}" in tex
    assert "\\toprule" in tex and "\\bottomrule" in tex


def test_the_hand_written_xlsx_tells_a_number_from_text():
    from polyx.gui.report import exportables as ex

    data = ex._xlsx_bytes([("T", [["n", "x"], ["12.5", "abc"], ["007", "—"]])])
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        sheet = z.read("xl/worksheets/sheet1.xml").decode()
    assert "<v>12.5</v>" in sheet
    # A code with leading zeros is not a number: it is kept as written.
    assert '<t xml:space="preserve">007</t>' in sheet


def test_capture_is_a_no_op_outside_a_report():
    """Drawing figures outside a report must not pay for the heavy render."""
    from polyx.gui.report import exportables as ex

    assert ex.recording() is False
    ex.capture(object(), "Zm9v")  # no registry: nothing happens, no exception
    assert ex.recording() is False
