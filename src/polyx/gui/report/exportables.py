"""Report tables and figures, ready to take into a manuscript.

The report is a self-contained HTML: figures are 120 dpi PNGs inside base64 and
tables are HTML text. That is convenient to read and useless to submit, because
a figure has to be vector (or 300 dpi) and a table has to be data, not a
screenshot. This module takes what the report already produced and leaves it in
three forms, without touching each point where something is drawn:

* **Numbering.** Every table and every figure gets its number ("Table 3",
  "Figure 5") and its caption, in the report's language.
* **Links inside the HTML.** Under each table, CSV and LaTeX; under each
  figure, SVG. They are what makes the HTML portable on its own.
* **An export folder** next to the report, holding every format.

How it hooks in::

    figures._fig_to_b64(fig)   -> capture(fig, b64)     (every matplotlib figure)
    render._generate_report_impl -> start_recording() ... stop_recording() around assembly
    final html                 -> process(html, path)    (number, link, export)

Tables are not registered one by one: they are read back out of the final HTML,
which is where they already sit with their headings and section numbers
resolved. A new table in the report therefore becomes exportable without
touching this file.

No new dependencies: the XLSX is written by hand (a zip of XML) instead of
pulling in openpyxl, which the project does not depend on. The cost is that
only the four parts every reader (Excel, LibreOffice, Numbers, pandas) opens are
written.

Ported from the upstream project's ``polyx/core/exportables.py`` (MIT, Cristofher
Ferrada): the behaviour is upstream's, the identifiers and strings are English
so the module matches the rest of this tree.
"""

from __future__ import annotations

import base64
import bisect
import csv
import hashlib
import html as _html
import io
import re
import shutil
import unicodedata
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.patches import Rectangle

from polyx.core.i18n import _, language

#: (panel name, columns, rows)
Panel = tuple[str, list[str], list[list]]

TABLES_DIR = "tables"
FIGURES_DIR = "figures"
DATA_DIR = "data"

#: Requested resolution for the raster copy of a figure. Journals ask for 300 dpi
#: at final size; the HTML copy is 120 dpi and cannot be upsampled honestly.
JOURNAL_DPI = 300


@dataclass
class _Figure:
    """One captured figure: its title, its rendered formats and its data."""

    title: str
    svg: bytes = b""
    pdf: bytes = b""
    png: bytes = b""
    panels: list[Panel] = field(default_factory=list)


class _Recorder:
    """Figures captured while a report is assembled.

    A holder object rather than two module globals: the report package is
    imported once per process, and a nested report (a test rendering two
    documents) must not be able to clobber the outer one's registry through a
    stray ``global``.
    """

    def __init__(self) -> None:
        self.figures: dict[str, _Figure] | None = None
        self.heavy: bool = False

    def start(self, heavy: bool = True) -> None:
        self.figures = {}
        self.heavy = heavy

    def stop(self) -> None:
        self.figures = None
        self.heavy = False

    def active(self) -> bool:
        return self.figures is not None


_RECORDER = _Recorder()


def start_recording(heavy: bool = True) -> None:
    """Begin recording the figures that get drawn.

    ``heavy`` decides whether SVG, PDF and 300 dpi PNG are rendered as well,
    which is the expensive part. Without it only the title is kept, which is
    enough to number the figures.
    """
    _RECORDER.start(heavy)


def stop_recording() -> None:
    """Stop recording. Captures after this call are ignored."""
    _RECORDER.stop()


def recording() -> bool:
    """Whether a report is being assembled right now."""
    return _RECORDER.active()


def _key(b64: str) -> str:
    """Registry key for an embedded PNG: the hash of its bytes.

    sha256 and not a cheaper digest because this is an identity, not a security
    boundary: a collision would silently attach one figure's vector copy to
    another figure.
    """
    return hashlib.sha256(b64.encode("ascii")).hexdigest()


def attach_data(fig, columns: Sequence[str], rows: Sequence[Sequence], name: str = "") -> None:
    """Pin a figure's data when reading it back out of the drawing falls short.

    A drawn histogram only contains the bar heights; whoever wants to redraw it
    in Origin or in Python needs the raw values. A boxplot is worse: what is
    drawn is the whiskers, not the measurements.
    """
    panels = getattr(fig, "_polyx_panels", [])
    panels.append((name, list(columns), [list(f) for f in rows]))
    fig._polyx_panels = panels


def _figure_title(fig) -> str:
    """The figure's caption source: its suptitle, else the first axis title."""
    sup = getattr(fig, "_suptitle", None)
    if sup is not None and sup.get_text():
        return sup.get_text()
    for ax in fig.axes:
        if ax.get_title():
            return ax.get_title()
    return ""


def capture(fig, b64: str) -> None:
    """Record a matplotlib figure. Call it before the figure is closed."""
    registry = _RECORDER.figures
    if registry is None:
        return
    f = _Figure(title=_figure_title(fig))
    if _RECORDER.heavy:
        # svg.fonttype "none" leaves the text as text: it is editable in Inkscape
        # or Illustrator. pdf.fonttype 42 embeds TrueType instead of Type 3,
        # which journals reject often.
        with matplotlib.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
            for fmt, dpi in (("svg", None), ("pdf", None), ("png", JOURNAL_DPI)):
                buf = io.BytesIO()
                kw = {"dpi": dpi} if dpi else {}
                fig.savefig(buf, format=fmt, bbox_inches="tight", **kw)
                setattr(f, fmt, buf.getvalue())
        # The axis labels only exist after drawing, which is why this comes last.
        f.panels = getattr(fig, "_polyx_panels", None) or _panels_from_figure(fig)
    registry[_key(b64)] = f


# ────────────────────────────────────────────────────────────────────
# A figure's data, read back out of the drawing itself
# ────────────────────────────────────────────────────────────────────
def _nearby_label(ax, centre: float, half_width: float) -> str:
    """Text of the X tick that falls under a bar, when there is one."""
    for pos, lab in zip(ax.get_xticks(), ax.get_xticklabels(), strict=False):
        txt = lab.get_text()
        if txt and abs(pos - centre) <= max(half_width, 1e-9) * 1.01:
            return txt.replace("\n", " ")
    return ""


def _panels_from_figure(fig) -> list[Panel]:
    """Every panel of a figure, with whatever data can be recovered per panel."""
    panels: list[Panel] = []
    for i, ax in enumerate(fig.axes, start=1):
        name = ax.get_title() or f"panel{i}"

        bars = [
            p for p in ax.patches if isinstance(p, Rectangle) and p.get_width() and p.get_height()
        ]
        if bars:
            rows = []
            for p in bars:
                x0, w = p.get_x(), p.get_width()
                rows.append(
                    [
                        _nearby_label(ax, x0 + w / 2, abs(w) / 2),
                        x0,
                        x0 + w,
                        p.get_height(),
                    ]
                )
            panels.append((name, ["category", "x_start", "x_end", "value"], rows))

        series = []
        for k, ln in enumerate(ax.lines, start=1):
            xs, ys = ln.get_data()
            if len(xs) == 0:
                continue
            label = str(ln.get_label() or "")
            label = f"series{k}" if (not label or label.startswith("_")) else label
            series += [[label, float(x), float(y)] for x, y in zip(xs, ys, strict=False)]
        if series:
            panels.append((name, ["series", "x", "y"], series))

        points = []
        for k, col in enumerate(ax.collections, start=1):
            if not isinstance(col, PathCollection):
                continue
            offsets = np.asarray(col.get_offsets())
            if offsets.ndim != 2 or offsets.shape[1] != 2:
                continue
            label = str(col.get_label() or "")
            label = f"series{k}" if (not label or label.startswith("_")) else label
            points += [[label, float(x), float(y)] for x, y in offsets]
        if points:
            panels.append((name, ["series", "x", "y"], points))

        for im in ax.images:
            m = np.asarray(im.get_array())
            if m.ndim != 2:
                continue
            cols = [t.get_text() for t in ax.get_xticklabels()][: m.shape[1]]
            rows_labels = [t.get_text() for t in ax.get_yticklabels()][: m.shape[0]]
            cols += [str(j) for j in range(len(cols), m.shape[1])]
            rows_labels += [str(j) for j in range(len(rows_labels), m.shape[0])]
            rows = [[rows_labels[r]] + [float(v) for v in m[r]] for r in range(m.shape[0])]
            panels.append((name, [ax.get_ylabel() or "row", *cols], rows))
    return panels


# ────────────────────────────────────────────────────────────────────
# Tables, read back out of the HTML
# ────────────────────────────────────────────────────────────────────
class _TableReader(HTMLParser):
    """Rows and cells of a table: plain text, no nested markup."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._cell: list[str] | None = None
        self._row: list[str] | None = None
        self._span: int = 1

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
            span = dict(attrs).get("colspan")
            self._span = int(span) if span and span.isdigit() else 1
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            txt = re.sub(r"\s+", " ", "".join(self._cell)).strip()
            self._row.append(txt)
            # A merged cell occupies several columns: it is padded so every row
            # has the same width in the CSV.
            self._row.extend([""] * (self._span - 1))
            self._cell = None
            self._span = 1
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def _table_rows(table_html: str) -> list[list[str]]:
    """Rows of one table's HTML, padded to a rectangle."""
    reader = _TableReader()
    reader.feed(table_html)
    rows = reader.rows
    width = max((len(f) for f in rows), default=0)
    return [f + [""] * (width - len(f)) for f in rows]


_RE_NUMBER = re.compile(r"^-?(0|[1-9]\d*)(\.\d+)?$")


def _is_number(s: str) -> bool:
    """Whether a cell holds a plain decimal number, so it can be written as one."""
    return bool(_RE_NUMBER.match(s.strip()))


def _csv_bytes(columns: Sequence[str], rows: Sequence[Sequence]) -> bytes:
    """CSV with a BOM: without it, Excel on Windows opens UTF-8 accents broken."""
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow(columns)
    w.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")


_TEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _tex(s: str) -> str:
    """Escape a cell for LaTeX."""
    return "".join(_TEX_ESCAPES.get(c, c) for c in s)


def _latex_table(rows: list[list[str]], caption: str, label: str) -> bytes:
    """Three-line (booktabs) table, the shape journals ask for."""
    if not rows:
        return b""
    head, body = rows[0], rows[1:]
    # A column is numeric if its whole body is (or is a dash): it is aligned
    # right. The first one always goes left, it is the label.
    align = ["l"]
    for j in range(1, len(head)):
        vals = [f[j] for f in body if f[j] not in ("", "—", "-")]
        align.append("r" if vals and all(_is_number(v) for v in vals) else "l")
    lines = [
        "% Requires \\usepackage{booktabs}",
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{_tex(caption)}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{''.join(align)}}}",
        "\\toprule",
        " & ".join(_tex(c) for c in head) + " \\\\",
        "\\midrule",
    ]
    lines += [" & ".join(_tex(c) for c in f) + " \\\\" for f in body]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines).encode("utf-8")


# ── Minimal XLSX, without openpyxl ──────────────────────────────────
def _column_letter(n: int) -> str:
    """Spreadsheet column name for a zero-based index (0 -> A, 26 -> AA)."""
    s = ""
    n += 1
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _xml(s: str) -> str:
    """Escape text for XML content."""
    return _html.escape(s, quote=False)


def _xlsx_bytes(sheets: list[tuple[str, list[list[str]]]]) -> bytes:
    """One workbook with one sheet per table. Numbers go in as numbers, not text."""
    names: list[str] = []
    for name, _rows in sheets:
        clean = re.sub(r"[\[\]*?/\\:]", " ", name)[:31].strip() or "Sheet"
        base, k = clean, 2
        while clean in names:
            clean = f"{base[:28]} {k}"
            k += 1
        names.append(clean)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            + "".join(
                f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                for i in range(1, len(sheets) + 1)
            )
            + "</Types>",
        )
        z.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        z.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
            + "".join(
                f'<sheet name="{_xml(n)}" sheetId="{i}" r:id="rId{i}"/>'
                for i, n in enumerate(names, start=1)
            )
            + "</sheets></workbook>",
        )
        z.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(
                f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
                for i in range(1, len(sheets) + 1)
            )
            + "</Relationships>",
        )
        for i, (_, rows) in enumerate(sheets, start=1):
            body = []
            for r, row in enumerate(rows, start=1):
                cells = []
                for c, v in enumerate(row):
                    ref = f"{_column_letter(c)}{r}"
                    if v == "":
                        continue
                    if r > 1 and _is_number(v):
                        cells.append(f'<c r="{ref}"><v>{v}</v></c>')
                    else:
                        cells.append(
                            f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">{_xml(v)}</t></is></c>'
                        )
                body.append(f'<row r="{r}">{"".join(cells)}</row>')
            z.writestr(
                f"xl/worksheets/sheet{i}.xml",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                f"<sheetData>{''.join(body)}</sheetData></worksheet>",
            )
    return buf.getvalue()


# ────────────────────────────────────────────────────────────────────
# Names
# ────────────────────────────────────────────────────────────────────
def _slug(text: str, maximum: int = 40) -> str:
    """ASCII slug of a title: no accents, no punctuation, for file names."""
    s = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return s[:maximum].strip("_")


def _without_numbering(heading_html: str) -> str:
    """Text of a heading without tags and without the leading '3.2'."""
    t = re.sub(r"<[^>]+>", "", heading_html)
    t = _html.unescape(re.sub(r"\s+", " ", t)).strip()
    return re.sub(r"^\d+(\.\d+)*\.?\s*", "", t)


def _file_name(prefix: str, n: int, title: str) -> str:
    """File stem for an item: prefix, number, slug of the title and the locale."""
    base = f"{prefix}{n:02d}"
    s = _slug(title)
    tag = language().upper()
    return f"{base}_{s}_{tag}" if s else f"{base}_{tag}"


# ────────────────────────────────────────────────────────────────────
# Assembly: number, link and write the folder
# ────────────────────────────────────────────────────────────────────
_RE_TABLE = re.compile(r"<table class='data'[^>]*>.*?</table>", re.DOTALL)
_RE_PNG = re.compile(r"<img src='data:image/png;base64,([A-Za-z0-9+/=]+)'[^>]*>")
_RE_CAPTION = re.compile(r"\s*<div class='caption'>(.*?)</div>", re.DOTALL)
_RE_HEADING = re.compile(r"<h([234])\b[^>]*>(.*?)</h\1>", re.DOTALL)
_RE_SECTION = re.compile(r"<h2 id='([^']+)'[^>]*>")
#: Sections whose tables are one item's measurements, not a report table.
#: The particle cards each carry a measure table; numbering those would give the
#: reader forty "Table 12"s and bury the one table that belongs in the paper.
NON_TABLE_SECTIONS = frozenset({"cards"})


def _excluded_spans(html: str) -> list[tuple[int, int]]:
    """Spans of the sections whose ``table.data`` elements must not be numbered."""
    starts = [(m.start(), m.group(1)) for m in _RE_SECTION.finditer(html)]
    spans = []
    for i, (start, sid) in enumerate(starts):
        if sid not in NON_TABLE_SECTIONS:
            continue
        end = starts[i + 1][0] if i + 1 < len(starts) else len(html)
        spans.append((start, end))
    return spans


@dataclass
class _TableItem:
    """One numbered, exportable table."""

    n: int
    title: str
    rows: list[list[str]]
    name: str


@dataclass
class _FigureItem:
    """One numbered, exportable figure."""

    n: int
    caption: str
    fig: _Figure
    name: str


def _download_link(name: str, mime: str, data: bytes, text: str) -> str:
    """Inline download link: the file travels inside the HTML itself."""
    b64 = base64.b64encode(data).decode("ascii")
    return f"<a download='{name}' href='data:{mime};base64,{b64}'>{text}</a>"


def process(html: str, report_path: Path, export: bool = True) -> str:
    """Number the HTML's tables and figures, add links and write the folder.

    Returns the modified HTML. With ``export`` false it only numbers: the report
    comes out just as tidy but without links or folder.
    """
    table_label, figure_label = _("Table"), _("Figure")
    headings = [(m.start(), _without_numbering(m.group(2))) for m in _RE_HEADING.finditer(html)]
    heading_pos = [p for p, _ in headings]
    registry = _RECORDER.figures or {}
    excluded = _excluded_spans(html)

    # Every edit is (start, end, text). An insertion has start == end.
    edits: list[tuple[int, int, str]] = []
    tables: list[_TableItem] = []
    figures: list[_FigureItem] = []

    # ── Tables ──
    for m in _RE_TABLE.finditer(html):
        if any(start <= m.start() < end for start, end in excluded):
            continue
        rows = _table_rows(m.group(0))
        if len(rows) < 2:
            continue
        k = bisect.bisect_right(heading_pos, m.start()) - 1
        title = headings[k][1] if k >= 0 else ""
        n = len(tables) + 1
        item = _TableItem(n, title, rows, _file_name(table_label, n, title))
        tables.append(item)
        caption = f"{table_label} {n}." + (f" {_html.escape(title)}" if title else "")
        edits.append((m.start(), m.start(), f"<div class='tabcap'>{caption}</div>"))
        if export:
            legend = f"{table_label} {n}. {title}".strip()
            links = " · ".join(
                [
                    _download_link(
                        item.name + ".csv",
                        "text/csv;charset=utf-8",
                        _csv_bytes(rows[0], rows[1:]),
                        "CSV",
                    ),
                    _download_link(
                        item.name + ".tex",
                        "text/plain;charset=utf-8",
                        _latex_table(rows, title or legend, f"tab:{n}"),
                        "LaTeX",
                    ),
                ]
            )
            edits.append((m.end(), m.end(), f"<div class='exp'>{_('Export')}: {links}</div>"))

    # ── Figures (only matplotlib ones, which are the ones in the registry) ──
    for m in _RE_PNG.finditer(html):
        fig = registry.get(_key(m.group(1)))
        if fig is None:
            continue
        n = len(figures) + 1
        # A figure with no title of its own (the calibration one, for instance)
        # takes the one from the heading it falls under, just like the tables.
        if not fig.title:
            k = bisect.bisect_right(heading_pos, m.start()) - 1
            fig.title = headings[k][1] if k >= 0 else ""
        cap = _RE_CAPTION.match(html, m.end())
        if cap:
            text = cap.group(1)
            new = re.sub(
                r"^(\s*)(Figure|Figura)\.",
                rf"\g<1>{figure_label} {n}.",
                text,
                count=1,
            )
            if new == text:
                new = f"{figure_label} {n}. {text.strip()}"
            legend = re.sub(r"<[^>]+>", "", new).strip()
            edits.append((cap.start(1), cap.end(1), new))
            point = cap.end()
        else:
            legend = f"{figure_label} {n}. {fig.title}".strip()
            edits.append(
                (
                    m.end(),
                    m.end(),
                    f"<div class='caption'>{_html.escape(legend)}</div>",
                )
            )
            point = m.end()
        item = _FigureItem(n, legend, fig, _file_name(figure_label, n, fig.title))
        figures.append(item)
        if export and fig.svg:
            edits.append(
                (
                    point,
                    point,
                    f"<div class='exp'>{_('Export')}: "
                    + _download_link(item.name + ".svg", "image/svg+xml", fig.svg, "SVG")
                    + "</div>",
                )
            )

    # Applied back to front so the positions do not move. At equal start, the
    # edit noted first ends up first.
    result = html
    order = sorted(range(len(edits)), key=lambda i: (edits[i][0], i), reverse=True)
    for i in order:
        start, end, txt = edits[i]
        result = result[:start] + txt + result[end:]

    if export and (tables or figures):
        write_export_dir(Path(report_path), tables, figures)
    return result


def export_dir(report_path: Path) -> Path:
    """Folder that holds a report's exported tables and figures."""
    report_path = Path(report_path)
    return report_path.with_name(report_path.stem + "_export")


def write_export_dir(
    report_path: Path, tables: list[_TableItem], figures: list[_FigureItem]
) -> Path:
    """Write ``tables/``, ``figures/``, ``data/`` and ``index.csv`` next to the report."""
    root = export_dir(report_path)
    # Only this module's own output is deleted: anything the user left in the
    # folder is respected. Regenerating the report must not leave old figures
    # mixed in with the new ones.
    for sub in (TABLES_DIR, FIGURES_DIR, DATA_DIR):
        shutil.rmtree(root / sub, ignore_errors=True)
    (root / TABLES_DIR).mkdir(parents=True, exist_ok=True)
    (root / FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    (root / DATA_DIR).mkdir(parents=True, exist_ok=True)

    table_label, figure_label = _("Table"), _("Figure")
    index: list[list] = []

    for t in tables:
        d = root / TABLES_DIR
        legend = f"{table_label} {t.n}. {t.title}".strip()
        (d / f"{t.name}.csv").write_bytes(_csv_bytes(t.rows[0], t.rows[1:]))
        (d / f"{t.name}.tex").write_bytes(_latex_table(t.rows, t.title or legend, f"tab:{t.n}"))
        index.append(
            [
                table_label,
                t.n,
                f"{TABLES_DIR}/{t.name}.csv, .tex",
                legend,
            ]
        )
    if tables:
        tag = language().upper()
        (root / TABLES_DIR / f"{table_label}s_{tag}.xlsx").write_bytes(
            _xlsx_bytes([(f"{table_label} {t.n}", t.rows) for t in tables])
        )

    for f in figures:
        d = root / FIGURES_DIR
        if f.fig.svg:
            (d / f"{f.name}.svg").write_bytes(f.fig.svg)
            (d / f"{f.name}.pdf").write_bytes(f.fig.pdf)
            (d / f"{f.name}.png").write_bytes(f.fig.png)
        files = [f"{FIGURES_DIR}/{f.name}.svg, .pdf, .png"]
        panels = f.fig.panels
        for i, (_panel, cols, rows) in enumerate(panels, start=1):
            suffix = f"_p{i}" if len(panels) > 1 else ""
            target = root / DATA_DIR / f"{f.name}_data{suffix}.csv"
            target.write_bytes(_csv_bytes(cols, rows))
            files.append(f"{DATA_DIR}/{target.name}")
        index.append([figure_label, f.n, ", ".join(files), f.caption])

    (root / "index.csv").write_bytes(
        _csv_bytes(
            [_("Kind"), _("Number"), _("Files"), _("Caption")],
            index,
        )
    )
    return root
