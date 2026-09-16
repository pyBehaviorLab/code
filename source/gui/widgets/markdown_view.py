"""Markdown → styled HTML viewer for in-app documentation.

Subset-of-CommonMark parser sufficient for the project's docs:
fenced code blocks, headings (#…####) with navigation anchors,
unordered / ordered lists, blockquotes, horizontal rules, tables,
images, and inline runs (code, bold, italic, links). Output is
dark-theme HTML aligned with the rest of the app.

Every heading gets an ``<a name>`` anchor derived from its text
(lowercase, runs of non-alphanumerics collapsed to ``-``), so a doc
can open with a clickable table of contents of ``[…](#anchor)``
links, QTextBrowser scrolls to fragment links internally while
still opening http(s) links in the external browser. Relative image
paths resolve against the .md file's directory (search path set by
``load_md_file``).

Used by operant._build_documentation_sidebar (loads a .md file) and
maze._build_doc_content (loads a small static HTML fallback).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

logger = logging.getLogger(__name__)

_PAGE_TEMPLATE = """<html>
<head><style>
    body {{ color: #f8f8f2; background-color: #1e1e1e;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            padding: 15px; }}
    a {{ color: #8be9fd; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
</style></head>
<body>{body}</body>
</html>"""


# The documentation sidebar panel is 550 px wide; images render at the
# usable content width and QTextBrowser scales height to keep the aspect.
_IMG_WIDTH = 500


def _slug(title: str) -> str:
    """Heading text → anchor name (lowercase, non-alphanumerics → '-')."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def _inline(text: str) -> str:
    """Inline markdown: image, link, backtick code, bold, italic.

    Images, links and code spans are stashed as placeholder tokens
    BEFORE the emphasis passes run, otherwise underscores/asterisks
    inside a URL or code span (``pyBehaviorLab_GUI_Guide.html``,
    ``c.*``) are eaten as italic/bold markers and the href breaks.
    """
    stash: list[str] = []

    def _hold(html: str) -> str:
        stash.append(html)
        return f"\x00{len(stash) - 1}\x00"

    text = re.sub(
        r"!\[([^\]]*)\]\(([^)]+)\)",
        lambda m: _hold(f'<img src="{m.group(2)}" alt="{m.group(1)}" '
                        f'width="{_IMG_WIDTH}">'), text)
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: _hold(f'<a href="{m.group(2)}">{m.group(1)}</a>'), text)
    text = re.sub(
        r"`([^`]+)`",
        lambda m: _hold(
            '<code style="background:#333;padding:2px 5px;border-radius:3px;'
            'font-family:Consolas,monospace;color:#50fa7b;">'
            f"{m.group(1)}</code>"), text)
    text = re.sub(r"\*\*([^*]+)\*\*",
                  r'<strong style="color:#f1fa8c;">\1</strong>', text)
    text = re.sub(r"__([^_]+)__",
                  r'<strong style="color:#f1fa8c;">\1</strong>', text)
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
    text = re.sub(r"_([^_]+)_", r"<em>\1</em>", text)
    for i, html in enumerate(stash):
        text = text.replace(f"\x00{i}\x00", html)
    return text


def markdown_to_html(md_text: str) -> str:
    """Convert a subset of CommonMark to styled HTML."""
    out: list[str] = []
    in_code = in_table = in_list = False
    for line in md_text.split("\n"):
        if line.strip().startswith("```"):
            if in_code:
                out.append("</code></pre>")
                in_code = False
            else:
                out.append(
                    '<pre style="background:#111;padding:10px;'
                    'border-radius:6px;margin:8px 0;overflow-x:auto;'
                    'font-family:Consolas,monospace;font-size:12px;'
                    'line-height:1.4;"><code style="color:#50fa7b;">')
                in_code = True
            continue
        if in_code:
            out.append(line.replace("&", "&amp;")
                       .replace("<", "&lt;").replace(">", "&gt;"))
            continue

        if "|" in line and line.strip().startswith("|"):
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if all(set(c) <= set("-: ") for c in cells):
                continue                                    # header divider
            if not in_table:
                out.append('<table style="border-collapse:collapse;'
                           'margin:10px 0;width:100%;">')
                in_table = True
            row = "<tr>" + "".join(
                f'<td style="border:1px solid #444;padding:6px 10px;'
                f'text-align:left;">{c}</td>' for c in cells) + "</tr>"
            out.append(row)
            continue
        if in_table and not line.strip():
            out.append("</table>")
            in_table = False

        for prefix, tag, color, margin in (
            ("#### ", "h4", "#8be9fd", "12px 0 6px 0"),
            ("### ",  "h3", "#8be9fd", "14px 0 8px 0"),
            ("## ",   "h2", "#bd93f9", "16px 0 10px 0"),
            ("# ",    "h1", "#ff79c6", "18px 0 12px 0"),
        ):
            if line.startswith(prefix):
                title = line[len(prefix):]
                out.append(
                    f'<a name="{_slug(title)}"></a>'
                    f'<{tag} style="color:{color};margin:{margin};">'
                    f'{title}</{tag}>')
                break
        else:
            if line.strip() in ("---", "***", "___"):
                out.append('<hr style="border:none;border-top:1px solid #444;'
                           'margin:15px 0;">')
                continue
            stripped = line.lstrip()
            bullet = stripped.startswith("- ") or stripped.startswith("* ")
            numbered = bool(re.match(r"^\d+\. ", stripped))
            if bullet or numbered:
                if not in_list:
                    out.append('<ul style="margin:8px 0;padding-left:20px;">'
                               if bullet
                               else '<ol style="margin:8px 0;padding-left:20px;">')
                    in_list = True
                body = (stripped[2:] if bullet
                        else re.sub(r"^\d+\. ", "", stripped))
                out.append(f'<li style="margin:4px 0;">{_inline(body)}</li>')
                continue
            if in_list and (not stripped or stripped[0] not in "-*0123456789"):
                out.append("</ul>" if out and out[-1].startswith("<li") else "</ol>")
                in_list = False
            if stripped.startswith("> "):
                out.append(
                    '<blockquote style="border-left:3px solid #bd93f9;'
                    'margin:10px 0;padding:5px 15px;color:#aaa;">'
                    f'{_inline(stripped[2:])}</blockquote>')
                continue
            if stripped:
                out.append(
                    f'<p style="margin:8px 0;line-height:1.5;">'
                    f'{_inline(stripped)}</p>')
            else:
                out.append("<br>")

    if in_code:
        out.append("</code></pre>")
    if in_table:
        out.append("</table>")
    if in_list:
        out.append("</ul>")
    return _PAGE_TEMPLATE.format(body="\n".join(out))


class MarkdownView(QtWidgets.QTextBrowser):
    """Read-only HTML view for in-app documentation."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setOpenExternalLinks(True)
        from source.gui.theme import THEME as _T
        self.setStyleSheet(
            "QTextBrowser {"
            f" background-color: {_T.palette.surface};"
            f" border: 1px solid {_T.palette.surface_border_strong};"
            f" border-radius: {_T.radius.md}px; padding: 15px;"
            f" font-family: '{_T.font.family}', sans-serif; font-size: 10pt;"
            f" color: {_T.palette.text};"
            "}")

    def doSetSource(self, url, type=None) -> None:  # Qt override
        """Route link clicks: fragments scroll in-view, real documents
        open in the system browser.

        The in-app panel is a lightweight quick reference by
        construction, the full guide is a standalone HTML that must
        never be loaded INTO this QTextBrowser (megabytes of markup in
        the sidebar). So http(s), ``file://`` and repo-relative links
        (e.g. ``docs/pyBehaviorLab_GUI_Guide.html``, resolved against
        the .md's folder) all go to ``QDesktopServices.openUrl``.
        """
        if not url.path():  # same-document #anchor, scroll internally
            if type is None:
                super().doSetSource(url)
            else:
                super().doSetSource(url, type)
            return
        if url.scheme() in ("http", "https"):
            QtGui.QDesktopServices.openUrl(url)
            return
        local = url.toLocalFile() if url.isLocalFile() else ""
        if not local:
            base = (Path(self.searchPaths()[0]) if self.searchPaths()
                    else Path.cwd())
            local = str((base / url.path()).resolve())
        if Path(local).exists():
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(local))
        else:
            logger.warning("Documentation link target not found: %s", local)

    def set_markdown(self, md_text: str) -> None:
        """Render a markdown string as HTML."""
        self.setHtml(markdown_to_html(md_text))

    def load_md_file(self, path: str | Path, fallback_html: str = "") -> bool:
        """Load + render a .md file. Returns True on success, False if
        the file is absent / unreadable (the caller's ``fallback_html``
        is shown in that case)."""
        try:
            with open(path, encoding="utf-8") as fh:
                md_text = fh.read()
            # Relative image paths in the doc resolve against its directory.
            self.setSearchPaths([str(Path(path).resolve().parent)])
            self.set_markdown(md_text)
            return True
        except (OSError, UnicodeDecodeError):
            if fallback_html:
                self.setHtml(fallback_html)
            return False
