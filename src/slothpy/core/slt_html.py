"""
HTML/CSS rendering for SlothPy objects in Jupyter and marimo.

Terminal output continues to use Rich trees via :mod:`slothpy.core.slt_common`.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from slothpy.core.slt_common import (
    SltDatasetNode,
    SltFileNode,
    SltGroupNode,
    SltVariableNode,
)

# ---------------------------------------------------------------------------
# Shared CSS
# ---------------------------------------------------------------------------


def _slt_theme_css(root: str) -> str:
    return f"""
{root} {{
  --slt-bg: #ffffff;
  --slt-surface: #ffffff;
  --slt-soft: #f8fafc;
  --slt-border: #e2e8f0;
  --slt-border-strong: #cbd5e1;
  --slt-text: #0f172a;
  --slt-muted: #64748b;
  --slt-red: #991b1b;
  --slt-red-soft: #fef2f2;
  --slt-green: #15803d;
  --slt-green-soft: #f0fdf4;
  --slt-yellow: #a16207;
  --slt-yellow-soft: #fefce8;
  --slt-orange: #c2410c;
  --slt-orange-soft: #fff7ed;
  --slt-gray: #475569;
  --slt-gray-soft: #f1f5f9;
  --slt-blue: #1d4ed8;
  --slt-blue-soft: #eff6ff;
  --slt-magenta: #a21caf;
  --slt-magenta-soft: #fdf4ff;
  --slt-cyan: #0e7490;
  --slt-cyan-soft: #ecfeff;
  width: 100%;
  max-width: 100%;
  box-sizing: border-box;
  font-family:
    ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif;
  color: var(--slt-text);
  background: var(--slt-bg);
}}

{root} * {{
  box-sizing: border-box;
}}

@media (prefers-color-scheme: dark) {{
  {root} {{
    --slt-bg: #0f172a;
    --slt-surface: #111827;
    --slt-soft: #1f2937;
    --slt-border: #334155;
    --slt-border-strong: #475569;
    --slt-text: #e5e7eb;
    --slt-muted: #94a3b8;
    --slt-red: #fca5a5;
    --slt-red-soft: #3f1d1d;
    --slt-green: #86efac;
    --slt-green-soft: #12351f;
    --slt-yellow: #fde68a;
    --slt-yellow-soft: #3f3412;
    --slt-orange: #fdba74;
    --slt-orange-soft: #3f2512;
    --slt-gray: #cbd5e1;
    --slt-gray-soft: #1e293b;
    --slt-blue: #93c5fd;
    --slt-blue-soft: #172554;
    --slt-magenta: #f0abfc;
    --slt-magenta-soft: #3b0764;
    --slt-cyan: #67e8f9;
    --slt-cyan-soft: #164e63;
  }}
}}
"""


def _slt_components_css() -> str:
    return """
.slt-card {
  border: 1px solid var(--slt-border);
  border-radius: 14px;
  background: var(--slt-surface);
  box-shadow:
    0 1px 2px rgba(15, 23, 42, 0.04),
    0 8px 24px rgba(15, 23, 42, 0.06);
  overflow: hidden;
}

.slt-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  padding: 16px 18px;
  border-bottom: 1px solid var(--slt-border);
  background:
    linear-gradient(135deg, #fff 0%, #fff 55%, #fef2f2 100%);
}

.slt-header.slt-header-blue {
  background:
    linear-gradient(135deg, #fff 0%, #fff 55%, #eff6ff 100%);
}

.slt-header.slt-header-magenta {
  background:
    linear-gradient(135deg, #fff 0%, #fff 55%, #fdf4ff 100%);
}

.slt-header.slt-header-yellow {
  background:
    linear-gradient(135deg, #fff 0%, #fff 55%, #fefce8 100%);
}

.slt-title {
  margin: 0;
  font-size: 16px;
  line-height: 1.2;
  font-weight: 750;
  letter-spacing: 0.01em;
}

.slt-dashboard .slt-title { color: var(--slt-red); }
.slt-title.file { color: var(--slt-red); }
.slt-title.group { color: var(--slt-blue); }
.slt-title.dataset { color: var(--slt-magenta); }
.slt-title.attrs { color: var(--slt-yellow); }

.slt-subtitle {
  margin-top: 4px;
  font-size: 12px;
  color: var(--slt-muted);
  word-break: break-word;
}

.slt-header-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: flex-end;
}

.slt-meta-pill {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 5px 10px;
  font-size: 12px;
  font-weight: 650;
  border: 1px solid var(--slt-border);
  background: var(--slt-soft);
  white-space: nowrap;
}

.slt-meta-pill.blue {
  color: var(--slt-blue);
  background: var(--slt-blue-soft);
  border-color: #bfdbfe;
}

.slt-meta-pill.magenta {
  color: var(--slt-magenta);
  background: var(--slt-magenta-soft);
  border-color: #f5d0fe;
}

.slt-meta-pill.cyan {
  color: var(--slt-cyan);
  background: var(--slt-cyan-soft);
  border-color: #a5f3fc;
}

.slt-meta-pill.green {
  color: var(--slt-green);
  background: var(--slt-green-soft);
  border-color: #bbf7d0;
}

.slt-meta-pill.yellow {
  color: var(--slt-yellow);
  background: var(--slt-yellow-soft);
  border-color: #fde68a;
}

.slt-meta-pill.gray {
  color: var(--slt-gray);
  background: var(--slt-gray-soft);
  border-color: #e2e8f0;
}

.slt-meta-pill.red {
  color: var(--slt-red);
  background: var(--slt-red-soft);
  border-color: #fecaca;
}

.slt-section {
  padding: 14px 18px 18px 18px;
}

.slt-section + .slt-section {
  border-top: 1px solid var(--slt-border);
}

.slt-section-title {
  margin: 0 0 10px 0;
  font-size: 12px;
  line-height: 1;
  font-weight: 750;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--slt-muted);
}

.slt-table-wrap {
  width: 100%;
  max-width: 100%;
  overflow-x: auto;
  border: 1px solid var(--slt-border);
  border-radius: 12px;
  background: var(--slt-surface);
}

.slt-table {
  width: 100%;
  border-collapse: collapse;
  table-layout: auto;
  font-size: 13px;
}

.slt-table th {
  background: var(--slt-soft);
  color: #334155;
  text-align: left;
  font-weight: 700;
  border-bottom: 1px solid var(--slt-border-strong);
  padding: 9px 10px;
  white-space: nowrap;
}

.slt-table td {
  border-bottom: 1px solid var(--slt-border);
  padding: 9px 10px;
  vertical-align: top;
}

.slt-table tr:last-child td {
  border-bottom: none;
}

.slt-table tr:hover td {
  background: #fafafa;
}

.slt-right {
  text-align: right;
  font-variant-numeric: tabular-nums;
}

.slt-mono {
  font-family:
    ui-monospace, SFMono-Regular, Menlo, Consolas,
    "Liberation Mono", monospace;
  font-variant-numeric: tabular-nums;
}

.slt-muted {
  color: var(--slt-muted);
}

.slt-empty {
  padding: 14px;
  color: var(--slt-muted);
  text-align: center;
  font-size: 13px;
}

.slt-cell-wrap {
  white-space: pre-wrap;
  word-break: break-word;
  overflow-wrap: anywhere;
}

.slt-error {
  color: var(--slt-red);
  font-weight: 650;
}

.slt-stack {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.slt-chip-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.slt-chip {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 12px;
  font-weight: 650;
  border: 1px solid #bfdbfe;
  color: var(--slt-blue);
  background: var(--slt-blue-soft);
}

.slt-primary-mark {
  display: inline-flex;
  margin-left: 8px;
  border-radius: 999px;
  padding: 2px 8px;
  font-size: 11px;
  font-weight: 750;
  color: var(--slt-yellow);
  background: var(--slt-yellow-soft);
  border: 1px solid #fde68a;
  vertical-align: middle;
}

.slt-attr-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.slt-attr-chip {
  display: inline-flex;
  border-radius: 8px;
  padding: 2px 8px;
  font-size: 12px;
  background: var(--slt-soft);
  border: 1px solid var(--slt-border);
}

.slt-attr-chip .slt-attr-key {
  color: var(--slt-yellow);
  font-weight: 650;
  margin-right: 4px;
}

@media (prefers-color-scheme: dark) {
  .slt-header {
    background:
      linear-gradient(135deg, #111827 0%, #111827 55%, #3f1d1d 100%);
  }

  .slt-header.slt-header-blue {
    background:
      linear-gradient(135deg, #111827 0%, #111827 55%, #172554 100%);
  }

  .slt-header.slt-header-magenta {
    background:
      linear-gradient(135deg, #111827 0%, #111827 55%, #3b0764 100%);
  }

  .slt-header.slt-header-yellow {
    background:
      linear-gradient(135deg, #111827 0%, #111827 55%, #3f3412 100%);
  }

  .slt-table th {
    color: #cbd5e1;
  }

  .slt-table tr:hover td {
    background: #172033;
  }
}
"""


def _dashboard_only_css() -> str:
    return """
.slt-status-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border-radius: 999px;
  padding: 5px 10px;
  font-size: 12px;
  font-weight: 650;
  border: 1px solid var(--slt-border);
  background: var(--slt-soft);
  white-space: nowrap;
}

.slt-dot {
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: var(--slt-green);
}

.slt-dot.closed {
  background: var(--slt-gray);
}

.slt-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding: 12px 18px 2px 18px;
}

.slt-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border-radius: 999px;
  padding: 5px 10px;
  font-size: 12px;
  font-weight: 650;
  border: 1px solid transparent;
}

.slt-badge .slt-badge-number {
  font-variant-numeric: tabular-nums;
}

.slt-badge.queued {
  color: var(--slt-yellow);
  background: var(--slt-yellow-soft);
  border-color: #fde68a;
}

.slt-badge.running {
  color: var(--slt-green);
  background: var(--slt-green-soft);
  border-color: #bbf7d0;
}

.slt-badge.cancelling {
  color: var(--slt-orange);
  background: var(--slt-orange-soft);
  border-color: #fed7aa;
}

.slt-badge.finished {
  color: var(--slt-green);
  background: var(--slt-green-soft);
  border-color: #bbf7d0;
}

.slt-badge.failed {
  color: var(--slt-red);
  background: var(--slt-red-soft);
  border-color: #fecaca;
}

.slt-badge.cancelled {
  color: var(--slt-gray);
  background: var(--slt-gray-soft);
  border-color: #e2e8f0;
}

.slt-center {
  text-align: center;
}

.slt-progress {
  position: relative;
  width: 100%;
  min-width: 160px;
  height: 18px;
  overflow: hidden;
  border-radius: 999px;
  background: #e5e7eb;
  border: 1px solid #d1d5db;
}

.slt-progress-fill {
  position: absolute;
  inset: 0 auto 0 0;
  height: 100%;
  border-radius: inherit;
  background:
    linear-gradient(90deg, #991b1b 0%, #dc2626 55%, #f97316 100%);
}

.slt-progress-muted .slt-progress-fill {
  background: #94a3b8;
}

.slt-progress-label {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0 8px;
  font-size: 11px;
  font-weight: 750;
  color: #0f172a;
  white-space: nowrap;
}

.slt-status {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  padding: 4px 9px;
  font-size: 12px;
  font-weight: 750;
  border: 1px solid transparent;
  white-space: nowrap;
}

.slt-status.queued {
  color: var(--slt-yellow);
  background: var(--slt-yellow-soft);
  border-color: #fde68a;
}

.slt-status.running {
  color: var(--slt-green);
  background: var(--slt-green-soft);
  border-color: #bbf7d0;
}

.slt-status.cancelling {
  color: var(--slt-orange);
  background: var(--slt-orange-soft);
  border-color: #fed7aa;
}

.slt-status.cancelled {
  color: var(--slt-gray);
  background: var(--slt-gray-soft);
  border-color: #e2e8f0;
}

.slt-status.finished {
  color: var(--slt-green);
  background: var(--slt-green-soft);
  border-color: #bbf7d0;
}

.slt-status.failed {
  color: var(--slt-red);
  background: var(--slt-red-soft);
  border-color: #fecaca;
}

.slt-exception {
  color: var(--slt-red);
  white-space: pre-wrap;
  word-break: break-word;
  overflow-wrap: anywhere;
}

@media (prefers-color-scheme: dark) {
  .slt-progress {
    background: #334155;
    border-color: #475569;
  }

  .slt-progress-label {
    color: #f8fafc;
  }
}
"""


def dashboard_css() -> str:
    """CSS for :class:`~slothpy.core.slt_dashboard` session dashboards."""
    return (
        "<style>"
        + _slt_theme_css(".slt-dashboard")
        + _slt_components_css()
        + _dashboard_only_css()
        + "</style>"
    )


def structure_css() -> str:
    """CSS for file/group/dataset/attributes structure views."""
    return (
        "<style>"
        + _slt_theme_css(".slt-structure")
        + _slt_components_css()
        + "</style>"
    )


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------


def _e(value: object) -> str:
    return escape(str(value), quote=True)


def _repr_value(value: object) -> str:
    return _e(repr(value))


def _wrap_structure(body: str) -> str:
    return structure_css() + f"<div class='slt-structure'>{body}</div>"


def _meta_pills(*items: tuple[str, str]) -> str:
    if not items:
        return ""
    pills = "".join(
        f"<span class='slt-meta-pill {css}'>{_e(label)}</span>"
        for label, css in items
    )
    return f"<div class='slt-header-meta'>{pills}</div>"


def _section(title: str, content: str) -> str:
    return (
        f"<div class='slt-section'>"
        f"<h4 class='slt-section-title'>{_e(title)}</h4>"
        f"{content}"
        f"</div>"
    )


def _empty_block(text: str = "(none)") -> str:
    return f"<div class='slt-empty'>{_e(text)}</div>"


def _kv_table(rows: list[tuple[str, str]]) -> str:
    if not rows:
        return _empty_block()

    body = "".join(
        f"<tr><th scope='row' class='slt-mono'>{_e(key)}</th>"
        f"<td class='slt-cell-wrap'>{value}</td></tr>"
        for key, value in rows
    )
    return (
        "<div class='slt-table-wrap'>"
        "<table class='slt-table'>"
        "<tbody>"
        f"{body}"
        "</tbody></table></div>"
    )


def _attrs_table(attrs: dict[str, Any]) -> str:
    rows = [(key, _repr_value(value)) for key, value in attrs.items()]
    return _kv_table(rows)


def _attr_chips(attrs: dict[str, Any]) -> str:
    if not attrs:
        return "<span class='slt-muted'>(none)</span>"

    chips = "".join(
        "<span class='slt-attr-chip'>"
        f"<span class='slt-attr-key'>{_e(key)}</span>"
        f"{_repr_value(value)}"
        "</span>"
        for key, value in attrs.items()
    )
    return f"<div class='slt-attr-chips'>{chips}</div>"


def _dimensions_table(dimensions: dict[str, int]) -> str:
    if not dimensions:
        return _empty_block()

    rows = [
        (name, f"<span class='slt-mono slt-right'>{size}</span>")
        for name, size in dimensions.items()
    ]
    body = "".join(
        f"<tr><th scope='row' class='slt-mono'>{_e(name)}</th><td>{value}</td></tr>"
        for name, value in rows
    )
    return (
        "<div class='slt-table-wrap'>"
        "<table class='slt-table'>"
        "<thead><tr><th>Dimension</th><th class='slt-right'>Size</th></tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


def _variables_table(variables: tuple[SltVariableNode, ...]) -> str:
    if not variables:
        return _empty_block()

    rows: list[str] = []
    for variable in variables:
        dims = ", ".join(variable.dims)
        shape = ", ".join(str(size) for size in variable.shape)
        kind_label = "Coordinate" if variable.kind == "coordinate" else "Data variable"
        kind_css = "cyan" if variable.kind == "coordinate" else "magenta"
        primary = (
            "<span class='slt-primary-mark'>primary</span>" if variable.primary else ""
        )
        rows.append(
            "<tr>"
            f"<td class='slt-mono'><strong>{_e(variable.name)}</strong>{primary}</td>"
            f"<td><span class='slt-meta-pill {kind_css}'>{_e(kind_label)}</span></td>"
            f"<td class='slt-mono slt-muted'>({ _e(dims) })</td>"
            f"<td class='slt-mono slt-right'>({ _e(shape) })</td>"
            f"<td class='slt-mono'>{_e(variable.dtype)}</td>"
            f"<td>{_attr_chips(variable.attrs)}</td>"
            "</tr>"
        )

    return (
        "<div class='slt-table-wrap'>"
        "<table class='slt-table'>"
        "<thead><tr>"
        "<th>Name</th><th>Kind</th><th>Dims</th>"
        "<th class='slt-right'>Shape</th><th>Dtype</th><th>Attributes</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _dataset_card(node: SltDatasetNode) -> str:
    meta = _meta_pills(
        (f"shape={node.shape}", "green"),
        (f"dtype={node.dtype}", "magenta"),
    )
    sections = [_section("Attributes", _attrs_table(node.attrs))]
    return (
        "<div class='slt-card'>"
        "<div class='slt-header slt-header-magenta'>"
        "<div>"
        "<h3 class='slt-title dataset'>Dataset</h3>"
        f"<div class='slt-subtitle slt-mono'>{_e(node.name)}"
        f" <span class='slt-muted'>·</span> {_e(node.path)}</div>"
        "</div>"
        f"{meta}"
        "</div>"
        f"{''.join(sections)}"
        "</div>"
    )


def _group_sections(node: SltGroupNode) -> str:
    sections: list[str] = []
    sections.append(_section("Attributes", _attrs_table(node.attrs)))

    if not node.readable:
        sections.append(
            _section(
                "Error",
                f"<p class='slt-error slt-cell-wrap'>{_e(node.error or 'Unknown error')}</p>",
            )
        )
        return "".join(sections)

    if not node.is_slothpy:
        if node.raw_datasets:
            dataset_cards = "".join(_dataset_card(dataset) for dataset in node.raw_datasets)
            sections.append(
                _section("Datasets", f"<div class='slt-stack'>{dataset_cards}</div>")
            )
        else:
            sections.append(_section("Datasets", _empty_block()))

        if node.child_groups:
            chips = "".join(
                f"<span class='slt-chip'>{_e(name)}</span>" for name in node.child_groups
            )
            sections.append(_section("Child groups", f"<div class='slt-chip-list'>{chips}</div>"))
        else:
            sections.append(_section("Child groups", _empty_block()))

        if not node.raw_datasets and not node.child_groups:
            sections.append(_section("Contents", _empty_block("(empty)")))

        return "".join(sections)

    sections.append(_section("Dimensions", _dimensions_table(node.dimensions)))
    sections.append(_section("Coordinates", _variables_table(node.coordinates)))
    sections.append(_section("Data variables", _variables_table(node.data_variables)))
    return "".join(sections)


def _group_header_meta(node: SltGroupNode) -> list[tuple[str, str]]:
    meta: list[tuple[str, str]] = []
    if not node.readable:
        meta.append(("unreadable", "red"))
        return meta

    if node.is_slothpy:
        slt_type = node.attrs.get("slt_type")
        if slt_type is not None:
            meta.append((f"Type={slt_type}", "yellow"))
        if node.primary is not None:
            meta.append((f"Primary={node.primary}", "green"))
    else:
        meta.append(("raw HDF5", "gray"))

    return meta


def _group_card_html(node: SltGroupNode) -> str:
    meta = _meta_pills(*_group_header_meta(node))
    return (
        "<div class='slt-card'>"
        "<div class='slt-header slt-header-blue'>"
        "<div>"
        "<h3 class='slt-title group'>Group</h3>"
        f"<div class='slt-subtitle slt-mono'>{_e(node.name)}"
        f" <span class='slt-muted'>·</span> {_e(node.path)}</div>"
        "</div>"
        f"{meta}"
        "</div>"
        f"{_group_sections(node)}"
        "</div>"
    )


def group_node_to_html(node: SltGroupNode) -> str:
    """Render a structured group node as an HTML dashboard fragment."""
    return _wrap_structure(_group_card_html(node))


def dataset_node_to_html(node: SltDatasetNode) -> str:
    """Render a raw dataset node as an HTML dashboard fragment."""
    return _wrap_structure(_dataset_card(node))


def attrs_mapping_to_html(
    attrs: dict[str, Any],
    *,
    title: str = "Attributes",
    subtitle: str | None = None,
) -> str:
    """Render an attributes mapping as an HTML dashboard fragment."""
    subtitle_html = (
        f"<div class='slt-subtitle slt-mono'>{_e(subtitle)}</div>" if subtitle else ""
    )
    body = (
        "<div class='slt-card'>"
        "<div class='slt-header slt-header-yellow'>"
        "<div>"
        f"<h3 class='slt-title attrs'>{_e(title)}</h3>"
        f"{subtitle_html}"
        "</div>"
        "</div>"
        f"{_section('Entries', _attrs_table(attrs))}"
        "</div>"
    )
    return _wrap_structure(body)


def proxy_group_to_html(*, group_name: str, file_path: Path) -> str:
    """Render a non-existent proxy group handle."""
    body = (
        "<div class='slt-card'>"
        "<div class='slt-header slt-header-blue'>"
        "<div>"
        "<h3 class='slt-title group'>Proxy group</h3>"
        f"<div class='slt-subtitle'>"
        f"<span class='slt-mono'>{_e(group_name)}</span> in "
        f"<span class='slt-mono'>{_e(file_path)}</span> does not exist."
        "</div>"
        "</div>"
        f"{_meta_pills(('missing', 'red'))}"
        "</div>"
        "</div>"
    )
    return _wrap_structure(body)


def file_node_to_html(node: SltFileNode) -> str:
    """Render a structured file node tree as an HTML dashboard fragment."""
    meta_items: list[tuple[str, str]] = []
    version = node.attrs.get("format_version")
    if version is not None:
        meta_items.append((f"version={version}", "yellow"))
    meta = _meta_pills(*meta_items)

    inner_sections: list[str] = []
    if node.attrs:
        inner_sections.append(_section("File attributes", _attrs_table(node.attrs)))

    if node.datasets:
        dataset_cards = "".join(_dataset_card(dataset) for dataset in node.datasets)
        inner_sections.append(
            _section("Root datasets", f"<div class='slt-stack'>{dataset_cards}</div>")
        )
    elif not node.groups:
        inner_sections.append(_section("Contents", _empty_block("(empty)")))

    if node.groups:
        group_cards = "".join(_group_card_html(group) for group in node.groups)
        inner_sections.append(
            _section("Groups", f"<div class='slt-stack'>{group_cards}</div>")
        )

    body = (
        "<div class='slt-card'>"
        "<div class='slt-header'>"
        "<div>"
        "<h3 class='slt-title file'>SltFile</h3>"
        f"<div class='slt-subtitle slt-mono'>{_e(node.path)}</div>"
        "</div>"
        f"{meta}"
        "</div>"
        f"{''.join(inner_sections)}"
        "</div>"
    )
    return _wrap_structure(body)