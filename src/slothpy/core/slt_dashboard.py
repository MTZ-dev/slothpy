"""SlothPy compute-session dashboard rendering and live terminal display."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from html import escape
from io import StringIO
from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from slothpy.core.slt_html import dashboard_css
from slothpy.core.slt_session import (
    SltAllocation,
    SltJob,
    SltJobSnapshot,
    SltJobStatus,
    SltSession,
    SltSessionSnapshot,
)

# Virtual terminal width for plain-text export (not the live TTY width).
_RICH_VIRTUAL_TERMINAL_WIDTH = 10000

def _dashboard_jobs_done(
    session: SltSession[Any],
    jobs: Sequence[SltJob[Any]] | None,
) -> bool:
    tracked = list(jobs) if jobs is not None else list(session.jobs().values())
    if not tracked:
        return True
    return all(job.done() for job in tracked)


def _print_live_render(
    console: Console,
    live_render: Any,
    renderable: Any,
) -> None:
    """
    Overwrite the previous dashboard frame in the terminal.

    Uses :class:`rich.live.LiveRender` cursor controls directly instead of
    :class:`rich.live.Live`, because Live's exit ``refresh()`` can leave a
    duplicated panel title when cursor restore is off by one line.
    """
    from rich.live import LiveRender

    if not isinstance(live_render, LiveRender):
        raise TypeError("live_render must be a rich.live.LiveRender instance.")

    live_render.set_renderable(renderable)
    live_render.vertical_overflow = "visible"

    with console:
        if console.is_terminal and live_render.last_render_height:
            console.control(live_render.position_cursor())
        console.print(live_render)


def run_session_dashboard(
    session: SltSession[Any],
    *,
    interval: float,
    once: bool,
    exit_when_done: bool,
    jobs: Sequence[SltJob[Any]] | None,
    console: Console | None,
    width: int | None,
) -> None:
    from rich.live import LiveRender
    from rich.text import Text

    from slothpy.core.slt_common import print_rich_renderable

    if not once and interval <= 0:
        raise ValueError("interval must be > 0 unless once=True")

    terminal = console or Console()

    if once:
        print_rich_renderable(
            session_snapshot_to_rich(session.snapshot(), width=width),
            console=terminal,
        )
        return

    interrupted = False
    live_render = LiveRender(Text(""))

    try:
        while True:
            _print_live_render(
                terminal,
                live_render,
                session_snapshot_to_rich(session.snapshot(), width=width),
            )
            if exit_when_done and _dashboard_jobs_done(session, jobs):
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        interrupted = True
        _interrupt_session_on_keyboard_interrupt(session)
    finally:
        terminal.show_cursor(True)

    if interrupted:
        raise KeyboardInterrupt


async def run_session_dashboard_async(
    session: SltSession[Any],
    *,
    interval: float,
    once: bool,
    exit_when_done: bool,
    jobs: Sequence[SltJob[Any]] | None,
    console: Console | None,
    width: int | None,
) -> None:
    from rich.live import LiveRender
    from rich.text import Text

    from slothpy.core.slt_common import print_rich_renderable

    if not once and interval <= 0:
        raise ValueError("interval must be > 0 unless once=True")

    terminal = console or Console()

    if once:
        print_rich_renderable(
            session_snapshot_to_rich(session.snapshot(), width=width),
            console=terminal,
        )
        return

    interrupted = False
    live_render = LiveRender(Text(""))

    try:
        while True:
            _print_live_render(
                terminal,
                live_render,
                session_snapshot_to_rich(session.snapshot(), width=width),
            )
            if exit_when_done and _dashboard_jobs_done(session, jobs):
                break
            await asyncio.sleep(interval)
    except KeyboardInterrupt:
        interrupted = True
        _interrupt_session_on_keyboard_interrupt(session)
    finally:
        terminal.show_cursor(True)

    if interrupted:
        raise KeyboardInterrupt


def _interrupt_session_on_keyboard_interrupt(session: SltSession[Any]) -> None:
    """
    Ensure MPI workers are killed when the dashboard loop catches KeyboardInterrupt.

    If signal handlers are installed, they usually run first; this call is idempotent.
    """
    from slothpy.core.slt_interrupt import interrupt_session

    interrupt_session(session, hard=True)


# ---------------------------------------------------------------------------
# Rich / HTML rendering helpers
# ---------------------------------------------------------------------------


# This is not a preferred dashboard width. It is a virtual "infinite enough"
# terminal width used only when converting Rich output to plain text.
# Direct Console().print(session) still uses the user's real terminal width.
_RICH_VIRTUAL_TERMINAL_WIDTH = 10000


def _rich_console(file: StringIO | None = None) -> Console:
    return Console(
        file=file or StringIO(),
        force_terminal=False,
        color_system="auto",
        width=_RICH_VIRTUAL_TERMINAL_WIDTH,
        soft_wrap=False,
    )


def _rich_to_ansi(renderable: Any) -> str:
    stream = StringIO()
    _rich_console(stream).print(renderable)
    return stream.getvalue().rstrip()


def _rich_to_html(renderable: Any) -> str:
    """
    Render a Rich object to HTML.

    This is kept for simple Rich objects and terminal-like debugging output.
    The live session dashboard uses dedicated CSS/HTML instead, because Rich
    tables exported as <pre> blocks can produce misaligned borders in notebook
    frontends.
    """
    console = Console(
        file=StringIO(),
        record=True,
        force_terminal=True,
        color_system="truecolor",
        width=120,
    )
    console.print(renderable)

    body = console.export_html(
        inline_styles=True,
        code_format=(
            "<pre style='white-space: pre; "
            "font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "
            "Liberation Mono, monospace; margin: 0; overflow-x: auto;'>{code}</pre>"
        ),
    ).rstrip()

    return (
        f"<div style='max-width:100%; overflow-x:auto; overflow-y:hidden;'>{body}</div>"
    )


def _progress_bar(fraction: float | None, *, width: int = 24) -> str:
    if fraction is None:
        return "░" * width

    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(width * fraction))
    return "█" * filled + "░" * (width - filled)


def _html_escape(value: object) -> str:
    return escape(str(value), quote=True)


def _fraction_to_percent(fraction: float | None) -> float:
    if fraction is None:
        return 0.0
    return 100.0 * max(0.0, min(1.0, fraction))


def _html_progress_bar(
    fraction: float | None,
    *,
    label: str | None = None,
    muted: bool = False,
) -> str:
    percent = _fraction_to_percent(fraction)
    label_text = "" if label is None else _html_escape(label)
    muted_class = " slt-progress-muted" if muted else ""

    return (
        f"<div class='slt-progress{muted_class}'>"
        f"  <div class='slt-progress-fill' style='width: {percent:.3f}%;'></div>"
        f"  <div class='slt-progress-label'>{label_text}</div>"
        f"</div>"
    )


def _status_style(status: SltJobStatus) -> str:
    if status == SltJobStatus.QUEUED:
        return "yellow"
    if status == SltJobStatus.RUNNING:
        return "green"
    if status == SltJobStatus.CANCELLING:
        return "orange1"
    if status == SltJobStatus.CANCELLED:
        return "bright_black"
    if status == SltJobStatus.FINISHED:
        return "bold green"
    if status == SltJobStatus.FAILED:
        return "bold red"
    return "default"


def _status_css_class(status: SltJobStatus) -> str:
    return _html_escape(status.value)


def _resource_table(snapshot: SltSessionSnapshot) -> Table:
    """
    Terminal/Rich resource table.

    The table uses its natural width. It does not fold, crop, or ellipsize.
    """
    table = Table(
        title="Resources",
        expand=False,
        show_lines=False,
        padding=(0, 1),
    )

    table.add_column("Node", style="cyan", no_wrap=True, overflow="ignore")
    table.add_column("Total cores", justify="right", no_wrap=True, overflow="ignore")
    table.add_column("Used", justify="right", no_wrap=True, overflow="ignore")
    table.add_column("Free", justify="right", no_wrap=True, overflow="ignore")
    table.add_column("Usage", no_wrap=True, overflow="ignore")
    table.add_column("Exclusive", justify="center", no_wrap=True, overflow="ignore")

    for node in snapshot.resources:
        fraction = node.used_cores / node.total_cores if node.total_cores > 0 else None
        table.add_row(
            node.node_name,
            str(node.total_cores),
            str(node.used_cores),
            str(node.free_cores),
            _progress_bar(fraction),
            "yes" if node.exclusive else "",
        )

    return table


def _job_allocation_text(allocation: SltAllocation | None) -> str:
    if allocation is None:
        return ""

    parts = [
        f"{node.node_name}: {node.ranks}×{node.threads_per_rank}"
        for node in allocation.nodes
    ]
    return ", ".join(parts)


def _job_resources_text(job: SltJobSnapshot) -> str:
    allocation_text = _job_allocation_text(job.allocation)
    if job.autotune_note:
        if allocation_text:
            return f"{allocation_text} ({job.autotune_note})"
        return job.autotune_note
    return allocation_text


def _job_progress_text(job: SltJobSnapshot) -> str:
    if job.status == SltJobStatus.FINISHED:
        return f"{_progress_bar(1.0)} 100.0%"

    progress = job.progress
    if progress is None:
        return _progress_bar(None)

    percent = progress.percent
    if percent is None:
        return f"{_progress_bar(None)} {progress.done}/{progress.total}"

    return (
        f"{_progress_bar(progress.fraction)} "
        f"{percent:5.1f}% "
        f"({progress.done}/{progress.total})"
    )


def _jobs_table(snapshot: SltSessionSnapshot) -> Table:
    """
    Terminal/Rich jobs table.

    The table uses its natural width. It does not fold, crop, or ellipsize.
    Extremely long text makes the table extremely wide.
    """
    table = Table(
        title="Jobs",
        expand=False,
        show_lines=False,
        padding=(0, 1),
    )

    table.add_column("Job id", style="cyan", no_wrap=True, overflow="ignore")
    table.add_column("Computation", style="magenta", no_wrap=True, overflow="ignore")
    table.add_column("Status", justify="center", no_wrap=True, overflow="ignore")
    table.add_column("Resources", no_wrap=True, overflow="ignore")
    table.add_column("Progress", no_wrap=True, overflow="ignore")
    table.add_column("Exception", style="red", no_wrap=True, overflow="ignore")

    for job in snapshot.jobs:
        table.add_row(
            job.job_id,
            job.name,
            Text(job.status.value, style=_status_style(job.status), no_wrap=True),
            _job_resources_text(job),
            _job_progress_text(job),
            job.exception or "",
        )

    if not snapshot.jobs:
        table.add_row("", "", Text("no jobs", style="bright_black"), "", "", "")

    return table


def _summary_text(snapshot: SltSessionSnapshot) -> Text:
    status = "closed" if snapshot.closed else "active"

    return Text.assemble(
        ("SlothPy session", "bold red"),
        ("  "),
        (f"[{status}]", "bright_black"),
        ("\n"),
        (f"queued={snapshot.queued}", "yellow"),
        ("  "),
        (f"running={snapshot.running}", "green"),
        ("  "),
        (f"cancelling={snapshot.cancelling}", "orange1"),
        ("  "),
        (f"finished={snapshot.finished}", "bold green"),
        ("  "),
        (f"failed={snapshot.failed}", "bold red"),
        ("  "),
        (f"cancelled={snapshot.cancelled}", "bright_black"),
    )


def session_snapshot_to_rich(
    snapshot: SltSessionSnapshot,
    *,
    width: int | None = None,
) -> Panel:
    return Panel(
        Group(
            _summary_text(snapshot),
            _resource_table(snapshot),
            _jobs_table(snapshot),
        ),
        title="SlothPy compute session",
        border_style="red",
        expand=False,
        width=width,
    )


def _summary_badge_html(
    *,
    css_class: str,
    label: str,
    value: int,
) -> str:
    return (
        f"<span class='slt-badge {css_class}'>"
        f"<span>{_html_escape(label)}</span>"
        f"<span class='slt-badge-number'>{value}</span>"
        f"</span>"
    )


def _resources_html(snapshot: SltSessionSnapshot) -> str:
    rows: list[str] = []

    for node in snapshot.resources:
        fraction = node.used_cores / node.total_cores if node.total_cores > 0 else None
        percent = _fraction_to_percent(fraction)
        progress_label = f"{percent:.1f}%"

        rows.append(
            "<tr>"
            f"<td class='slt-mono'>{_html_escape(node.node_name)}</td>"
            f"<td class='slt-right'>{node.total_cores}</td>"
            f"<td class='slt-right'>{node.used_cores}</td>"
            f"<td class='slt-right'>{node.free_cores}</td>"
            f"<td>{_html_progress_bar(fraction, label=progress_label, muted=node.used_cores == 0)}</td>"
            f"<td class='slt-center'>{'yes' if node.exclusive else ''}</td>"
            "</tr>"
        )

    if not rows:
        rows.append(
            "<tr><td colspan='6' class='slt-empty'>No resources registered.</td></tr>"
        )

    return (
        "<section class='slt-section'>"
        "<h3 class='slt-section-title'>Resources</h3>"
        "<div class='slt-table-wrap'>"
        "<table class='slt-table'>"
        "<thead>"
        "<tr>"
        "<th>Node</th>"
        "<th class='slt-right'>Total cores</th>"
        "<th class='slt-right'>Used</th>"
        "<th class='slt-right'>Free</th>"
        "<th>Usage</th>"
        "<th class='slt-center'>Exclusive</th>"
        "</tr>"
        "</thead>"
        "<tbody>" + "".join(rows) + "</tbody>"
        "</table>"
        "</div>"
        "</section>"
    )


def _job_progress_html(job: SltJobSnapshot) -> str:
    if job.status == SltJobStatus.FINISHED:
        return _html_progress_bar(1.0, label="100.0%")

    progress = job.progress

    if progress is None:
        if job.status == SltJobStatus.QUEUED:
            return _html_progress_bar(None, label="waiting", muted=True)
        return _html_progress_bar(None, label="", muted=True)

    if progress.percent is None:
        label = f"{progress.done}/{progress.total}"
    else:
        label = f"{progress.percent:.1f}% ({progress.done}/{progress.total})"

    return _html_progress_bar(progress.fraction, label=label)


def _jobs_html(snapshot: SltSessionSnapshot) -> str:
    rows: list[str] = []

    for job in snapshot.jobs:
        status_class = _status_css_class(job.status)

        rows.append(
            "<tr>"
            f"<td class='slt-mono'>{_html_escape(job.job_id)}</td>"
            f"<td class='slt-cell-wrap'>{_html_escape(job.name)}</td>"
            f"<td><span class='slt-status {status_class}'>{_html_escape(job.status.value)}</span></td>"
            f"<td class='slt-muted slt-cell-wrap'>{_html_escape(_job_resources_text(job))}</td>"
            f"<td>{_job_progress_html(job)}</td>"
            f"<td><div class='slt-exception slt-cell-wrap'>{_html_escape(job.exception or '')}</div></td>"
            "</tr>"
        )

    if not rows:
        rows.append(
            "<tr><td colspan='6' class='slt-empty'>No jobs submitted.</td></tr>"
        )

    return (
        "<section class='slt-section'>"
        "<h3 class='slt-section-title'>Jobs</h3>"
        "<div class='slt-table-wrap'>"
        "<table class='slt-table'>"
        "<thead>"
        "<tr>"
        "<th>Job id</th>"
        "<th>Computation</th>"
        "<th>Status</th>"
        "<th>Resources</th>"
        "<th>Progress</th>"
        "<th>Exception</th>"
        "</tr>"
        "</thead>"
        "<tbody>" + "".join(rows) + "</tbody>"
        "</table>"
        "</div>"
        "</section>"
    )


def session_snapshot_to_html(snapshot: SltSessionSnapshot) -> str:
    status = "closed" if snapshot.closed else "active"
    dot_class = "closed" if snapshot.closed else ""

    badges = "".join(
        [
            _summary_badge_html(
                css_class="queued",
                label="queued",
                value=snapshot.queued,
            ),
            _summary_badge_html(
                css_class="running",
                label="running",
                value=snapshot.running,
            ),
            _summary_badge_html(
                css_class="cancelling",
                label="cancelling",
                value=snapshot.cancelling,
            ),
            _summary_badge_html(
                css_class="finished",
                label="finished",
                value=snapshot.finished,
            ),
            _summary_badge_html(
                css_class="failed",
                label="failed",
                value=snapshot.failed,
            ),
            _summary_badge_html(
                css_class="cancelled",
                label="cancelled",
                value=snapshot.cancelled,
            ),
        ]
    )

    return (
        dashboard_css() + "<div class='slt-dashboard'>"
        "<div class='slt-card'>"
        "<header class='slt-header'>"
        "<div>"
        "<h2 class='slt-title'>SlothPy compute session</h2>"
        "<div class='slt-subtitle'>Asynchronous MPI/resource scheduler</div>"
        "</div>"
        f"<div class='slt-status-pill'><span class='slt-dot {dot_class}'></span>{_html_escape(status)}</div>"
        "</header>"
        f"<div class='slt-badges'>{badges}</div>"
        f"{_resources_html(snapshot)}"
        f"{_jobs_html(snapshot)}"
        "</div>"
        "</div>"
    )


def session_snapshot_to_text(
    snapshot: SltSessionSnapshot,
    *,
    width: int | None = None,
) -> str:
    """Render a session snapshot as an ANSI-colored plain-text dashboard."""
    return _rich_to_ansi(session_snapshot_to_rich(snapshot, width=width))


__all__ = [
    "run_session_dashboard",
    "run_session_dashboard_async",
    "session_snapshot_to_html",
    "session_snapshot_to_rich",
    "session_snapshot_to_text",
]
