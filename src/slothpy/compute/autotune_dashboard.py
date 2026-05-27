"""Rich and HTML summaries for :class:`~slothpy.compute.autotune.AutotuneResult`."""

from __future__ import annotations

from html import escape
from io import StringIO
import numpy as np
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from slothpy.compute.autotune import (
    AutotuneResult,
    AutotuneSearchSnapshot,
    AutotuneTrial,
)
from slothpy.core.slt_html import dashboard_css


def _format_seconds(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.2f} s"


def _html_escape(value: object) -> str:
    return escape(str(value), quote=True)


def _trial_outcome_label(trial: AutotuneTrial) -> str:
    if trial.timed_out:
        return "timeout"
    if trial.skipped:
        return "skipped"
    if trial.improved:
        return "best"
    return "worse"


def _trial_outcome_css(trial: AutotuneTrial) -> str:
    if trial.timed_out:
        return "failed"
    if trial.skipped:
        return "cancelled"
    if trial.improved:
        return "finished"
    return "queued"


def _trial_outcome_style(trial: AutotuneTrial) -> str:
    if trial.timed_out:
        return "bold red"
    if trial.skipped:
        return "bright_black"
    if trial.improved:
        return "bold green"
    return "yellow"


def _count_trials(result: AutotuneResult) -> dict[str, int]:
    trials = result.trials
    return {
        "total": len(trials),
        "skipped": sum(1 for trial in trials if trial.skipped),
        "timed_out": sum(1 for trial in trials if trial.timed_out),
        "improved": sum(
            1 for trial in trials if trial.improved and not trial.skipped
        ),
    }


def _summary_text(result: AutotuneResult) -> Text:
    counts = _count_trials(result)
    estimate = _format_seconds(result.estimated_seconds)

    return Text.assemble(
        ("MPI/thread autotune\n", "bold"),
        (
            f"selected: {result.num_processes} rank(s) × "
            f"{result.num_threads} thread(s)  "
            f"({result.logical_cpus} logical CPUs)\n",
        ),
        (f"estimated main-loop time: {estimate}\n", "green"),
        (
            f"cpu budget: {result.num_cpu}  nodes: {result.nodes}  "
            f"trials: {counts['total']}  "
            f"skipped: {counts['skipped']}  "
            f"timed out: {counts['timed_out']}\n",
            "bright_black",
        ),
    )


def _search_status_style(status: str) -> str:
    if status == "searching":
        return "bold cyan"
    if status == "aborted":
        return "bold red"
    return "bold green"


def _search_status_css(status: str) -> str:
    if status == "searching":
        return "running"
    if status == "aborted":
        return "failed"
    return "finished"


def _search_status_label(status: str) -> str:
    if status == "searching":
        return "searching"
    if status == "aborted":
        return "aborted"
    return "done"


def _search_summary_text(snapshot: AutotuneSearchSnapshot) -> Text:
    counts = _count_trials(snapshot.to_result())
    estimate = _format_seconds(snapshot.best_time)
    status = _search_status_label(snapshot.status)

    lines: list[tuple[str, str | None]] = [
        ("MPI/thread autotune\n", "bold"),
        (
            f"status: {status}  cpu budget: {snapshot.num_cpu}  "
            f"nodes: {snapshot.nodes}\n",
            _search_status_style(snapshot.status),
        ),
    ]

    if snapshot.current_processes is not None and snapshot.current_threads is not None:
        lines.append(
            (
                (
                    f"benchmarking: {snapshot.current_processes} rank(s) × "
                    f"{snapshot.current_threads} thread(s)\n"
                ),
                "bold yellow",
            )
        )

    lines.extend(
        [
            (
                (
                    f"best so far: {snapshot.best_processes} rank(s) × "
                    f"{snapshot.best_threads} thread(s)  "
                    f"({snapshot.best_processes * snapshot.best_threads} CPUs)  "
                    f"estimate {estimate}\n"
                ),
                "green",
            ),
            (
                (
                    f"trials: {counts['total']}  skipped: {counts['skipped']}  "
                    f"timed out: {counts['timed_out']}\n"
                ),
                "bright_black",
            ),
        ]
    )

    if snapshot.message:
        lines.append((f"{snapshot.message}\n", "yellow"))

    return Text.assemble(*lines)


def _trials_table_from_trials(
    trials: tuple[AutotuneTrial, ...],
    *,
    current_processes: int | None = None,
    current_threads: int | None = None,
) -> Table:
    table = Table(
        title="Trials",
        expand=False,
        show_lines=False,
        padding=(0, 1),
    )

    table.add_column("Ranks", justify="right", no_wrap=True)
    table.add_column("Threads", justify="right", no_wrap=True)
    table.add_column("Estimate", justify="right", no_wrap=True)
    table.add_column("Outcome", no_wrap=True)
    table.add_column("Note", overflow="fold")

    if not trials and current_processes is None:
        table.add_row("—", "—", "—", "—", "No trials recorded.")
        return table

    for trial in trials:
        outcome = _trial_outcome_label(trial)
        note = trial.skip_reason or ""
        estimate = (
            "—"
            if trial.skipped
            else _format_seconds(trial.estimated_seconds)
        )

        table.add_row(
            str(trial.num_processes),
            str(trial.num_threads),
            Text(estimate, style=_trial_outcome_style(trial)),
            Text(outcome, style=_trial_outcome_style(trial)),
            note,
        )

    if current_processes is not None and current_threads is not None:
        table.add_row(
            str(current_processes),
            str(current_threads),
            "—",
            Text("running", style="bold cyan"),
            "MPI benchmark in progress",
        )

    return table


def autotune_search_snapshot_to_rich(
    snapshot: AutotuneSearchSnapshot,
    *,
    width: int | None = None,
) -> Panel:
    return Panel(
        Group(
            _search_summary_text(snapshot),
            _trials_table_from_trials(
                snapshot.trials,
                current_processes=snapshot.current_processes,
                current_threads=snapshot.current_threads,
            ),
        ),
        title="SlothPy autotune",
        border_style="red",
        expand=False,
        width=width,
    )


def _trials_table(result: AutotuneResult) -> Table:
    table = Table(
        title="Trials",
        expand=False,
        show_lines=False,
        padding=(0, 1),
    )

    table.add_column("Ranks", justify="right", no_wrap=True)
    table.add_column("Threads", justify="right", no_wrap=True)
    table.add_column("Estimate", justify="right", no_wrap=True)
    table.add_column("Outcome", no_wrap=True)
    table.add_column("Note", overflow="fold")

    if not result.trials:
        table.add_row("—", "—", "—", "—", "No trials recorded.")
        return table

    for trial in result.trials:
        outcome = _trial_outcome_label(trial)
        note = trial.skip_reason or ""
        estimate = (
            "—"
            if trial.skipped
            else _format_seconds(trial.estimated_seconds)
        )

        table.add_row(
            str(trial.num_processes),
            str(trial.num_threads),
            Text(estimate, style=_trial_outcome_style(trial)),
            Text(outcome, style=_trial_outcome_style(trial)),
            note,
        )

    return table


def autotune_result_to_rich(
    result: AutotuneResult,
    *,
    width: int | None = None,
) -> Panel:
    snapshot = AutotuneSearchSnapshot(
        num_cpu=result.num_cpu,
        nodes=result.nodes,
        trials=result.trials,
        best_processes=result.num_processes,
        best_threads=result.num_threads,
        best_time=result.estimated_seconds,
        status="done",
    )
    return autotune_search_snapshot_to_rich(snapshot, width=width)


def _summary_badge_html(*, css_class: str, label: str, value: int) -> str:
    return (
        f"<span class='slt-badge {css_class}'>"
        f"<span>{_html_escape(label)}</span>"
        f"<span class='slt-badge-number'>{value}</span>"
        f"</span>"
    )


def _selection_html(result: AutotuneResult) -> str:
    estimate = _format_seconds(result.estimated_seconds)
    return (
        "<section class='slt-section'>"
        "<h3 class='slt-section-title'>Selected configuration</h3>"
        "<div class='slt-table-wrap'>"
        "<table class='slt-table'>"
        "<tbody>"
        "<tr>"
        "<th class='slt-right'>MPI ranks</th>"
        f"<td class='slt-right slt-mono'>{result.num_processes}</td>"
        "</tr>"
        "<tr>"
        "<th class='slt-right'>Threads / rank</th>"
        f"<td class='slt-right slt-mono'>{result.num_threads}</td>"
        "</tr>"
        "<tr>"
        "<th class='slt-right'>Logical CPUs</th>"
        f"<td class='slt-right slt-mono'>{result.logical_cpus}</td>"
        "</tr>"
        "<tr>"
        "<th class='slt-right'>CPU budget</th>"
        f"<td class='slt-right slt-mono'>{result.num_cpu}</td>"
        "</tr>"
        "<tr>"
        "<th class='slt-right'>Nodes</th>"
        f"<td class='slt-right slt-mono'>{result.nodes}</td>"
        "</tr>"
        "<tr>"
        "<th class='slt-right'>Est. main-loop time</th>"
        f"<td class='slt-right slt-mono'>{_html_escape(estimate)}</td>"
        "</tr>"
        "</tbody>"
        "</table>"
        "</div>"
        "</section>"
    )


def _trials_html(result: AutotuneResult) -> str:
    rows: list[str] = []

    for trial in result.trials:
        outcome_class = _trial_outcome_css(trial)
        outcome = _trial_outcome_label(trial)
        estimate = (
            "—"
            if trial.skipped
            else _html_escape(_format_seconds(trial.estimated_seconds))
        )

        rows.append(
            "<tr>"
            f"<td class='slt-right slt-mono'>{trial.num_processes}</td>"
            f"<td class='slt-right slt-mono'>{trial.num_threads}</td>"
            f"<td class='slt-right'>{estimate}</td>"
            f"<td><span class='slt-status {outcome_class}'>{_html_escape(outcome)}</span></td>"
            f"<td class='slt-muted slt-cell-wrap'>{_html_escape(trial.skip_reason or '')}</td>"
            "</tr>"
        )

    if not rows:
        rows.append(
            "<tr><td colspan='5' class='slt-empty'>No trials recorded.</td></tr>"
        )

    return (
        "<section class='slt-section'>"
        "<h3 class='slt-section-title'>Trials</h3>"
        "<div class='slt-table-wrap'>"
        "<table class='slt-table'>"
        "<thead>"
        "<tr>"
        "<th class='slt-right'>Ranks</th>"
        "<th class='slt-right'>Threads</th>"
        "<th class='slt-right'>Estimate</th>"
        "<th>Outcome</th>"
        "<th>Note</th>"
        "</tr>"
        "</thead>"
        "<tbody>" + "".join(rows) + "</tbody>"
        "</table>"
        "</div>"
        "</section>"
    )


def _trials_html_from_snapshot(snapshot: AutotuneSearchSnapshot) -> str:
    rows: list[str] = []

    for trial in snapshot.trials:
        outcome_class = _trial_outcome_css(trial)
        outcome = _trial_outcome_label(trial)
        estimate = (
            "—"
            if trial.skipped
            else _html_escape(_format_seconds(trial.estimated_seconds))
        )

        rows.append(
            "<tr>"
            f"<td class='slt-right slt-mono'>{trial.num_processes}</td>"
            f"<td class='slt-right slt-mono'>{trial.num_threads}</td>"
            f"<td class='slt-right'>{estimate}</td>"
            f"<td><span class='slt-status {outcome_class}'>{_html_escape(outcome)}</span></td>"
            f"<td class='slt-muted slt-cell-wrap'>{_html_escape(trial.skip_reason or '')}</td>"
            "</tr>"
        )

    if snapshot.current_processes is not None and snapshot.current_threads is not None:
        rows.append(
            "<tr>"
            f"<td class='slt-right slt-mono'>{snapshot.current_processes}</td>"
            f"<td class='slt-right slt-mono'>{snapshot.current_threads}</td>"
            "<td class='slt-right'>—</td>"
            "<td><span class='slt-status running'>running</span></td>"
            "<td class='slt-muted slt-cell-wrap'>MPI benchmark in progress</td>"
            "</tr>"
        )

    if not rows:
        rows.append(
            "<tr><td colspan='5' class='slt-empty'>No trials recorded.</td></tr>"
        )

    return (
        "<section class='slt-section'>"
        "<h3 class='slt-section-title'>Trials</h3>"
        "<div class='slt-table-wrap'>"
        "<table class='slt-table'>"
        "<thead>"
        "<tr>"
        "<th class='slt-right'>Ranks</th>"
        "<th class='slt-right'>Threads</th>"
        "<th class='slt-right'>Estimate</th>"
        "<th>Outcome</th>"
        "<th>Note</th>"
        "</tr>"
        "</thead>"
        "<tbody>" + "".join(rows) + "</tbody>"
        "</table>"
        "</div>"
        "</section>"
    )


def autotune_search_snapshot_to_html(snapshot: AutotuneSearchSnapshot) -> str:
    counts = _count_trials(snapshot.to_result())
    status_class = _search_status_css(snapshot.status)
    status_label = _search_status_label(snapshot.status)
    estimate = _html_escape(_format_seconds(snapshot.best_time))
    pill_value = (
        f"{snapshot.current_processes}×{snapshot.current_threads}"
        if snapshot.current_processes is not None
        else f"{snapshot.best_processes}×{snapshot.best_threads}"
    )

    badge_parts = [
        _summary_badge_html(
            css_class="finished",
            label="trials",
            value=counts["total"],
        ),
        _summary_badge_html(
            css_class="running",
            label="improved",
            value=counts["improved"],
        ),
        _summary_badge_html(
            css_class="cancelled",
            label="skipped",
            value=counts["skipped"],
        ),
    ]
    if counts["timed_out"]:
        badge_parts.append(
            _summary_badge_html(
                css_class="failed",
                label="timed out",
                value=counts["timed_out"],
            )
        )
    badges = "".join(badge_parts)

    message_html = ""
    if snapshot.message:
        message_html = (
            "<p class='slt-muted slt-cell-wrap'>"
            f"{_html_escape(snapshot.message)}</p>"
        )

    selection = (
        "<section class='slt-section'>"
        "<h3 class='slt-section-title'>Best so far</h3>"
        "<div class='slt-table-wrap'>"
        "<table class='slt-table'>"
        "<tbody>"
        "<tr>"
        "<th class='slt-right'>MPI ranks</th>"
        f"<td class='slt-right slt-mono'>{snapshot.best_processes}</td>"
        "</tr>"
        "<tr>"
        "<th class='slt-right'>Threads / rank</th>"
        f"<td class='slt-right slt-mono'>{snapshot.best_threads}</td>"
        "</tr>"
        "<tr>"
        "<th class='slt-right'>Est. main-loop time</th>"
        f"<td class='slt-right slt-mono'>{estimate}</td>"
        "</tr>"
        "</tbody>"
        "</table>"
        "</div>"
        f"{message_html}"
        "</section>"
    )

    return (
        dashboard_css()
        + "<div class='slt-dashboard'>"
        "<div class='slt-card'>"
        "<header class='slt-header'>"
        "<div>"
        "<h2 class='slt-title'>SlothPy autotune</h2>"
        "<div class='slt-subtitle'>MPI ranks and per-rank thread search</div>"
        "</div>"
        "<div class='slt-status-pill'>"
        "<span class='slt-dot'></span>"
        f"<span class='slt-status {status_class}'>{_html_escape(status_label)}</span>"
        f" {_html_escape(pill_value)}"
        "</div>"
        "</header>"
        f"<div class='slt-badges'>{badges}</div>"
        f"{selection}"
        f"{_trials_html_from_snapshot(snapshot)}"
        "</div>"
        "</div>"
    )


def autotune_result_to_html(result: AutotuneResult) -> str:
    snapshot = AutotuneSearchSnapshot(
        num_cpu=result.num_cpu,
        nodes=result.nodes,
        trials=result.trials,
        best_processes=result.num_processes,
        best_threads=result.num_threads,
        best_time=result.estimated_seconds,
        status="done",
    )
    return autotune_search_snapshot_to_html(snapshot)


def autotune_result_to_text(
    result: AutotuneResult,
    *,
    width: int | None = None,
) -> str:
    stream = StringIO()
    console = Console(file=stream, force_terminal=True, width=width or 120)
    console.print(autotune_result_to_rich(result, width=width))
    return stream.getvalue()


__all__ = [
    "autotune_result_to_html",
    "autotune_result_to_rich",
    "autotune_result_to_text",
    "autotune_search_snapshot_to_html",
    "autotune_search_snapshot_to_rich",
]
