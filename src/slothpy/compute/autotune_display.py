"""Live autotune progress display (Rich terminal and notebook HTML)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from rich.console import Console
from rich.text import Text

from slothpy.compute.autotune import (
    AutotuneConfig,
    AutotuneSearchSnapshot,
    AutotuneTrial,
    resolve_autotune_display_mode,
)
from slothpy.compute.autotune_dashboard import (
    autotune_search_snapshot_to_html,
    autotune_search_snapshot_to_rich,
)
from slothpy.types.composite import AutotuneDisplay

LiveOutputStyle = Literal["rich", "html"]
LiveOutputBackend = Literal["marimo", "ipython", "terminal"]


def _live_output_backend() -> LiveOutputBackend:
    """Choose how to refresh live autotune output without stacking duplicate frames."""
    try:
        from marimo._runtime.context.types import runtime_context_installed

        if runtime_context_installed():
            return "marimo"
    except ImportError:
        pass

    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            return "ipython"
    except ImportError:
        pass

    return "terminal"


def _terminal_supports_inplace(console: Console) -> bool:
    return bool(console.is_terminal and getattr(console, "is_interactive", True))


class _LiveAutotunePublisher:
    """
    Update one autotune dashboard frame in place.

    marimo uses ``mo.output.replace``; Jupyter uses ``update_display`` (with an
    initial ``clear_output``); real terminals use Rich cursor controls.
    """

    def __init__(
        self,
        *,
        style: LiveOutputStyle,
        console: Console | None = None,
        width: int | None = None,
    ) -> None:
        self._style = style
        self._console = console or Console()
        self._width = width
        self._backend = _live_output_backend()
        self._display_id: str | None = None
        self._live_render: Any | None = None
        self._rich_live: Any | None = None

        if self._backend == "terminal" and _terminal_supports_inplace(self._console):
            from rich.live import LiveRender

            self._live_render = LiveRender(Text(""))

    def _html_text(self, snapshot: AutotuneSearchSnapshot) -> str:
        if self._style == "html":
            return autotune_search_snapshot_to_html(snapshot)

        from slothpy.core.slt_common import _rich_to_html

        panel = autotune_search_snapshot_to_rich(snapshot, width=self._width)
        return _rich_to_html(panel)

    def _marimo_output(self, snapshot: AutotuneSearchSnapshot) -> Any:
        import marimo as mo

        return mo.Html(self._html_text(snapshot))

    def _ipython_output(self, snapshot: AutotuneSearchSnapshot) -> Any:
        from IPython.display import HTML

        return HTML(self._html_text(snapshot))

    def update(self, snapshot: AutotuneSearchSnapshot) -> None:
        if self._backend == "marimo":
            import marimo as mo

            mo.output.replace(self._marimo_output(snapshot))
            return

        if self._backend == "ipython":
            from IPython.display import clear_output, display, update_display

            payload = self._ipython_output(snapshot)
            if self._display_id is None:
                clear_output(wait=True)
                handle = display(payload, display_id=True)
                self._display_id = handle.display_id
                return
            update_display(payload, display_id=self._display_id)
            return

        panel = autotune_search_snapshot_to_rich(snapshot, width=self._width)
        if self._live_render is not None:
            from slothpy.core.slt_dashboard import _print_live_render

            _print_live_render(self._console, self._live_render, panel)
            return

        if self._rich_live is None:
            from rich.live import Live

            self._rich_live = Live(
                panel,
                console=self._console,
                refresh_per_second=4,
                transient=False,
            )
            self._rich_live.start()
            return

        self._rich_live.update(panel)

    def close(self) -> None:
        if self._rich_live is not None:
            self._rich_live.stop()
            self._rich_live = None
        if self._backend == "terminal":
            self._console.show_cursor(True)


def autotune_notebook_output(
    result: Any,
    *,
    title_md: str | None = None,
    summary_md: str | None = None,
) -> Any:
    """
    Build a notebook cell output that preserves the final autotune dashboard.

    Parameters
    ----------
    result
        :class:`~slothpy.compute.autotune.AutotuneResult` (or any object with
        ``dashboard_html()``).
    title_md, summary_md
        Optional markdown shown above / below the dashboard.
    """
    html = result.dashboard_html()
    backend = _live_output_backend()

    if backend == "marimo":
        import marimo as mo

        parts: list[Any] = []
        if title_md:
            parts.append(mo.md(title_md))
        parts.append(mo.Html(html))
        if summary_md:
            parts.append(mo.md(summary_md))
        output = mo.vstack(parts, gap=1) if len(parts) > 1 else parts[0]
        mo.output.replace(output)
        return output

    if backend == "ipython":
        from IPython.display import HTML, Markdown, display

        parts: list[Any] = []
        if title_md:
            parts.append(Markdown(title_md))
        parts.append(HTML(html))
        if summary_md:
            parts.append(Markdown(summary_md))
        display(*parts)
        return parts[-2] if len(parts) >= 2 else parts[0]

    return result


class AutotuneDisplayDriver(ABC):
    """Callbacks invoked while :func:`~slothpy.compute.autotune.autotune_mpi_threading` runs."""

    @abstractmethod
    def on_search_start(self, snapshot: AutotuneSearchSnapshot) -> None:
        """Search is about to benchmark the first configuration."""

    @abstractmethod
    def on_trial_start(self, snapshot: AutotuneSearchSnapshot) -> None:
        """A benchmark subprocess is starting for ``snapshot.current_*``."""

    @abstractmethod
    def on_trial_finished(self, snapshot: AutotuneSearchSnapshot) -> None:
        """One configuration finished (recorded in ``snapshot.trials``)."""

    @abstractmethod
    def on_search_finished(self, snapshot: AutotuneSearchSnapshot) -> None:
        """Search completed or was aborted."""

    @abstractmethod
    def close(self) -> None:
        """Release terminal or notebook display resources."""


class NullAutotuneDisplayDriver(AutotuneDisplayDriver):
    def on_search_start(self, snapshot: AutotuneSearchSnapshot) -> None:
        return None

    def on_trial_start(self, snapshot: AutotuneSearchSnapshot) -> None:
        return None

    def on_trial_finished(self, snapshot: AutotuneSearchSnapshot) -> None:
        return None

    def on_search_finished(self, snapshot: AutotuneSearchSnapshot) -> None:
        return None

    def close(self) -> None:
        return None


class PrintAutotuneDisplayDriver(AutotuneDisplayDriver):
    """Plain-text trial lines (legacy ``verbose=True`` behaviour)."""

    def on_search_start(self, snapshot: AutotuneSearchSnapshot) -> None:
        return None

    def on_trial_start(self, snapshot: AutotuneSearchSnapshot) -> None:
        if snapshot.current_processes is None or snapshot.current_threads is None:
            return
        print(
            f"Benchmarking processes={snapshot.current_processes}, "
            f"threads={snapshot.current_threads}..."
        )

    def on_trial_finished(self, snapshot: AutotuneSearchSnapshot) -> None:
        if not snapshot.trials:
            return
        trial = snapshot.trials[-1]
        if trial.skipped:
            if trial.skip_reason:
                print(
                    f"Skipping processes={trial.num_processes}, "
                    f"threads={trial.num_threads}: {trial.skip_reason}"
                )
            return
        if trial.timed_out:
            print("Autotune timeout reached during warmup.")
            return

        marker = "best" if trial.improved else "worse"
        estimate = trial.estimated_seconds
        estimate_text = (
            f"{estimate:.2f} s ({marker})" if np.isfinite(estimate) else "unreliable"
        )
        print(
            f"processes={trial.num_processes}, threads={trial.num_threads}: "
            f"estimated main-loop time {estimate_text}"
        )

    def on_search_finished(self, snapshot: AutotuneSearchSnapshot) -> None:
        if snapshot.message:
            print(snapshot.message)
        if snapshot.status == "aborted":
            return

        print(
            f"Selected {snapshot.best_processes} MPI rank(s) x "
            f"{snapshot.best_threads} thread(s) "
            f"({snapshot.best_processes * snapshot.best_threads} logical CPUs)."
        )
        if np.isfinite(snapshot.best_time):
            print(f"Estimated main-loop time: {snapshot.best_time:.2f} s.")

    def close(self) -> None:
        return None


class _VisualAutotuneDisplayDriver(AutotuneDisplayDriver):
    """Shared live dashboard driver for Rich and HTML display modes."""

    def __init__(
        self,
        *,
        style: LiveOutputStyle,
        console: Console | None = None,
        width: int | None = None,
    ) -> None:
        self._publisher = _LiveAutotunePublisher(
            style=style,
            console=console,
            width=width,
        )

    def _refresh(self, snapshot: AutotuneSearchSnapshot) -> None:
        self._publisher.update(snapshot)

    def on_search_start(self, snapshot: AutotuneSearchSnapshot) -> None:
        self._refresh(snapshot)

    def on_trial_start(self, snapshot: AutotuneSearchSnapshot) -> None:
        self._refresh(snapshot)

    def on_trial_finished(self, snapshot: AutotuneSearchSnapshot) -> None:
        self._refresh(snapshot)

    def on_search_finished(self, snapshot: AutotuneSearchSnapshot) -> None:
        self._refresh(snapshot)

    def close(self) -> None:
        if self._publisher._backend in ("marimo", "ipython"):
            return
        self._publisher.close()


class RichLiveAutotuneDisplayDriver(_VisualAutotuneDisplayDriver):
    """Live Rich panel (terminal) or Rich-styled HTML (marimo / Jupyter)."""

    def __init__(
        self,
        *,
        console: Console | None = None,
        width: int | None = None,
    ) -> None:
        super().__init__(style="rich", console=console, width=width)


class HtmlLiveAutotuneDisplayDriver(_VisualAutotuneDisplayDriver):
    """Live SlothPy HTML dashboard (marimo / Jupyter) or Rich fallback in terminal."""

    def __init__(
        self,
        *,
        console: Console | None = None,
        width: int | None = None,
    ) -> None:
        super().__init__(style="html", console=console, width=width)


@dataclass
class _AutotuneSearchState:
    num_cpu: int
    nodes: int
    best_time: float
    best_processes: int
    best_threads: int
    trials: list[AutotuneTrial]
    status: str = "searching"
    current_processes: int | None = None
    current_threads: int | None = None
    message: str | None = None

    def snapshot(self) -> AutotuneSearchSnapshot:
        return AutotuneSearchSnapshot(
            num_cpu=self.num_cpu,
            nodes=self.nodes,
            trials=tuple(self.trials),
            best_processes=self.best_processes,
            best_threads=self.best_threads,
            best_time=self.best_time,
            status=self.status,  # type: ignore[arg-type]
            current_processes=self.current_processes,
            current_threads=self.current_threads,
            message=self.message,
        )


def create_autotune_display_driver(
    config: AutotuneConfig,
    *,
    console: Console | None = None,
    width: int | None = None,
) -> AutotuneDisplayDriver:
    mode = resolve_autotune_display_mode(config)
    if mode is AutotuneDisplay.NONE:
        return NullAutotuneDisplayDriver()
    if mode is AutotuneDisplay.PRINT:
        return PrintAutotuneDisplayDriver()
    if mode is AutotuneDisplay.RICH:
        return RichLiveAutotuneDisplayDriver(console=console, width=width)
    if mode is AutotuneDisplay.HTML:
        return HtmlLiveAutotuneDisplayDriver()
    raise ValueError(f"Unknown autotune display mode: {mode!r}")


def run_autotune_search_with_display(
    *,
    config: AutotuneConfig,
    num_cpu: int,
    nodes: int,
    n_parallel_tasks: int,
    steps_per_task: int,
    total_work: int,
    driver: AutotuneDisplayDriver | None = None,
    session: object | None = None,
) -> AutotuneSearchSnapshot:
    """
    Run the MPI/thread configuration search loop.

    This is the core of :func:`~slothpy.compute.autotune.autotune_mpi_threading`,
    extracted so display drivers can be tested without MPI.
    """
    from slothpy.compute.autotune import (
        AutotuneTrial,
        _run_benchmark_trial,
        iter_mpi_thread_configs,
        normalize_process_thread_pair,
    )

    cfg = config
    display = create_autotune_display_driver(cfg) if driver is None else driver
    owns_driver = driver is None

    best_time = float("inf")
    best_processes = max(1, min(num_cpu, n_parallel_tasks))
    best_threads = max(1, num_cpu // best_processes)
    worse_counter = 0
    state = _AutotuneSearchState(
        num_cpu=num_cpu,
        nodes=nodes,
        best_time=best_time,
        best_processes=best_processes,
        best_threads=best_threads,
        trials=[],
    )

    try:
        display.on_search_start(state.snapshot())

        for raw_processes, raw_threads in iter_mpi_thread_configs(
            num_cpu,
            n_parallel_tasks=n_parallel_tasks,
            max_threads=cfg.max_threads,
            max_processes=cfg.max_processes,
        ):
            num_processes, num_threads = normalize_process_thread_pair(
                num_cpu,
                raw_processes,
                raw_threads,
                n_parallel_tasks=n_parallel_tasks,
            )

            state.current_processes = num_processes
            state.current_threads = num_threads
            display.on_trial_start(state.snapshot())

            trial = _run_benchmark_trial(
                num_processes=num_processes,
                num_threads=num_threads,
                n_tasks=n_parallel_tasks,
                steps_per_task=steps_per_task,
                total_work_units=total_work,
                config=cfg,
            )

            state.current_processes = None
            state.current_threads = None

            if trial.skipped:
                state.trials.append(trial)
                display.on_trial_finished(state.snapshot())
                if trial.skip_reason and "tasks per MPI rank" in trial.skip_reason:
                    state.status = "aborted"
                    state.message = trial.skip_reason
                    break
                continue

            if trial.timed_out:
                state.trials.append(trial)
                display.on_trial_finished(state.snapshot())
                state.status = "aborted"
                state.message = "Autotune timeout reached during warmup."
                break

            improved = trial.estimated_seconds < state.best_time
            if improved:
                state.best_time = trial.estimated_seconds
                state.best_processes = num_processes
                state.best_threads = num_threads
                worse_counter = 0
            else:
                worse_counter += 1

            recorded = AutotuneTrial(
                num_processes=num_processes,
                num_threads=num_threads,
                estimated_seconds=trial.estimated_seconds,
                improved=improved,
                skip_reason=trial.skip_reason,
            )
            state.trials.append(recorded)
            display.on_trial_finished(state.snapshot())

            if trial.skip_reason and "too little" in trial.skip_reason.lower():
                state.status = "aborted"
                state.message = (
                    "Benchmark progress too fast to measure; stopping search."
                )
                break

            if worse_counter > cfg.worse_stop_count:
                state.message = (
                    f"Stopping after {cfg.worse_stop_count} worsening trials. "
                    f"Best estimate: {state.best_time:.2f} s."
                )
                break

        if state.status == "searching":
            state.status = "done"

        finished = state.snapshot()
        display.on_search_finished(finished)
        return finished
    except BaseException:
        from slothpy.compute.autotune_runtime import abort_compute_runtime

        abort_compute_runtime(session=session, cancel_session_jobs=True)
        raise
    finally:
        if owns_driver:
            display.close()


__all__ = [
    "AutotuneDisplayDriver",
    "HtmlLiveAutotuneDisplayDriver",
    "NullAutotuneDisplayDriver",
    "PrintAutotuneDisplayDriver",
    "RichLiveAutotuneDisplayDriver",
    "autotune_notebook_output",
    "create_autotune_display_driver",
    "run_autotune_search_with_display",
]
