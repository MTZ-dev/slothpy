import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    from slothpy.compute.magnetisation import (
        create_demo_hamiltonian_group,
        demo_magnetisation,
    )
    from slothpy.core.slt_session import SltSession

    return SltSession, create_demo_hamiltonian_group, demo_magnetisation, mo


@app.cell
def _(SltSession, create_demo_hamiltonian_group):
    """
    Start a local compute session and prepare a tiny Hamiltonian source group.

    The session is also registered for automatic shutdown on interpreter exit
      (atexit), but prefer an explicit ``session.shutdown()`` when you are done.
    """
    session = SltSession.local(
        cores=8,
        max_running_jobs=4,
        install_signal_handlers=False,
    )

    demo_slt_path = "notebooks/demo_session.slt"
    hamiltonian = create_demo_hamiltonian_group(
        demo_slt_path,
        group_name="demo_hamiltonian",
        overwrite=True,
    )

    session, hamiltonian, demo_slt_path
    return hamiltonian, session


@app.cell
def _():
    from slothpy.core.slt import open_slt_file
    slt_file = open_slt_file(path="notebooks/demo_session.slt")
    return


@app.cell
def _(demo_magnetisation, hamiltonian, session):
    """
    Submit several slow placeholder magnetisation jobs to exercise MPI progress.
    """
    jobs = []

    jobs.append(
        session.submit(
            demo_magnetisation(
                hamiltonian,
                n_fields=2,
                n_orientations=3,
                num_processes=2,
                num_threads=2,
                steps_per_task=12,
                sleep_seconds=0.12,
            ),
            job_id="mag-fast",
        )
    )

    jobs.append(
        session.submit(
            demo_magnetisation(
                hamiltonian,
                n_fields=3,
                n_orientations=4,
                num_processes=2,
                num_threads=2,
                steps_per_task=16,
                sleep_seconds=0.18,
            ),
            job_id="mag-medium",
        )
    )

    jobs.append(
        session.submit(
            demo_magnetisation(
                hamiltonian,
                n_fields=2,
                n_orientations=5,
                num_processes=1,
                num_threads=4,
                steps_per_task=20,
                sleep_seconds=0.25,
            ),
            job_id="mag-slow",
        )
    )

    {job.job_id: job.name for job in jobs}
    return (jobs,)


@app.cell
def _(mo):
    """
    Live dashboard: re-run this cell (or use refresh) while jobs are running.
    """
    refresh = mo.ui.refresh(default_interval="500ms", label="Refresh dashboard")
    refresh
    return (refresh,)


@app.cell
def _(mo, refresh, session):
    mo.vstack(
        [
            refresh,
            mo.Html(session.dashboard_html()),
        ]
    )
    return


@app.cell
def _(session):
    """
    Rich terminal view (use in marimo console or scripts; cells above use HTML).
    """
    session.to_rich()
    return


@app.cell
def _(jobs, mo):
    """
    Fetch finished results after the jobs complete.
    """
    status = {job.job_id: job.status.value for job in jobs}
    mo.md(f"**Job status:** `{status}`")
    return


@app.cell
def _(session):
    session.shutdown(wait=True)
    return


@app.cell
def _(jobs, mo):
    finished = []
    for job in jobs:
        if not job.done():
            finished.append(f"{job.job_id}: still running")
            continue
        try:
            view = job.result(timeout=0)
            value = float(view.magnetisation.values.mean())
            finished.append(f"{job.job_id}: mean magnetisation = {value:.6f}")
        except Exception as exc:
            finished.append(f"{job.job_id}: {type(exc).__name__}: {exc}")

    mo.md("\n".join(f"- {line}" for line in finished))
    return


@app.cell
def _(mo):
    mo.md("""
    ## Autotune quick checks

    The cells below exercise both autotune entry points:
    - explicit `computation.autotune(...)`
    - implicit `computation.run(autotune=True, ...)`

    After each run, the **HTML dashboard** cells show the autotune summary
    (selected ranks/threads, trial table). Re-run those cells if you change
    tuning parameters. Use `tune_result.to_rich()` in the terminal for the Rich view.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Autotune live display

    Pass ``display`` as a keyword to ``computation.autotune(...)`` or
    ``computation.run(autotune=True, ...)`` (Pydantic-validated):

    - ``print`` — plain trial lines (default when ``verbose=True``)
    - ``rich`` / ``html`` — live dashboard in the **cell output** (marimo/Jupyter) or Rich panel in a terminal
    - ``none`` — silent

    End the cell with ``tune_result.notebook_output(...)`` so the dashboard stays visible
    (a trailing ``mo.md`` alone replaces the live view).
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Autotune and `SltSession`

    ``session.submit(computation)`` does **not** autotune automatically. Tune first, then submit::

        comp = demo_magnetisation(...)
        comp.autotune(session=session, num_cpu=4, display="none")
        job = session.submit(comp)

    The session dashboard (refresh cell) shows scheduled MPI×threads and an
    ``autotuned …`` note when ``autotune_result`` is set on the computation.
    It does not re-run autotune benchmarks while jobs execute.
    """)
    return


@app.cell
def _(demo_magnetisation, hamiltonian, session):
    """
    Explicit autotune() with live dashboard in this cell's output (display='html').
    """
    autotune_demo = demo_magnetisation(
        hamiltonian,
        n_fields=2,
        n_orientations=2,
        n_temperatures=1,
        num_processes=1,
        num_threads=1,
        steps_per_task=6,
        sleep_seconds=0.03,
    )

    before = (
        autotune_demo.resources.num_processes,
        autotune_demo.resources.num_threads,
    )

    tune_result = autotune_demo.autotune(
        session=session,
        num_cpu=4,
        steps_per_task=6,
        sleep_seconds=0.003,
        measurement_progress=2,
        worse_stop_count=1,
        min_tasks_per_process=1,
        timeout_seconds=45.0,
        display="html",
        verbose=False,
    )

    after = (
        autotune_demo.resources.num_processes,
        autotune_demo.resources.num_threads,
    )
    tune_result.notebook_output(
        title_md="### Explicit `autotune(display='html')`",
        summary_md="\n".join(
            [
                f"- Before: processes={before[0]}, threads={before[1]}",
                (
                    f"- Selected: processes={tune_result.num_processes}, "
                    f"threads={tune_result.num_threads}, "
                    f"estimated={tune_result.estimated_seconds:.3f}s"
                ),
                f"- Applied to computation.resources: processes={after[0]}, threads={after[1]}",
            ]
        ),
    )
    return autotune_demo, tune_result


@app.cell
def _(autotune_demo, session):
    """
    Same search with live HTML dashboard (updates in this cell while autotune runs).
    """
    html_tune_result = autotune_demo.autotune(
        session=session,
        force=True,
        num_cpu=4,
        steps_per_task=6,
        sleep_seconds=0.003,
        measurement_progress=2,
        worse_stop_count=1,
        min_tasks_per_process=1,
        timeout_seconds=45.0,
        display="html",
        verbose=False,
    )
    html_tune_result.notebook_output(
        title_md="### Live `autotune(display='html')`",
        summary_md=(
            f"Finished: {html_tune_result.num_processes}×"
            f"{html_tune_result.num_threads} "
            f"({html_tune_result.estimated_seconds:.3f}s est.)"
        ),
    )
    return (html_tune_result,)


@app.cell
def _(html_tune_result, mo):
    mo.vstack(
        [
            mo.md("**Final HTML snapshot** (`display='html'` path)"),
            mo.Html(html_tune_result.dashboard_html()),
        ]
    )
    return


@app.cell
def _(mo, tune_result):
    """
    HTML autotune summary after explicit `computation.autotune(...)`.
    """
    mo.vstack(
        [
            mo.md("**Autotune dashboard** (`autotune()` path)"),
            mo.Html(tune_result.dashboard_html()),
        ]
    )
    return


@app.cell
def _(tune_result):
    """
    Rich terminal autotune summary (optional; same data as the HTML cell above).
    """
    tune_result.to_rich()
    return


@app.cell
def _(mo):
    mo.md("""
    **`run(autotune=True)`** runs short MPI benchmark trials, then the full
    magnetisation job. Earlier cells may still have session jobs on the same
    CPUs — wait for them to finish (or restart the kernel) before running this cell.
    """)
    return


@app.cell
def _(demo_magnetisation, hamiltonian, session):
    """
    Test run(..., autotune=True) path and inspect autotune_result afterwards.
    """
    autotune_run_demo = demo_magnetisation(
        hamiltonian,
        n_fields=2,
        n_orientations=2,
        n_temperatures=1,
        num_processes=1,
        num_threads=1,
        steps_per_task=6,
        sleep_seconds=0.03,
    )

    result = autotune_run_demo.run(
        autotune=True,
        session=session,
        num_cpu=4,
        min_tasks_per_process=1,
        measurement_progress=2,
        worse_stop_count=1,
        timeout_seconds=45.0,
        display="rich",
        verbose=False,
    )
    tuned = autotune_run_demo.autotune_result

    tuned.notebook_output(
        title_md="### `run(autotune=True)`",
        summary_md="\n".join(
            [
                (
                    f"- Tuned to processes={tuned.num_processes}, "
                    f"threads={tuned.num_threads}, "
                    f"estimated={tuned.estimated_seconds:.3f}s"
                ),
                f"- Mean magnetisation: {float(result.magnetisation.values.mean()):.6f}",
            ]
        ),
    )
    return (tuned,)


@app.cell
def _(mo, tuned):
    """
    HTML autotune summary after `run(autotune=True, ...)`.
    """
    mo.vstack(
        [
            mo.md("**Autotune dashboard** (`run(autotune=True)` path)"),
            mo.Html(tuned.dashboard_html()),
        ]
    )
    return


@app.cell
def _(tuned):
    tuned.to_rich()
    return


@app.cell
def _(session):
    """
    When you are done experimenting, run in a one-off cell::

        session.shutdown(wait=True)

    Stopping an autotune cell (marimo interrupt) cancels the benchmark **and**
    hard-cancels session MPI jobs. Active sessions are also cleaned up on exit.
    """
    session
    return


if __name__ == "__main__":
    app.run()
