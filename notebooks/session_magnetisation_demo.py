import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    from slothpy.compute.autotune import AutotuneConfig
    from slothpy.compute.magnetisation import (
        create_demo_hamiltonian_group,
        demo_magnetisation,
    )
    from slothpy.core.slt_session import SltSession

    return (
        AutotuneConfig,
        SltSession,
        create_demo_hamiltonian_group,
        demo_magnetisation,
        mo,
    )


@app.cell
def _(SltSession, create_demo_hamiltonian_group):
    """
    Start a local compute session and prepare a tiny Hamiltonian source group.
    """
    session = SltSession.local(cores=8, max_running_jobs=4)

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
    """)
    return


@app.cell
def _(AutotuneConfig, demo_magnetisation, hamiltonian, mo, session):
    """
    Build a small computation for explicit autotune() testing.
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
        config=AutotuneConfig(
            num_cpu=4,
            steps_per_task=6,
            sleep_seconds=0.003,
            measurement_progress=2,
            worse_stop_count=1,
            timeout_seconds=45.0,
            verbose=False,
        ),
    )

    after = (
        autotune_demo.resources.num_processes,
        autotune_demo.resources.num_threads,
    )
    mo.md(
        "\n".join(
            [
                f"- Before autotune: processes={before[0]}, threads={before[1]}",
                (
                    f"- Suggested: processes={tune_result.num_processes}, "
                    f"threads={tune_result.num_threads}, "
                    f"estimated={tune_result.estimated_seconds:.3f}s"
                ),
                f"- Applied to computation resources: processes={after[0]}, threads={after[1]}",
            ]
        )
    )
    return


@app.cell
def _(demo_magnetisation, hamiltonian, mo, session):
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
        cores=4,
        measurement_progress=2,
        worse_stop_count=1,
        timeout_seconds=45.0,
        verbose=False,
    )
    tuned = autotune_run_demo.autotune_result

    mo.md(
        "\n".join(
            [
                (
                    f"- run(autotune=True) tuned to "
                    f"processes={tuned.num_processes}, threads={tuned.num_threads}"
                ),
                f"- Mean magnetisation: {float(result.magnetisation.values.mean()):.6f}",
            ]
        )
    )
    return


@app.cell
def _(session):
    # Uncomment to tear down the session when finished experimenting.
    # session.shutdown(wait=True)
    session
    return


if __name__ == "__main__":
    app.run()
