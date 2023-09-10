import sys
import IPython

# os.cpu_count()!!! make it default with thread = 1 across whole slothpy


def get_num_of_processes(num_cpu, num_threads):
    if (
        (not isinstance(num_cpu, int))
        or (not isinstance(num_threads, int))
        or (num_cpu < 0)
        or (num_threads < 0)
    ):
        raise ValueError(
            "Number of CPUs and Threads have to be positive integers!"
        )

    # Check CPUs number considering the desired number of threads and assign
    # number of processes
    if num_cpu < num_threads:
        raise ValueError(
            "Insufficient number of CPU cores assigned. Desired threads:"
            f" {num_threads}, Actual available processors: {num_cpu}"
        )

    num_process = num_cpu // num_threads

    return num_process


# Determine if the module is executed in a Jupyter Notebook
def _is_notebook():
    # Check if the get_ipython function is defined (typically only defined
    # in Jupyter environments).
    return "ipykernel" in sys.modules


# Set a custom traceback limit for printing the SltErrors
# for system and Jupyter Notebook. Edit it for debugging.
def set_plain_error_reporting_mode():
    if _is_notebook():
        IPython.get_ipython().run_line_magic("xmode", "Plain")
    else:
        sys.tracebacklimit = 0


def set_default_error_reporting_mode():
    if _is_notebook():
        IPython.get_ipython().run_line_magic("xmode", "Context")
    else:
        sys.tracebacklimit = None
