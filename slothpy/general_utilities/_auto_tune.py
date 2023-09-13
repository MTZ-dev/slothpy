from os import cpu_count
from time import perf_counter
from math import ceil
from typing import Tuple
from numpy import (
    float64,
    array,
    linspace,
)
from slothpy.magnetism.magnetisation import _mth
from slothpy.general_utilities._constants import YELLOW, BLUE, PURPLE, RESET


def _auto_tune(
    filename, group, num_to_parallelize: int, matrix_size: int
) -> Tuple[int, int]:
    temperatures = linspace(1, 600, 600, dtype=float64)
    grid = array(
        [[1, 1, 1, 1]],
        dtype=float64,
    )

    num_cpu = cpu_count()
    final_num_of_processes = num_cpu
    final_num_of_threads = 1
    best_time = float("inf")

    threads_to_check = []

    for i in range(1, num_cpu + 1):
        if num_cpu % i == 0:
            threads_to_check.append(i)

    for num_threads in threads_to_check:
        num_process = num_cpu // num_threads

        fields = linspace(1, 10, num_process, dtype=float64)

        start_time = perf_counter()
        _mth(
            filename,
            group,
            fields,
            grid,
            temperatures,
            matrix_size,
            num_cpu,
            num_threads,
        )
        end_time = perf_counter()
        current_time = (end_time - start_time) * ceil(
            num_to_parallelize / num_process
        )
        print(
            "Processes:"
            f" {num_process}, threads: {num_threads}."
            f" Benchmarked time: {current_time}."
        )

        if current_time < best_time:
            best_time = current_time
            final_num_of_processes = num_process
            final_num_of_threads = num_threads

    print(
        "Job will run using"
        + YELLOW
        + f" {num_cpu}"
        + RESET
        + " logical"
        + YELLOW
        + " Processors"
        + RESET
        + " with"
        + BLUE
        + f" {final_num_of_processes}"
        + RESET
        + " parallel"
        + BLUE
        + " Processes"
        + RESET
        + " each utilizing"
        + PURPLE
        + f" {final_num_of_threads} threads "
        + RESET
        + "."
    )
    return num_cpu, final_num_of_threads
