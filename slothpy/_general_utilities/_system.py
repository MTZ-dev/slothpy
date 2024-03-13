# SlothPy
# Copyright (C) 2023 Mikolaj Tadeusz Zychowicz (MTZ)

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import sys
from os import cpu_count
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from numpy import ndarray, dtype
from numpy import array as np_array


def _get_num_of_processes(num_cpu, num_threads, num_to_parallelize):
    # Check available CPU count
    total_num_of_cpu = int(cpu_count())

    if num_cpu > total_num_of_cpu:
        raise ValueError(
            f"Insufficient number of logical cores ({total_num_of_cpu}) was"
            f" detected on the machine, to accomodate {num_cpu} desired CPUs."
        )

    # Check CPUs number considering the desired number of threads and assign
    # number of processes
    if num_cpu < num_threads:
        raise ValueError(
            "Insufficient number of CPU cores assigned. Desired threads:"
            f" {num_threads}, Actual available processors: {num_cpu}"
        )

    # Check if there is more processes than the things to parallelize over
    num_process = num_cpu // num_threads
    if num_process >= num_to_parallelize:
        num_process = num_to_parallelize
    
    num_threads = num_cpu // num_process

    return num_process, num_threads


def _to_shared_memory(smm: SharedMemoryManager, array: ndarray):
    shm = smm.SharedMemory(size=array.nbytes)
    shared_array = ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
    shared_array[:] = array
    del array
    return shm.name, shared_array.shape, shared_array.dtype


def _from_shared_memory(array_info: tuple):
    shared_memory = SharedMemory(array_info[0])
    array = np_array(ndarray(array_info[1], array_info[2], shared_memory.buf), copy=True, order="C")
    return array


def _chunk_from_shared_memory(array_info: tuple, chunk: tuple):
    offset = dtype(array_info[2]).itemsize * chunk[0]
    chunk_length = chunk[1] - chunk[0]
    smm = SharedMemory(array_info[0])
    array_chunk = np_array(ndarray((chunk_length,), array_info[2], smm.buf, offset), copy=True, order="C")
    return array_chunk


def _distribute_chunks(data_len, num_process):
    chunk_size = data_len // num_process
    remainder = data_len % num_process

    for i in range(num_process):
        start = i * chunk_size + min(i, remainder)
        end = start + chunk_size + (1 if i < remainder else 0)
        yield (start, end)


# Determine if the module is executed in a Jupyter Notebook
def _is_notebook():
    # Check if the get_ipython function is defined (typically only defined
    # in Jupyter environments).
    return "ipykernel" in sys.modules
