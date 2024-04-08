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
from time import sleep
from os import cpu_count
from multiprocessing import Process
from multiprocessing.synchronize import Event
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from numpy import ndarray, dtype, array
from numpy import array as np_array

def _get_num_of_processes():
    pass #only for import compatibility


def _get_number_of_processes_threads(number_cpu, number_threads, number_to_parallelize):
    total_number_of_cpu = int(cpu_count())
    if number_cpu > total_number_of_cpu:
        raise ValueError(
            f"Insufficient number of logical cores ({total_number_of_cpu}) was"
            f" detected on the machine, to accomodate {number_cpu} desired CPUs."
        )
    if number_cpu < number_threads:
        raise ValueError(
            "Insufficient number of CPU cores assigned. Desired threads:"
            f" {number_threads}, Actual available processors: {number_cpu}"
        )
    number_process = number_cpu // number_threads
    if number_process >= number_to_parallelize:
        number_process = number_to_parallelize
    
    number_threads = number_cpu // number_process

    return number_process, number_threads


def _to_shared_memory(smm: SharedMemoryManager, array: ndarray):
    shm = smm.SharedMemory(size=array.nbytes)
    shared_array = ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
    shared_array[:] = array
    del array
    return shm.name, shared_array.shape, shared_array.dtype


def _from_shared_memory(sm: SharedMemory, sm_array_info: tuple):
    return ndarray(sm_array_info[1], sm_array_info[2], sm.buf)


def _from_shared_memory_to_array(sm_array_info: tuple):
    sm_array = SharedMemory(sm_array_info[0])
    return array(ndarray(sm_array_info[1], sm_array_info[2], sm_array.buf), copy=True, order="C")


def _chunk_from_shared_memory(sm: SharedMemory, sm_array_info: tuple, chunk: tuple):
    offset = dtype(sm_array_info[2]).itemsize * chunk[0]
    chunk_length = chunk[1] - chunk[0]
    return ndarray((chunk_length,), sm_array_info[2], sm.buf, offset)


def _distribute_chunks(data_len, number_processes):
    chunk_size = data_len // number_processes
    remainder = data_len % number_processes

    for i in range(number_processes):
        start = i * chunk_size + min(i, remainder)
        end = start + chunk_size + (1 if i < remainder else 0)
        yield (start, end)


def _slt_processes_pool(worker: callable, jobs, terminate_event: Event):
    processes = []

    for job_args in jobs:
        process = Process(target=worker, args=(job_args,))
        process.start()
        processes.append(process)

    if terminate_event is not None:
        while True:
            if terminate_event.is_set():
                for process in processes:
                    process.terminate()
                for process in processes:
                    process.join()
                return
            else:
                all_finished = all(not process.is_alive() for process in processes)
                if all_finished:
                    break
            sleep(0.4)
    
    for process in processes:
        process.join()
        process.close()


def _is_notebook():
    return "ipykernel" in sys.modules
