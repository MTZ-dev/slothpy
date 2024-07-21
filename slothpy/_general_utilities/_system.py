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
from typing import Iterable
from multiprocessing import Process, Queue
from multiprocessing.synchronize import Event
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory

from threadpoolctl import threadpool_limits
from numba import set_num_threads
from numpy import ndarray, dtype, array

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


class SharedMemoryArrayInfo:
    def __init__(self, name, shape, dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype


def _to_shared_memory(smm: SharedMemoryManager, array: ndarray):
    shm = smm.SharedMemory(size=array.nbytes)
    shared_array = ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
    shared_array[:] = array
    del array
    return SharedMemoryArrayInfo(shm.name, shared_array.shape, shared_array.dtype)


def _from_shared_memory(sm_array_info: SharedMemoryArrayInfo):
    sm = SharedMemory(sm_array_info.name)
    return sm, ndarray(sm_array_info.shape, sm_array_info.dtype, sm.buf)


def _from_shared_memory_to_array(sm_array_info: SharedMemoryArrayInfo, reshape: tuple = None):
    sm = SharedMemory(sm_array_info.name)
    return array(ndarray(sm_array_info.shape if reshape is None else reshape, sm_array_info.dtype, sm.buf), copy=True, order="C")


class ChunkInfo:
    def __init__(self, start, end):
        __slots__ = ["start", "end"]
        self.start = start
        self.end = end


def _distribute_chunks(data_len, number_processes):
    chunk_size = data_len // number_processes
    remainder = data_len % number_processes

    for i in range(number_processes):
        start = i * chunk_size + min(i, remainder)
        end = start + chunk_size + (1 if i < remainder else 0)
        yield ChunkInfo(start, end)


def _chunk_from_shared_memory(sm: SharedMemory, sm_array_info: SharedMemoryArrayInfo, chunk: ChunkInfo):
    offset = dtype(sm_array_info.dtype).itemsize * chunk.start
    chunk_length = chunk.end - chunk.start
    return ndarray((chunk_length,), sm_array_info.dtype, sm.buf, offset)


def _load_shared_memory_arrays(sm_arrays_info_list):
    sm_list = []
    arrays_list = []
    for sm_array_info in sm_arrays_info_list:
        sm, array = _from_shared_memory(sm_array_info)
        sm_list.append(sm)
        arrays_list.append(array)
    return sm_list, arrays_list


def _worker_wrapper(worker, args, number_threads, result_queue=None):
    with threadpool_limits(limits=number_threads):
        set_num_threads(number_threads)
        result = worker(*args)
    if result_queue is not None:
        result_queue.put(result)


class SltProcessPool:
    def __init__(self, worker: callable, jobs: Iterable, number_threads: int, returns: bool, gather_results: callable, terminate_event: Event = None):
        self._worker = worker
        self._jobs = jobs
        self._number_threads = number_threads
        self._terminate_event = terminate_event
        self._processes = []
        self._returns = returns
        if returns:
            self._result_queue = Queue()
            self._gather_results = gather_results
        self._result = None

    def start_and_collect(self):
        try:
            for job_args in self._jobs:
                if self._returns:
                    process = Process(target=_worker_wrapper, args=(self._worker, job_args, self._number_threads, self._result_queue))
                else:
                    process = Process(target=_worker_wrapper, args=(self._worker, job_args, self._number_threads))
                process.start()
                self._processes.append(process)

            if self._terminate_event:
                while True:
                    if self._terminate_event.is_set():
                        for process in self._processes:
                            if process is not None:
                                process.terminate()
                        break
                    elif all(p is None or not p.is_alive() for p in self._processes):
                        break
                    sleep(0.2)

            for process in self._processes:
                process.join()
                process.close()
            
            if self._returns:
                self._result = self._gather_results(self._result_queue)

            return self._result
        
        except KeyboardInterrupt:
            print("\nSltProcessPool interrupted. Clearing and terminating...")
            for process in self._processes:
                if process is not None:
                    process.terminate()
            for process in self._processes:
                if process is not None:
                    process.join()
            for process in self._processes:
                if process is not None:
                    process.close()
            raise


def _is_notebook():
    return "ipykernel" in sys.modules


def _dummy(queue):
    pass
