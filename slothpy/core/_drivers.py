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

from __future__ import annotations
from abc import ABC, abstractmethod
from contextlib import contextmanager, ExitStack
from multiprocessing.synchronize import Event
from multiprocessing.managers import SharedMemoryManager
from multiprocessing import Process
from multiprocessing import Event as terminate_event
from os.path import join
from time import perf_counter_ns, sleep
from datetime import datetime

from numpy import array, zeros, any, all, median, int64

from slothpy.core._config import settings
from slothpy.core._slothpy_exceptions import slothpy_exc, SltCompError, SltSaveError, SltPlotError, SltReadError
from slothpy._general_utilities._system import SltProcessPool, _get_number_of_processes_threads, _to_shared_memory, _from_shared_memory, _distribute_chunks, _from_shared_memory_to_array
from slothpy._general_utilities._constants import RED, GREEN, BLUE, YELLOW, PURPLE, RESET
from slothpy._general_utilities._io import _save_data_to_slt
from slothpy._gui._monitor_gui import _run_monitor_gui

def ensure_ready(func):
    def wrapper(self, *args, **kwargs):
        if not self._ready:
            self.run()
        return func(self, *args, **kwargs)
    
    return wrapper

class SingleProcessed(ABC):

    __slots__ = ["_slt_group", "_hdf5", "_group_name", "_result", "_ready", "_slt_save", "_df", "_method_name", "_metadata_dict", "_data_dict", "_is_from_file"]

    @abstractmethod
    def __init__(self, slt_group, slt_save: str = None) -> None:
        super().__init__()
        self._slt_group = slt_group
        self._hdf5 = slt_group._hdf5
        self._group_name = slt_group._group_name
        self._result = None
        self._ready = False
        self._slt_save = slt_save
        self._df = None
        self._method_name = None
        self._data_dict = None
        self._metadata_dict = None
        self._is_from_file = False

    def __repr__(self) -> str:
        return f"<{RED}{self.__class__.__name__}{RESET} object from {BLUE}Group{RESET} '{self._group_name}' {GREEN}File{RESET} '{self._hdf5}'.>"
    
    @classmethod
    def _from_file(cls, slt_group) -> SingleProcessed:
        instance = cls.__new__(cls)
        instance._slt_group = slt_group
        instance._hdf5 = slt_group._hdf5
        instance._group_name = slt_group._group_name
        instance._result = None
        instance._slt_save = None
        instance._df = None
        instance._load_from_file()
        instance._ready = True
        instance._is_from_file = True

        return instance
    
    @abstractmethod
    def _executor():
        pass
    
    @slothpy_exc("SltCompError")
    def run(self):
        if not self._ready:
            self._result = self._executor()
            self._ready = True
        if self._slt_save is not None:
            self.save()
    
    @ensure_ready
    def eval(self):
        return self._result

    @abstractmethod
    def _save(self):
        pass
    
    @slothpy_exc("SltSaveError")
    def save(self, slt_save = None):
        self._save()
        _save_data_to_slt(self._hdf5, self._slt_save if slt_save is None else slt_save, self._data_dict, self._metadata_dict)

    @abstractmethod
    def _load_from_file(self):
        pass
    
    @abstractmethod
    def _plot(self):
        pass

    @slothpy_exc("SltPlotError")
    def plot(self, *args, **kwargs):
        self._plot(*args, **kwargs)
    
    @ensure_ready
    def to_numpy_array(self):
        return self._result
    
    @abstractmethod
    def _to_data_frame(self):
        pass
    
    @slothpy_exc("SltReadError")
    def to_data_frame(self):
        return self._to_data_frame()
    
    @slothpy_exc("SltSaveError")
    def to_csv(self, file_path=".", file_name="states_energies_cm_1.csv", separator=","):
        if self._df is None:
            self.to_data_frame()
        self._df.to_csv(join(file_path, file_name), sep=separator)
    
    @slothpy_exc("SltSaveError")
    def data_frame_to_slt_file(self, group_name):
        if self._df is None:
            self._to_data_frame()
        self._df.to_hdf(self._hdf5)
       

class MulitProcessed(SingleProcessed):

    __slots__ = SingleProcessed.__slots__ + ["_number_to_parallelize", "_number_cpu", "_number_processes", "_number_threads", "_executor_proxy", "_process_pool", "_autotune", "_smm", "_sm", "_sm_arrays_info", "_sm_progress_array_info",  "_sm_result_info", "_terminate_event", "_returns", "_args_arrays", "_args", "_result_shape"]

    @abstractmethod
    def __init__(self, slt_group, number_to_parallelize: int, number_cpu: int, number_threads: int, autotune: bool, smm: SharedMemoryManager = None, terminate_event: Event = None, slt_save: str = None) -> None:
        super().__init__(slt_group, slt_save)
        self._executor_proxy = None
        self._number_to_parallelize = number_to_parallelize
        self._number_cpu = number_cpu
        self._number_processes, self._number_threads = _get_number_of_processes_threads(number_cpu, number_threads, number_to_parallelize)
        self._autotune = autotune
        self._smm = smm
        self._sm = []
        self._sm_arrays_info = []
        self._sm_progress_array_info = None
        self._sm_result_info = None
        self._terminate_event = terminate_event
        self._returns = False
        self._args_arrays = []
        self._args = ()
        self._result_shape = ()

    @contextmanager
    def _ensure_shared_memory_manager(self):
        if self._smm is None:
            with SharedMemoryManager() as smm:
                self._smm = smm
                yield
                self._smm = None
        else:
            yield

    def _create_shared_memory(self):
        for array in self._args_arrays:
            self._sm_arrays_info.append(_to_shared_memory(self._smm, array))
        self._args_arrays = []
        self._sm_progress_array_info = _to_shared_memory(self._smm, zeros((self._number_processes,), dtype=int64, order="C"))
        if not self._returns:
            self._sm_result_info = _to_shared_memory(self._smm, self._result)
            self._result = None
    
    def _retrieve_arrays_and_results_from_shared_memory(self):
        self._args_arrays = []
        for sm_array_info in self._sm_arrays_info:
            self._args_arrays.append(_from_shared_memory_to_array(sm_array_info))
        self._sm_arrays_info = []
        self._sm_progress_array_info = None
        self._result = _from_shared_memory_to_array(self._sm_result_info)
        self._sm_result_info = None

    @abstractmethod
    def _load_args_arrays():
        pass

    def _create_jobs(self):
        sm_arrays_info_list = self._sm_arrays_info
        sm_arrays_info_list.append(self._sm_progress_array_info)
        if not self._returns:
            sm_arrays_info_list.append(self._sm_result_info)
        return [(sm_arrays_info_list, self._args, process_index, chunk.start, chunk.end, self._number_threads, self._returns) for process_index, chunk in enumerate(_distribute_chunks(self._number_to_parallelize, self._number_processes))]
            
    @abstractmethod
    def _gather_results(self, results):
        pass

    def _executor(self):
        self._process_pool = SltProcessPool(self._executor_proxy, self._create_jobs(), self._returns, self._terminate_event)
        result_queue = self._process_pool.start_and_collect()
        self._process_pool = None
        return result_queue

    @slothpy_exc("SltCompError")
    def autotune(self, _from_run: bool = False):
        if self._is_from_file:
            print(f"The {self.__class__.__name__} object was loaded from the .slt file. There is nothing to autotune.")
            return
        final_number_of_processes = self._number_cpu
        final_number_of_threads = 1
        best_time = float("inf")
        old_processes = 0
        worse_counter = 0
        current_terminate_event = self._terminate_event
        if self._ready:
            result_tmp = self._result
        with ExitStack() as stack:
            stack.enter_context(self._ensure_shared_memory_manager())
            self._load_args_arrays()
            self._create_shared_memory()
            for number_threads in range(min(64, self._number_cpu), 0, -1):
                number_processes = self._number_cpu // number_threads
                if number_processes >= self._number_to_parallelize:
                    number_processes = self._number_to_parallelize
                    number_threads = self._number_cpu // number_processes
                if number_processes != old_processes:
                    old_processes = number_processes
                    chunk_size = self._number_to_parallelize // number_processes
                    remainder = self._number_to_parallelize % number_processes
                    max_tasks_per_process = array([(chunk_size + (1 if i < remainder else 0)) for i in range(number_processes)])
                    if any(max_tasks_per_process < 5):
                        print(f"The job for {number_processes} {BLUE}Processes{RESET} and {number_threads} {PURPLE}Threads{RESET} is already too small to be autotuned! Quitting here.")
                        break
                    self._number_processes = number_processes
                    self._number_threads = number_threads
                    progress_array = zeros((number_processes,), dtype=int64, order="C")
                    self._sm_progress_array_info = _to_shared_memory(self._smm, progress_array)
                    self._terminate_event = terminate_event()
                    self._process_pool = SltProcessPool(self._executor_proxy, self._create_jobs(), self._returns, self._terminate_event)
                    benchmark_process = Process(target=self._process_pool.start_and_collect)
                    sm_progress, progress_array = _from_shared_memory(self._sm_progress_array_info)
                    benchmark_process.start()
                    while any(progress_array <= 1):
                        sleep(0.001)
                    start_time = perf_counter_ns()
                    start_progress = progress_array.copy()
                    final_progress = start_progress
                    stop_time = start_time
                    while any(progress_array - start_progress <= 4) and all(progress_array < max_tasks_per_process):
                        stop_time = perf_counter_ns()
                        final_progress = progress_array.copy()
                        sleep(0.01)
                    self._terminate_event.set()
                    overall_time = stop_time - start_time
                    progress = final_progress - start_progress
                    if any(progress <= 1) or overall_time == 0:
                        print(f"Jobs iterations for {number_processes} {BLUE}Processes{RESET} and {number_threads} {PURPLE}Threads{RESET} are too fast to be reliably autotuned! Quitting here.")
                        benchmark_process.join()
                        benchmark_process.close()
                        sm_progress.close()
                        sm_progress.unlink()
                        self._sm_arrays_info = self._sm_arrays_info[:-2]
                        break
                    current_estimated_time = overall_time * (max_tasks_per_process/(progress))
                    current_estimated_time = median(current_estimated_time[:remainder] if remainder != 0 else current_estimated_time)
                    benchmark_process.join()
                    benchmark_process.close()
                    info = f"{BLUE}Processes{RESET}: {number_processes}, {PURPLE}Threads{RESET}: {number_threads}. Estimated execution time of the main loop: "
                    if current_estimated_time < best_time:
                        best_time = current_estimated_time
                        final_number_of_processes = number_processes
                        final_number_of_threads = number_threads
                        info += f"{GREEN}{current_estimated_time/1e9:.2f}{RESET} s."
                        worse_counter = 0
                    else:
                        info += f"{RED}{current_estimated_time/1e9:.2f}{RESET} s."
                        worse_counter += 1
                    if worse_counter > 3:
                        break
                    info += f" The best time: {GREEN}{best_time/1e9:.2f}{RESET} s."
                    print(info)
                    sm_progress.close()
                    sm_progress.unlink()
                    self._sm_arrays_info = self._sm_arrays_info[:-2]
            time_info = f" (starting from now - [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}])" if _from_run else ''
            print(f"Job will run using{YELLOW} {final_number_of_processes * final_number_of_threads}{RESET} logical{YELLOW} CPU(s){RESET} with{BLUE} {final_number_of_processes}{RESET} parallel{BLUE} Processe(s){RESET} each utilizing{PURPLE} {final_number_of_threads} Thread(s){RESET}.\nThe calculation time{time_info} is estimated to be at least: {GREEN}{best_time/1e9} s{RESET}.")
            self._number_processes, self._number_threads = final_number_of_processes, final_number_of_threads
            if self._ready:
                self._result = result_tmp
            if _from_run:
                self._retrieve_arrays_and_results_from_shared_memory()
            else:
                self._sm_arrays_info = []
                self._sm_progress_array_info = None
                self._sm_result_info = None
            self._process_pool = None
            self._terminate_event = current_terminate_event
            self._autotune = False
    
    @slothpy_exc("SltCompError")
    def run(self):
        if not self._ready:
            if self._autotune:
                self.autotune(True)
            else:
                self._load_args_arrays()
            with ExitStack() as stack:
                stack.enter_context(self._ensure_shared_memory_manager())
                self._create_shared_memory()
                if settings.monitor:
                    monitor = Process(target=_run_monitor_gui, args=(self._sm_progress_array_info, self._number_to_parallelize, self._number_processes, "zeeman_splitting"))
                    monitor.start()
                results = self._executor()
                if settings.monitor and monitor is not None:
                    monitor.join()
                    monitor.close()
                if self._returns:
                    self._result = self._gather_results(results)
                else:
                    self._result = _from_shared_memory_to_array(self._sm_result_info, reshape=(self._result_shape))
                self._ready = True
        if self._slt_save is not None:
            self.save()
        self._sm_arrays_info = []
        self._process_pool = None
    
    @slothpy_exc("SltCompError")
    def clear(self):
        if self._is_from_file:
            print(f"The {self.__class__.__name__} object was loaded from the .slt file. It cannot be cleared.")
            return
        self._result = None
        self._ready = False


    


    