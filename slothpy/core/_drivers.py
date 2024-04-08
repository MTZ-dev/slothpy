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
from os.path import join

from slothpy.core._slothpy_exceptions import slothpy_exc
from slothpy.core._slt_file import SltGroup
from slothpy._general_utilities._system import _get_number_of_processes_threads, _slt_processes_pool

def ensure_ready(func):
    def wrapper(self, *args, **kwargs):
        if self._result is None:
            self.run()
        return func(self, *args, **kwargs)
    
    return wrapper

class SingleProcessed(ABC):

    __slots__ = ["_slt_group", "_hdf5", "_group_name", "_driver", "_result", "_slt_save", "_df"]

    @abstractmethod
    def __init__(self, slt_group: SltGroup, slt_save: str = None) -> None:
        super().__init__()
        self._slt_group = slt_group
        self._hdf5 = slt_group._hdf5
        self._group_name = slt_group._group_name
        self._driver = "single"
        self._result = None
        self._slt_save = slt_save
        self._df = None

    @abstractmethod
    def __repr__(self) -> str:
        pass
    
    @classmethod
    def _from_file(cls, slt_group: SltGroup) -> SingleProcessed:
        instance = cls.__new__(cls)
        instance._slt_group = slt_group
        instance._hdf5 = slt_group._hdf5
        instance._group_name = slt_group._group_name
        instance._driver = None
        instance._result = None
        instance._slt_save = None
        instance._df = None

        instance._load()

        return instance
    
    @abstractmethod
    def _executor():
        pass
    
    @slothpy_exc("SltCompError")
    def run(self):
        if self._result is None:
            self._result = self._executor()
        if self._slt_save is not None:
            self.save()
        return self._result
    
    @ensure_ready
    def eval(self):
        return self._result

    @abstractmethod
    def save(self, slt_save = None):
        pass

    @abstractmethod
    def _load(self):
        pass
    
    @abstractmethod
    def plot(self):
        pass
    
    @ensure_ready
    def to_numpy_array(self):
        return self._result
    
    @abstractmethod
    def to_data_frame(self):
        pass
    
    def to_csv(self, file_path=".", file_name="states_energies_cm_1.csv", separator=","):
        if self._df is None:
            self.to_data_frame()
        self._df.to_csv(join(file_path, file_name), sep=separator)
       

class MulitProcessed(SingleProcessed):

    __slots__ = SingleProcessed.__slots__ + ["_number_to_parallelize", "_number_tasks_per_process", "_number_cpu", "_number_processes", "_number_threads", "_autotune", "_smm", "_terminate_event", "_jobs"]

    @abstractmethod
    def __init__(self, slt_group: SltGroup, number_to_parallelize: int, number_tasks_per_process: int, number_cpu: int, number_threads: int, autotune: bool, smm: SharedMemoryManager = None, terminate_event: Event = None, slt_save: str = None) -> None:
        super().__init__(slt_group, slt_save)
        self._driver = "multi"
        self._number_to_parallelize = number_to_parallelize
        self._number_tasks_per_process = number_tasks_per_process
        self._number_cpu = number_cpu
        self._number_processes, self._number_threads = _get_number_of_processes_threads(number_cpu, number_threads, number_to_parallelize)
        self._autotune = autotune
        self._smm = smm
        self._terminate_event = terminate_event
        self._jobs = None

    @contextmanager
    def _ensure_shared_memory_manager(self):
        if self._smm is None:
            with SharedMemoryManager() as smm:
                self._smm = smm
                yield
                self._smm = None
        else:
            yield

    @abstractmethod
    def _create_shared_memory(self):
        pass
    
    @abstractmethod
    def _create_jobs(self):
        pass

    def _parallel_executor(self):
        _slt_processes_pool(self._executor, self._jobs, self._terminate_event)

    @slothpy_exc("SltCompError")
    def run(self):
        if self._result is None:
            with ExitStack() as stack:
                stack.enter_context(self._ensure_shared_memory_manager())
                self._create_shared_memory()
                self._create_jobs()
                self._parallel_executor()
        if self._slt_save is not None:
            self.save()
        return self._result

    


    