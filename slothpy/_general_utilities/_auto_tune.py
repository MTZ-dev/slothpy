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

import inspect 
from functools import wraps
from time import perf_counter_ns, sleep
from multiprocessing import Process, Event
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
from numpy import array, zeros, any, all, median, int64
from slothpy._general_utilities._constants import YELLOW, BLUE, PURPLE, GREEN, RED, RESET
from slothpy._general_utilities._system import _from_shared_memory, _to_shared_memory


def _autotune(number_tasks: int, number_cpu: int):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            signature = inspect.signature(func)
            bound_args = signature.bind_partial(*args, **kwargs)
            bound_args.apply_defaults()

            final_number_of_processes = number_cpu
            final_number_of_threads = 1
            best_time = float("inf")

            old_processes = 0
            worse_counter = 0

            for number_threads in range(min(64, number_cpu), 0, -1):
                number_processes = number_cpu // number_threads
                if number_processes >= number_tasks:
                    number_processes = number_tasks
                    number_threads = number_cpu // number_processes
                if number_processes != old_processes:
                    old_processes = number_processes
                    with SharedMemoryManager() as smm:
                        chunk_size = number_tasks // number_processes
                        remainder = number_tasks % number_processes
                        max_tasks_per_process = array([(chunk_size + (1 if i < remainder else 0)) for i in range(number_processes)])
                        if any(max_tasks_per_process < 5):
                            print(f"The job for {number_processes} {BLUE}Processes{RESET} and {number_threads} {PURPLE}Threads{RESET} is already too small to be autotuned! Quitting here.")
                            break
                        terminate_event = Event()
                        progress_array = zeros((number_processes,), dtype=int64, order="C")
                        progress_array_info = _to_shared_memory(smm, progress_array)
                        bound_args.arguments["_progress_array_info"] = progress_array_info
                        bound_args.arguments["number_processes"] = number_processes
                        bound_args.arguments["number_threads"] = number_threads
                        bound_args.arguments["_terminate_event"] = terminate_event
                        benchmark_process = Process(target=func, args=bound_args.args, kwargs=bound_args.kwargs)
                        sm_progress, progress_array = _from_shared_memory(progress_array_info)
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
                        terminate_event.set()
                        overall_time = stop_time - start_time
                        progress = final_progress - start_progress
                        if any(progress <= 1) or overall_time == 0:
                            print(f"Jobs iterations for {number_processes} {BLUE}Processes{RESET} and {number_threads} {PURPLE}Threads{RESET} are too fast to be reliably autotuned! Quitting here.")
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

            print(f"Job will run using{YELLOW} {final_number_of_processes * final_number_of_threads}{RESET} logical{YELLOW} CPU(s){RESET} with{BLUE} {final_number_of_processes}{RESET} parallel{BLUE} Processe(s){RESET} each utilizing{PURPLE} {final_number_of_threads} Thread(s){RESET}.\nThe calculation time (starting from now) is estimated to be at most: {GREEN}{best_time/1e9} s{RESET}.")

            return final_number_of_processes, final_number_of_threads
        
        return wrapper
    return decorator

def _auto_tune(): #compatibility
    pass
