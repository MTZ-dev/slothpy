def get_num_of_processes(num_cpu, num_threads):

    if (not isinstance(num_cpu, int)) or (not isinstance(num_threads, int)) or (num_cpu < 0) or (num_threads < 0):
        raise ValueError(f'Number of CPUs and Threads have to be positive integers!')

    # Check CPUs number considering the desired number of threads and assign number of processes
    if num_cpu < num_threads:
        raise ValueError(f"Insufficient number of CPU cores assigned. Desired threads: {num_threads}, Actual available processors: {num_cpu}")
    else:
        num_process = num_cpu//num_threads
    
    return num_process