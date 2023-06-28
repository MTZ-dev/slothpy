import os

def get_num_of_processes(num_cpu):

    # Check CPUs number considering the desired number of threads and assign number of processes
    if num_cpu < int(os.getenv('OMP_NUM_THREADS')):
        raise ValueError(f"Insufficient number of CPU cores assigned. Desired threads: {int(os.getenv('OMP_NUM_THREADS'))}, Actual processors: {num_cpu}")
    else:
        num_process = num_cpu//int(os.getenv('OMP_NUM_THREADS'))
    
    return num_process