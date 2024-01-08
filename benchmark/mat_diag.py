import numpy as np
import time
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits


def benchmark_hermitian_diagonalization(matrix_size, num_threads):
    """
    Generates a random Hermitian matrix of given size and diagonalizes it using numpy.linalg.eigh.
    Limits BLAS and OMP threads to the specified number for testing.

    Args:
    matrix_size (int): Size of the Hermitian matrix.
    num_threads (int): Number of threads to use in BLAS and OMP.

    Returns:
    float: Execution time.
    """

    # Start the timer
    start_time = time.perf_counter_ns()

    with threadpool_limits(limits=num_threads, user_api="blas"):
        with threadpool_limits(limits=num_threads, user_api="openmp"):
            # Generate a random Hermitian matrix
            A = np.random.rand(matrix_size, matrix_size) + 1j * np.random.rand(
                matrix_size, matrix_size
            )
            A = A + A.conj().T

            # Diagonalize the matrix using numpy.linalg.eigh
            eigenvalues, eigenvectors = np.linalg.eigh(A)

    # Stop the timer
    execution_time = time.perf_counter_ns() - start_time

    return execution_time / 1e9


def benchmark_thread_matrix_size(matrix_sizes, thread_counts):
    """
    Benchmarks the execution time for different matrix sizes and thread counts.

    Args:
    matrix_sizes (list of int): List of matrix sizes to test.
    thread_counts (list of int): List of thread counts to test.

    Returns:
    dict: Dictionary of execution times with keys as thread counts and values as lists of times for each matrix size.
    """

    results = {threads: [] for threads in thread_counts}

    for size in matrix_sizes:
        for threads in thread_counts:
            exec_time = benchmark_hermitian_diagonalization(size, threads)
            results[threads].append(exec_time)
            print(
                f"Matrix size: {size}, Threads: {threads}, Time: {exec_time}"
            )

    return results


if __name__ == "__main__":
    # Define the range of matrix sizes and thread counts to test
    matrix_sizes = [
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        750,
        1024,
        1500,
        2048,
        2500,
        3000,
    ]  # Modify these values as needed
    thread_counts = [
        1,
        3,
        2,
        4,
        5,
        6,
        7,
        8,
        16,
        32,
        64,
    ]  # Modify these values as needed

    # Perform the benchmark
    benchmark_results = benchmark_thread_matrix_size(
        matrix_sizes, thread_counts
    )

    # Plotting
    plt.figure(figsize=(10, 6))
    for threads, times in benchmark_results.items():
        plt.plot(matrix_sizes, times, marker="o", label=f"{threads} threads")

    plt.xlabel("Matrix Size")
    plt.ylabel("Execution Time (s)")
    plt.title("Execution Time for Different Matrix Sizes and Thread Counts")
    plt.legend()
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True)
    plt.show()
