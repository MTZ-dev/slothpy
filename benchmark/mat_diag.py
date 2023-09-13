import threadpoolctl
import numpy as np
import scipy.linalg
import time


def generate_random_hermitian_matrix(shape):
    # Generate a random complex matrix
    random_matrix = np.random.rand(*shape) + 1j * np.random.rand(*shape)

    # Make it Hermitian by adding its conjugate transpose
    hermitian_matrix = random_matrix + np.conj(random_matrix.T)

    return hermitian_matrix


def diagonalize_hermitian_matrix(matrix):
    # Diagonalize the Hermitian matrix
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    return eigenvalues, eigenvectors


def main():
    # Define the shape of the matrix
    matrix_shape = (2002, 2002)  # Change this to your desired shape

    # Generate a random Hermitian matrix
    random_hermitian_matrix = generate_random_hermitian_matrix(matrix_shape)
    with threadpoolctl.threadpool_limits(limits=2, user_api="blas"):
        with threadpoolctl.threadpool_limits(limits=2, user_api="openmp"):
            # Measure the time taken for diagonalization
            start_time = time.time()
            eigenvalues, eigenvectors = diagonalize_hermitian_matrix(
                random_hermitian_matrix
            )
            end_time = time.time()

    # Calculate and print the time taken
    diagonalization_time = end_time - start_time
    print(f"Diagonalization time: {diagonalization_time} seconds")


if __name__ == "__main__":
    # main()
    threads_to_check = []

    for i in range(1, 257):
        if 256 % i == 0:
            threads_to_check.append(i)

    for num_threads in threads_to_check:
        num_process = 256 // num_threads
        print(f"{num_threads} {num_process}")
