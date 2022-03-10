from typing import Optional
from benchmark import BenchmarkSet, BenchmarkGroup
import numpy as np
import sys

GENERATOR = np.random.default_rng(13221231)


def plus(A, B):
    return A + B


def minus(A, B):
    return A - B


def multiplies(A, B):
    return A * B


def divides(A, B):
    return A / B


def plus_inplace(A, B):
    A += B
    return A


def minus_inplace(A, B):
    A -= B
    return A


def multiplies_inplace(A, B):
    A *= B
    return A


def divides_inplace(A, B):
    A /= B
    return A


def extract_first_column(A, B):
    return B[:, 0].copy()


def extract_first_row(A, B):
    return B[0, :].copy()


def copy_first_column(A, B):
    A[:, 0] = B[:, 0]


def copy_first_row(A, B):
    A[0, :] = B[0, :]


def copy_first_column_index(A, B, ndx):
    A[ndx, 0] = B[ndx, 0]


def copy_first_row_index(A, B, ndx):
    A[0, ndx] = B[0, ndx]


def apply_sum(A, B):
    return np.sum(A)


def apply_cos(A, B):
    return np.cos(A)


def apply_exp(A, B):
    return np.exp(A)


def fold_ij_jk(A, B):
    return np.einsum("ji,kj->ik", A, B)


def fold_ij_kj(A, B):
    return np.einsum("ji,jk->ik", A, B)


def fold_ji_jk(A, B):
    return np.einsum("ij,kj->ik", A, B)


def fold_ji_kj(A, B):
    return np.einsum("ij,jk->ik", A, B)


def mmult_N_N(A, B):
    return A @ B


def mmult_N_T(A, B):
    return A @ B.T


def mmult_T_N(A, B):
    return A.T @ B


def mmult_T_T(A, B):
    return A.T @ B.T


def warmup(size, dtype=np.double):
    for _ in range(10):
        a2 = np.empty(size, dtype=dtype)


def make_random_real_array(*dimensions) -> np.ndarray:
    print(dimensions)
    return GENERATOR.random(size=dimensions)


def make_two_real_ndarrays(size: int) -> np.ndarray:
    return (make_random_real_array(size), make_random_real_array(size))


def make_two_real_columns(size: int, cols: int = 50) -> np.ndarray:
    return (make_random_real_array(size, cols), make_random_real_array(size, cols))


def make_two_real_rows(size: int, rows: int = 50) -> np.ndarray:
    return (make_random_real_array(rows, size), make_random_real_array(rows, size))


def make_two_real_rows_and_index(size: int, rows: int = 50) -> np.ndarray:
    return (
        make_random_real_array(rows, size),
        make_random_real_array(rows, size),
        np.arange(0, size, 2),
    )


def make_two_real_columns_and_index(size: int, cols: int = 50) -> np.ndarray:
    return (
        make_random_real_array(size, cols),
        make_random_real_array(size, cols),
        np.arange(0, size, 2),
    )


def make_real_ndarray_and_number(size: int) -> np.ndarray:
    return (make_random_real_array(size), 3.0)


def make_real_ndarray_and_one(size: int) -> np.ndarray:
    return (make_random_real_array(size), 1.0)


def make_random_complex_array(*dimensions) -> np.ndarray:
    return GENERATOR.random(size=dimensions) + 1j * GENERATOR.random(size=dimensions)


def make_two_complex_ndarrays(size: int) -> np.ndarray:
    return (make_random_complex_array(size), make_random_complex_array(size))


def make_two_complex_columns(size: int, cols: int = 50) -> np.ndarray:
    return (
        make_random_complex_array(size, cols),
        make_random_complex_array(size, cols),
    )


def make_two_complex_rows(size: int, rows: int = 50) -> np.ndarray:
    return (
        make_random_complex_array(rows, size),
        make_random_complex_array(rows, size),
    )


def make_two_complex_rows_and_index(size: int, rows: int = 50) -> np.ndarray:
    return (
        make_random_complex_array(rows, size),
        make_random_complex_array(rows, size),
        np.arange(0, size, 2),
    )


def make_two_complex_columns_and_index(size: int, cols: int = 50) -> np.ndarray:
    return (
        make_random_complex_array(size, cols),
        make_random_complex_array(size, cols),
        np.arange(0, size, 2),
    )


def make_complex_ndarray_and_number(size: int) -> np.ndarray:
    return (make_random_complex_array(size), 3.0)


def make_complex_ndarray_and_one(size: int) -> np.ndarray:
    return (make_random_complex_array(size), 1.0)


def make_two_real_matrices(size: int) -> np.ndarray:
    return (make_random_real_array(size, size), make_random_real_array(size, size))


def make_two_complex_matrices(size: int) -> np.ndarray:
    return (
        make_random_complex_array(size, size),
        make_random_complex_array(size, size),
    )


def system_version():
    v = sys.version_info
    return f"Python {v.major}.{v.minor}.{v.micro} NumPy {np.version.full_version}"


def run_all():
    small_sizes = [2**n for n in range(2, 12, 1)]
    warmup(100 * 8 * 4194304)
    data = BenchmarkSet(
        name="Numpy",
        environment=system_version(),
        groups=[
            BenchmarkGroup.run(
                name="CTensor access",
                items=[
                    ("copy_N0", copy_first_row_index, make_two_complex_rows_and_index),
                    (
                        "copy_0N",
                        copy_first_column_index,
                        make_two_complex_columns_and_index,
                    ),
                    ("extract_i0", extract_first_row, make_two_complex_rows),
                    ("extract_0i", extract_first_column, make_two_complex_columns),
                    ("copy_i0", copy_first_row, make_two_complex_rows),
                    ("copy_0i", copy_first_column, make_two_complex_columns),
                    ("fold_ij_jk", fold_ij_jk, make_two_complex_matrices, small_sizes),
                    ("fold_ij_kj", fold_ij_kj, make_two_complex_matrices, small_sizes),
                    ("fold_ji_jk", fold_ji_jk, make_two_complex_matrices, small_sizes),
                    ("fold_ji_kj", fold_ji_kj, make_two_complex_matrices, small_sizes),
                    ("mmult_N_N", mmult_N_N, make_two_complex_matrices, small_sizes),
                    ("mmult_N_T", mmult_N_T, make_two_complex_matrices, small_sizes),
                    ("mmult_T_N", mmult_T_N, make_two_complex_matrices, small_sizes),
                    ("mmult_T_T", mmult_T_T, make_two_complex_matrices, small_sizes),
                ],
            ),
            BenchmarkGroup.run(
                name="RTensor access",
                items=[
                    ("copy_N0", copy_first_row_index, make_two_real_rows_and_index),
                    (
                        "copy_0N",
                        copy_first_column_index,
                        make_two_real_columns_and_index,
                    ),
                    ("extract_i0", extract_first_row, make_two_real_rows),
                    ("extract_0i", extract_first_column, make_two_real_columns),
                    ("copy_i0", copy_first_row, make_two_real_rows),
                    ("copy_0i", copy_first_column, make_two_real_columns),
                    ("fold_ij_jk", fold_ij_jk, make_two_real_matrices, small_sizes),
                    ("fold_ij_kj", fold_ij_kj, make_two_real_matrices, small_sizes),
                    ("fold_ji_jk", fold_ji_jk, make_two_real_matrices, small_sizes),
                    ("fold_ji_kj", fold_ji_kj, make_two_real_matrices, small_sizes),
                    ("mmult_N_N", mmult_N_N, make_two_real_matrices, small_sizes),
                    ("mmult_N_T", mmult_N_T, make_two_real_matrices, small_sizes),
                    ("mmult_T_N", mmult_T_N, make_two_real_matrices, small_sizes),
                    ("mmult_T_T", mmult_T_T, make_two_real_matrices, small_sizes),
                ],
            ),
            BenchmarkGroup.run(
                name="RTensor",
                items=[
                    ("plus", plus, make_two_real_ndarrays),
                    ("minus", minus, make_two_real_ndarrays),
                    ("multiplies", multiplies, make_two_real_ndarrays),
                    ("divides", divides, make_two_real_ndarrays),
                    ("plus_N", plus, make_real_ndarray_and_number),
                    ("minus_N", minus, make_real_ndarray_and_number),
                    ("multiplies_N", multiplies, make_real_ndarray_and_number),
                    ("divides_N", divides, make_real_ndarray_and_number),
                    ("plus_N_inplace", plus_inplace, make_real_ndarray_and_one),
                    ("minus_N_inplace", minus_inplace, make_real_ndarray_and_one),
                    (
                        "multiplies_N_inplace",
                        multiplies_inplace,
                        make_real_ndarray_and_one,
                    ),
                    ("divides_N_inplace", divides_inplace, make_real_ndarray_and_one),
                    ("sum", apply_sum, make_real_ndarray_and_number),
                    ("exp", apply_exp, make_real_ndarray_and_number),
                    ("cos", apply_cos, make_real_ndarray_and_number),
                ],
            ),
            BenchmarkGroup.run(
                name="CTensor",
                items=[
                    ("plus", plus, make_two_complex_ndarrays),
                    ("minus", minus, make_two_complex_ndarrays),
                    ("multiplies", multiplies, make_two_complex_ndarrays),
                    ("divides", divides, make_two_complex_ndarrays),
                    ("plus_N", plus, make_complex_ndarray_and_number),
                    ("minus_N", minus, make_complex_ndarray_and_number),
                    ("multiplies_N", multiplies, make_complex_ndarray_and_number),
                    ("divides_N", divides, make_complex_ndarray_and_number),
                    ("plus_N_inplace", plus_inplace, make_complex_ndarray_and_one),
                    ("minus_N_inplace", minus_inplace, make_complex_ndarray_and_one),
                    (
                        "multiplies_N_inplace",
                        multiplies_inplace,
                        make_complex_ndarray_and_one,
                    ),
                    (
                        "divides_N_inplace",
                        divides_inplace,
                        make_complex_ndarray_and_one,
                    ),
                    ("sum", apply_sum, make_complex_ndarray_and_one),
                    ("exp", apply_exp, make_complex_ndarray_and_one),
                    ("cos", apply_cos, make_complex_ndarray_and_one),
                ],
            ),
        ],
    )
    if len(sys.argv) > 1:
        data.write(sys.argv[1])
    else:
        data.write("./benchmark_numpy.json")


if __name__ == "__main__":
    print(sys.argv)
    run_all()
