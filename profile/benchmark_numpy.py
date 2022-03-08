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


def warmup(size, dtype=np.double):
    for _ in range(10):
        a2 = np.empty(size, dtype=dtype)


def make_two_real_ndarrays(size: int) -> np.ndarray:
    warmup(size)
    return (GENERATOR.normal(size=size), GENERATOR.normal(size=size))


def make_two_real_columns(size: int, cols: int = 100) -> np.ndarray:
    warmup(size)
    return (GENERATOR.normal(size=(size, cols)), GENERATOR.normal(size=(size, cols)))


def make_two_real_rows(size: int, rows: int = 100) -> np.ndarray:
    warmup(size)
    return (GENERATOR.normal(size=(rows, size)), GENERATOR.normal(size=(rows, size)))


def make_two_real_rows_and_index(size: int, rows: int = 100) -> np.ndarray:
    warmup(size)
    ndx = np.arange(0, size, 2)
    return (GENERATOR.normal(size=(rows, size)), GENERATOR.normal(size=(rows, size)), ndx)


def make_two_real_columns_and_index(size: int, cols: int = 100) -> np.ndarray:
    warmup(size)
    ndx = np.arange(0, size, 2)
    return (GENERATOR.normal(size=(size, cols)), GENERATOR.normal(size=(size, cols)), ndx)


def make_real_ndarray_and_number(size: int) -> np.ndarray:
    warmup(size)
    return (GENERATOR.normal(size=size), 3.0)


def make_real_ndarray_and_one(size: int) -> np.ndarray:
    warmup(size)
    return (GENERATOR.normal(size=size), 1.0)


def make_two_complex_ndarrays(size: int) -> np.ndarray:
    warmup(size, np.complex128)
    a1, b1 = make_two_real_ndarrays(size)
    a2, b2 = make_two_real_ndarrays(size)
    return (a1 + 1j * a2, b1 + 1j * b2)


def make_two_complex_columns(size: int, cols: int = 100) -> np.ndarray:
    warmup(size*cols, np.complex128)
    return (GENERATOR.normal(size=(size, cols)), GENERATOR.normal(size=(size, cols)))


def make_two_complex_rows(size: int, rows: int = 100) -> np.ndarray:
    warmup(size*rows, np.complex128)
    return (GENERATOR.normal(size=(rows, size)), GENERATOR.normal(size=(rows, size)))


def make_two_complex_columns_and_index(size: int, cols: int = 100) -> np.ndarray:
    warmup(size*cols, np.complex128)
    ndx = np.arange(0, size, 2)
    return (GENERATOR.normal(size=(size, cols)), GENERATOR.normal(size=(size, cols)), ndx)


def make_two_complex_rows_and_index(size: int, rows: int = 100) -> np.ndarray:
    warmup(size*rows, np.complex128)
    ndx = np.arange(0, size, 2)
    return (GENERATOR.normal(size=(rows, size)), GENERATOR.normal(size=(rows, size)), ndx)


def make_complex_ndarray_and_number(size: int) -> np.ndarray:
    warmup(size, np.complex128)
    a1, a2 = make_two_real_ndarrays(size)
    return (a1 + 1j * a2, 3.0 + 0.0j)


def make_complex_ndarray_and_one(size: int) -> np.ndarray:
    warmup(size, np.complex128)
    a1, a2 = make_two_real_ndarrays(size)
    return (a1 + 1j * a2, 1.0 + 0.0j)


def system_version():
    v = sys.version_info
    return f"Python {v.major}.{v.minor}.{v.micro} NumPy {np.version.full_version}"


def run_all():
    data = BenchmarkSet(
        name="Numpy",
        environment=system_version(),
        groups=[
            BenchmarkGroup.run(
                name="CTensor access",
                items=[
                    ("copy_N0", copy_first_row_index,
                     make_two_complex_rows_and_index),
                    ("copy_0N", copy_first_column_index,
                     make_two_complex_columns_and_index),
                    ("extract_i0", extract_first_row, make_two_complex_rows),
                    ("extract_0i", extract_first_column, make_two_complex_columns),
                    ("copy_i0", copy_first_row, make_two_complex_rows),
                    ("copy_0i", copy_first_column, make_two_complex_columns),
                ],
            ),
            BenchmarkGroup.run(
                name="RTensor access",
                items=[
                    ("copy_N0", copy_first_row_index,
                     make_two_real_rows_and_index),
                    ("copy_0N", copy_first_column_index,
                     make_two_real_columns_and_index),
                    ("extract_i0", extract_first_row, make_two_real_rows),
                    ("extract_0i", extract_first_column, make_two_real_columns),
                    ("copy_i0", copy_first_row, make_two_real_rows),
                    ("copy_0i", copy_first_column, make_two_real_columns),
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
                    ("divides_N_inplace", divides_inplace,
                     make_real_ndarray_and_one),
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
                    ("minus_N_inplace", minus_inplace,
                     make_complex_ndarray_and_one),
                    (
                        "multiplies_N_inplace",
                        multiplies_inplace,
                        make_complex_ndarray_and_one,
                    ),
                    ("divides_N_inplace", divides_inplace,
                     make_complex_ndarray_and_one),
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
