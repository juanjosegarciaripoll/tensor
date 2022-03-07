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


def copy_first_column(A, B):
    A[:, 0] = B[:, 0]


def copy_first_row(A, B):
    A[0, :] = A[0, :]


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


def make_two_real_matrices(size: int, cols: int = 10) -> np.ndarray:
    warmup(size)
    return (GENERATOR.normal(size=(size, cols)), GENERATOR.normal(size=(size, cols)))


def make_two_real_matrices_t(size: int, rows: int = 10) -> np.ndarray:
    warmup(size)
    return (GENERATOR.normal(size=(rows, size)), GENERATOR.normal(size=(rows, size)))


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


def make_two_complex_matrices(size: int, cols: int = 20) -> np.ndarray:
    warmup(size*cols, np.complex128)
    return (GENERATOR.normal(size=(size, cols)), GENERATOR.normal(size=(size, cols)))


def make_two_complex_matrices_t(size: int, rows: int = 20) -> np.ndarray:
    warmup(size*rows, np.complex128)
    return (GENERATOR.normal(size=(rows, size)), GENERATOR.normal(size=(rows, size)))


def make_complex_ndarray_and_number(size: int) -> np.ndarray:
    a1, b1 = make_real_ndarray_and_number(size)
    for _ in range(10):
        a2 = np.empty(size, dtype=np.complex128)
    return (a1 + 1j * a2, 3.0 + 0.0j)


def make_complex_ndarray_and_one(size: int) -> np.ndarray:
    a1, b1 = make_real_ndarray_and_number(size)
    for _ in range(10):
        a2 = np.empty(size, dtype=np.complex128)
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
                name="RTensor",
                items=[
                    ("plus", plus, make_two_real_ndarrays),
                    ("minus", minus, make_two_real_ndarrays),
                    ("multiplies", multiplies, make_two_real_ndarrays),
                    ("divides", divides, make_two_real_ndarrays),
                    ("copy_column", copy_first_row, make_two_real_matrices),
                    ("copy_row", copy_first_column, make_two_real_matrices_t),
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
                    ("copy_column", copy_first_row, make_two_complex_matrices),
                    ("copy_row", copy_first_column, make_two_complex_matrices_t),
                    ("sum", apply_sum, make_complex_ndarray_and_one),
                    ("exp", apply_exp, make_complex_ndarray_and_one),
                    ("cos", apply_cos, make_complex_ndarray_and_one),
                ],
            ),
            BenchmarkGroup.run(
                name="RTensor with number",
                items=[
                    ("plusN", plus, make_real_ndarray_and_number),
                    ("minusN", minus, make_real_ndarray_and_number),
                    ("multipliesN", multiplies, make_real_ndarray_and_number),
                    ("dividesN", divides, make_real_ndarray_and_number),
                    ("plusNinplace", plus_inplace, make_real_ndarray_and_one),
                    ("minusNinplace", minus_inplace, make_real_ndarray_and_one),
                    (
                        "multipliesNinplace",
                        multiplies_inplace,
                        make_real_ndarray_and_one,
                    ),
                    ("dividesNinplace", divides_inplace, make_real_ndarray_and_one),
                ],
            ),
            BenchmarkGroup.run(
                name="CTensor with number",
                items=[
                    ("plusN", plus, make_complex_ndarray_and_number),
                    ("minusN", minus, make_complex_ndarray_and_number),
                    ("multipliesN", multiplies, make_complex_ndarray_and_number),
                    ("dividesN", divides, make_complex_ndarray_and_number),
                    ("plusNinplace", plus_inplace, make_complex_ndarray_and_one),
                    ("minusNinplace", minus_inplace, make_complex_ndarray_and_one),
                    (
                        "multipliesNinplace",
                        multiplies_inplace,
                        make_complex_ndarray_and_one,
                    ),
                    ("dividesNinplace", divides_inplace,
                     make_complex_ndarray_and_one),
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
