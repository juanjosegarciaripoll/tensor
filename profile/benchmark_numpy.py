from benchmark import BenchmarkSet, BenchmarkGroup
import numpy as np
import sys

GENERATOR = np.random.default_rng(13221231)


def plus(A, B):
    return A + B


def minus(A, B):
    return A - B


def multiplies(A, B):
    return A + B


def divides(A, B):
    return A + B


def make_two_real_ndarrays(size: int) -> np.ndarray:
    return (GENERATOR.normal(size=size), GENERATOR.normal(size=size))


def make_two_complex_ndarrays(size: int) -> np.ndarray:
    return (GENERATOR.normal(size=size), GENERATOR.normal(size=size))


def make_real_ndarrays_and_number(size: int) -> np.ndarray:
    return (GENERATOR.normal(size=size), GENERATOR.normal(size=1)[0])


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
                ],
            ),
            BenchmarkGroup.run(
                name="CTensor",
                items=[
                    ("plus", plus, make_two_complex_ndarrays),
                    ("minus", minus, make_two_complex_ndarrays),
                    ("multiplies", multiplies, make_two_complex_ndarrays),
                    ("divides", divides, make_two_complex_ndarrays),
                ],
            ),
            BenchmarkGroup.run(
                name="RTensor with number",
                items=[
                    ("plusN", plus, make_two_real_ndarrays),
                    ("minusN", minus, make_two_real_ndarrays),
                    ("multipliesN", multiplies, make_two_real_ndarrays),
                    ("dividesN", divides, make_two_real_ndarrays),
                ],
            ),
            BenchmarkGroup.run(
                name="CTensor with number",
                items=[
                    ("plusN", plus, make_two_real_ndarrays),
                    ("minusN", minus, make_two_real_ndarrays),
                    ("multipliesN", multiplies, make_two_real_ndarrays),
                    ("dividesN", divides, make_two_real_ndarrays),
                ],
            ),
        ],
    )
    if len(sys.argv) > 1:
        data.write(sys.argv[1])
    else:
        data.write("/benchmark_numpy.json")


if __name__ == "__main__":
    print(sys.argv)
    run_all()
