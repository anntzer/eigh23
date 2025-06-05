import numpy as np
import pytest
import scipy as sp

from eigh23 import eigh22, eigh33


@pytest.mark.benchmark(group="eigh22")
@pytest.mark.parametrize(
    "impl", [
        "np",
        "sp-ev", "sp-ev-nocheck",
        "sp-evd", "sp-evd-nocheck",
        "sp-evr", "sp-evr-nocheck",
        "sp-evx", "sp-evx-nocheck",
        "eigh",
    ])
def test_eigh22(impl, benchmark):
    np.random.seed(1)
    m = np.random.standard_normal((100, 2, 2))
    m += m.transpose((0, 2, 1))
    if impl == "np":
        benchmark(np.linalg.eigh, m)
    elif impl == "sp-ev":
        benchmark(sp.linalg.eigh, m, driver="ev")
    elif impl == "sp-ev-nocheck":
        benchmark(sp.linalg.eigh, m, driver="ev", check_finite=False)
    elif impl == "sp-evd":
        benchmark(sp.linalg.eigh, m, driver="evd")
    elif impl == "sp-evd-nocheck":
        benchmark(sp.linalg.eigh, m, driver="evd", check_finite=False)
    elif impl == "sp-evr":
        benchmark(sp.linalg.eigh, m, driver="evr")
    elif impl == "sp-evr-nocheck":
        benchmark(sp.linalg.eigh, m, driver="evr", check_finite=False)
    elif impl == "sp-evx":
        benchmark(sp.linalg.eigh, m, driver="evx")
    elif impl == "sp-evx-nocheck":
        benchmark(sp.linalg.eigh, m, driver="evx", check_finite=False)
    elif impl == "eigh":
        benchmark(eigh22, m[:, 0, 0], m[:, 1, 1], m[:, 0, 1])


@pytest.mark.benchmark(group="eigh33")
@pytest.mark.parametrize(
    "impl", [
        "np",
        "sp-ev", "sp-ev-nocheck",
        "sp-evd", "sp-evd-nocheck",
        "sp-evr", "sp-evr-nocheck",
        "sp-evx", "sp-evx-nocheck",
        "eigh",
    ])
def test_eig33(impl, benchmark):
    np.random.seed(1)
    m = np.random.standard_normal((100, 3, 3))
    m += m.transpose((0, 2, 1))
    if impl == "np":
        benchmark(np.linalg.eigh, m)
    elif impl == "sp-ev":
        benchmark(sp.linalg.eigh, m, driver="ev")
    elif impl == "sp-ev-nocheck":
        benchmark(sp.linalg.eigh, m, driver="ev", check_finite=False)
    elif impl == "sp-evd":
        benchmark(sp.linalg.eigh, m, driver="evd")
    elif impl == "sp-evd-nocheck":
        benchmark(sp.linalg.eigh, m, driver="evd", check_finite=False)
    elif impl == "sp-evr":
        benchmark(sp.linalg.eigh, m, driver="evr")
    elif impl == "sp-evr-nocheck":
        benchmark(sp.linalg.eigh, m, driver="evr", check_finite=False)
    elif impl == "sp-evx":
        benchmark(sp.linalg.eigh, m, driver="evx")
    elif impl == "sp-evx-nocheck":
        benchmark(sp.linalg.eigh, m, driver="evx", check_finite=False)
    elif impl == "eigh":
        benchmark(eigh33,
                  m[:, 0, 0], m[:, 1, 1], m[:, 2, 2],
                  m[:, 0, 1], m[:, 1, 2], m[:, 2, 0])
