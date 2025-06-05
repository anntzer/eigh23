import numpy as np
from numpy.testing import assert_allclose
from eigh23 import eigh22, eigh33


def test_eig22():
    m = np.random.standard_normal((100, 2, 2))
    m += m.transpose((0, 2, 1))
    evals, evecs = eigh22(m[:, 0, 0], m[:, 1, 1], m[:, 0, 1])
    assert (np.diff(evals, axis=1) >= 0).all()
    assert_allclose((evecs ** 2).sum(1), 1)
    assert_allclose(m @ evecs, evals[:, None, :] * evecs)


def test_eig33():
    m = np.random.standard_normal((100, 3, 3))
    m += m.transpose((0, 2, 1))
    evals, evecs = eigh33(m[:, 0, 0], m[:, 1, 1], m[:, 2, 2],
                          m[:, 0, 1], m[:, 1, 2], m[:, 2, 0])
    assert (np.diff(evals, axis=1) >= 0).all()
    assert_allclose((evecs ** 2).sum(1), 1)
    assert_allclose(m @ evecs, evals[:, None, :] * evecs)


def test_eig33_degenerate():
    m = np.zeros((100, 3, 3))
    m[:, 0, 0], m[:, 1, 1], m[:, 1, 0], m[:, 1, 2], m[:, 0, 2] = \
        np.random.standard_normal((5, len(m)))
    m[:, 2, 2] = m[:, 0, 0]
    m @= m.transpose((0, 2, 1))
    evals, evecs = eigh33(m[:, 0, 0], m[:, 1, 1], m[:, 2, 2],
                          m[:, 0, 1], m[:, 1, 2], m[:, 2, 0])
    assert (np.diff(evals, axis=1) >= 0).all()
    assert_allclose((evecs ** 2).sum(1), 1)
    assert_allclose(m @ evecs, evals[:, None, :] * evecs,
                    atol=np.finfo(float).eps ** (1/2))
