import pytest
import numpy as np

from PyQt4.QtGui import QTransform

import pyqthelp as ph

ARRAY_DATA_1 = [[1, 2, 3], [15, 12, 11], [21, 16, 23]]
IDENTITY_DATA = np.identity(3, dtype=np.float64)


def assert_transform_array_equals(qtransform, ndarray):
    assert isinstance(qtransform, QTransform)
    assert isinstance(ndarray, np.ndarray)

    print(qtransform.m11() - ndarray[0, 0])
    print(qtransform.m12() - ndarray[0, 1])
    print(qtransform.m13() - ndarray[0, 2])

    print(qtransform.m21() - ndarray[1, 0])
    print(qtransform.m22() - ndarray[1, 1])
    print(qtransform.m23() - ndarray[1, 2])

    print(qtransform.m31() - ndarray[2, 0])
    print(qtransform.m32() - ndarray[2, 1])
    print(qtransform.m33() - ndarray[2, 2])

    assert np.abs(qtransform.m11() - ndarray[0, 0]) < 1e-13
    assert np.abs(qtransform.m12() - ndarray[0, 1]) < 1e-13
    assert np.abs(qtransform.m13() - ndarray[0, 2]) < 1e-13

    assert np.abs(qtransform.m21() - ndarray[1, 0]) < 1e-13
    assert np.abs(qtransform.m22() - ndarray[1, 1]) < 1e-13
    assert np.abs(qtransform.m23() - ndarray[1, 2]) < 1e-13

    assert np.abs(qtransform.m31() - ndarray[2, 0]) < 1e-13
    assert np.abs(qtransform.m32() - ndarray[2, 1]) < 1e-13
    assert np.abs(qtransform.m33() - ndarray[2, 2]) < 1e-13


def make_array(data):
    return np.array(data, dtype=np.float64)


def make_transform(data):
    data = np.array(data, dtype=np.float64).ravel()
    data = tuple(data)
    return QTransform(*data)


def invert_numpy_array(array):
    try:
        res = np.linalg.inv(array)
        return res, True
    except np.linalg.LinAlgError:
        return np.identity(3, dtype=np.float64), False


@pytest.mark.parametrize('data', [ARRAY_DATA_1, IDENTITY_DATA])
def test_invert_works_correctly(data):

    transform = make_transform(data)

    array_inv, ok_np = invert_numpy_array(make_array(data))
    transform_inv, ok_ph = ph.native.invertTransform(transform)

    transform_inv2, ok_ph2 = transform.inverted()

    assert ok_ph == ok_ph2
    assert ok_np == ok_ph

    assert transform_inv == transform_inv2

    assert_transform_array_equals(transform_inv, array_inv)
    assert_transform_array_equals(transform_inv2, array_inv)

