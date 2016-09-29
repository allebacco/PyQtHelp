
import pytest
import numpy as np

from PyQt4.QtGui import QPainterPath

import pyqthelp as ph

ARRAY_X_DATA = np.arange(100)
ARRAY_Y_DATA = np.random.rand(100) * 100.


def create_qpainterpath(x, y, connect):
    if isinstance(connect, np.ndarray):
        path = QPainterPath()
        path.moveTo(x[0], y[0])
        for i in range(1, x.size):
            if connect[i-1]:
                path.lineTo(x[i], y[i])
            else:
                path.moveTo(x[i], y[i])
        return path
    elif connect == 'all':
        path = QPainterPath()
        path.moveTo(x[0], y[0])
        for i in range(1, x.size):
            path.lineTo(x[i], y[i])
        return path
    elif connect == 'finite':
        connect = np.isfinite(x) & np.isfinite(y)
        return create_qpainterpath(x, y, connect)
    elif connect == 'pairs':
        connect = np.ones(x.size - 1, dtype=np.bool)
        connect[1::2] = 0
        return create_qpainterpath(x, y, connect)


def assert_path_equals(path1, path2):

    assert path1.elementCount() == path2.elementCount()

    for i in range(path1.elementCount()):
        element1 = path1.elementAt(i)
        element2 = path2.elementAt(i)

        assert element1.x == element2.x
        assert element1.y == element2.y
        assert element1.type == element2.type


@pytest.mark.parametrize('xtype,ytype', [
    (np.float64, np.float64),
    (np.float32, np.float32),
    (np.int64, np.float64),
    (np.int64, np.int64),
    (np.int32, np.int32),
    (np.uint64, np.uint64),
    (np.uint32, np.uint32),
])
@pytest.mark.parametrize('x', [ARRAY_X_DATA])
@pytest.mark.parametrize('y', [ARRAY_Y_DATA])
@pytest.mark.parametrize('connect', ['all', 'finite', 'pairs'])
def test_arrayToQPath_str(xtype, ytype, x, y, connect):

    x = x.astype(xtype)
    y = y.astype(ytype)

    test_path = ph.native.arrayToQPath(x, y, connect)
    ref_path = create_qpainterpath(x, y, connect)

    assert_path_equals(test_path, ref_path)
