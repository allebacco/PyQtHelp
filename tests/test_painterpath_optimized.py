
import pytest
import numpy as np

from PyQt5.QtGui import QPainterPath, QTransform

import pyqthelp as ph

INPUT_ARRAY_X = np.arange(0, 10, 1).astype(np.float64)
INPUT_ARRAY_Y = np.arange(0, 10, 1).astype(np.float64)
INPUT_CONNECT = np.ones(INPUT_ARRAY_Y.size - 1, dtype=np.bool)


@pytest.mark.parametrize('x,y,connect', [(INPUT_ARRAY_X, INPUT_ARRAY_Y, INPUT_CONNECT)])
def test_create_path_optimized_identity(x, y, connect):

    path = ph.native.arrayToQPathOptimized(x, y, connect, QTransform(), 1)

    assert path.elementCount() == len(x)
    assert path.elementCount() == len(y)

    for i in range(path.elementCount()):
        element = path.elementAt(i)
        assert element.x == x[i]
        assert element.y == y[i]


@pytest.mark.parametrize('x,y,connect', [(INPUT_ARRAY_X, INPUT_ARRAY_Y, INPUT_CONNECT)])
def test_create_path_optimized_xscale(x, y, connect):
    transform = QTransform.fromScale(2.0, 3.0) * QTransform.fromTranslate(0.5, 2.2)
    path = ph.native.arrayToQPathOptimized(x, y, connect, transform, 1)

    assert path.elementCount() == len(x)
    assert path.elementCount() == len(y)

    for i in range(path.elementCount()):
        element = path.elementAt(i)
        assert element.x == 0.5 + x[i] * 2.0
        assert element.y == 2.2 + y[i] * 3.0
