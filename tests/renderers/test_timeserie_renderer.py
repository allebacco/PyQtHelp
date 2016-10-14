import pytest
import numpy as np

from PyQt5.Qt import Qt
from PyQt5.QtCore import QRect
from PyQt5.QtGui import QImage, QPainter, QPen

from pyqthelp.renderers import TimeserieRenderer
from pyqthelp.path import arrayToQPath


def create_reference_image(dataX, dataY, size, pen, fill=Qt.black,
                           transform=None, viewport=None):
    image = QImage(size[0], size[1], QImage.Format_ARGB32)
    image.fill(Qt.black)

    painter = QPainter()
    painter.begin(image)
    painter.setPen(pen)

    path = arrayToQPath(dataX, dataY, 'all')

    if transform is not None:
        path = path * transform

    if viewport is not None:
        painter.setWindow(viewport)

    painter.drawPath(path)
    painter.end()

    return image


@pytest.mark.parametrize('viewport', [None, QRect(-50, -20, 200, 100)])
def test_basic_rendering(qtbot, viewport):

    numElements = 100
    imageSize = (200, 100)
    pen = QPen(Qt.white, 1.)

    dataY = np.random.rand(numElements) * 50
    dataX = np.arange(numElements) + 10.

    refImage = create_reference_image(dataX, dataY, imageSize, pen, viewport=viewport)

    image = QImage(imageSize[0], imageSize[1], QImage.Format_ARGB32)
    image.fill(Qt.black)

    renderer = TimeserieRenderer()
    renderer.setPen(pen)
    renderer.setData(dataX, dataY)

    painter = QPainter()
    painter.begin(image)
    if viewport is not None:
        painter.setWindow(viewport)

    renderer.render(painter)

    painter.end()

    # Save the image
    # if viewport is not None:
    #     refImage.save('ref_image_viewport.png')
    #     image.save('test_image_viewport.png')
    # else:
    #     refImage.save('ref_image.png')
    #     image.save('test_image.png')

    assert refImage == image
