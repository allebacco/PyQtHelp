import struct
import sys
import numpy as np

from PyQt5.Qt import pyqtSignal
from pyqtgraph import GraphicsObject, functions, Point


__all__ = ['PlotTimeserieItem']


class PlotTimeserieItem(GraphicsObject):
    """
    Class representing a plot of a time serie data.
    Features:

    - Fast data update
    - Fill under curve
    - Mouse interaction

    ====================  ===============================================
    **Signals:**
    sigPlotChanged(self)  Emitted when the data being plotted has changed
    sigClicked(self)      Emitted when the curve is clicked
    ====================  ===============================================
    """

    sigPlotChanged = pyqtSignal(object)
    sigClicked = pyqtSignal(object)

    def __init__(self, *args, **kargs):
        """
        Forwards all arguments to :func:`setData <pyqtgraph.PlotTimeserieItem.setData>`.

        Some extra arguments are accepted as well:

        ==============  =======================================================
        **Arguments:**
        parent          The parent GraphicsObject (optional)
        clickable       If True, the item will emit sigClicked when it is
                        clicked on. Defaults to False.
        ==============  =======================================================
        """
        GraphicsObject.__init__(self, kargs.get('parent', None))
        self.clear()

        self._pixelPadding = None

        self.metaData = {}
        self.opts = {
            'pen': fn.mkPen('w'),
            'shadowPen': None,
            'fillLevel': None,
            'brush': None,
            'stepMode': False,
            'name': None,
            'antialias': getConfigOption('antialias'),
            'connect': 'all',
            'mouseWidth': 8,  # width of shape responding to mouse click
        }
        self.setClickable(kargs.get('clickable', False))
        self.setData(*args, **kargs)

    def implements(self, interface=None):
        ints = ['plotData']
        if interface is None:
            return ints
        return interface in ints

    def name(self):
        return self.opts.get('name', None)

    def setClickable(self, s, width=None):
        """Sets whether the item responds to mouse clicks.

        The *width* argument specifies the width in pixels orthogonal to the
        curve that will respond to a mouse click.
        """
        self.clickable = s
        if width is not None:
            self.opts['mouseWidth'] = width
            self._mouseShape = None
            self._boundingRect = None

    def getData(self):
        return self.xData, self.yData

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        # Need this to run as fast as possible.
        # check cache first:
        cache = self._boundsCache[ax]
        if cache is not None and cache[0] == (frac, orthoRange):
            return cache[1]

        (x, y) = self.getData()
        if x is None or len(x) == 0:
            return (None, None)

        if ax == 0:
            d = x
            d2 = y
        elif ax == 1:
            d = y
            d2 = x

        # If an orthogonal range is specified, mask the data now
        if orthoRange is not None:
            mask = (d2 >= orthoRange[0]) * (d2 <= orthoRange[1])
            d = d[mask]

        if len(d) == 0:
            return (None, None)

        # Get min/max (or percentiles) of the requested data range
        if frac >= 1.0:
            b = (np.nanmin(d), np.nanmax(d))
        elif frac <= 0.0:
            raise Exception("Value for parameter 'frac' must be > 0. (got %s)" % str(frac))
        else:
            mask = np.isfinite(d)
            d = d[mask]
            b = np.percentile(d, [50 * (1 - frac), 50 * (1 + frac)])

        # adjust for fill level
        if ax == 1 and self.opts['fillLevel'] is not None:
            b = (min(b[0], self.opts['fillLevel']), max(b[1], self.opts['fillLevel']))

        # Add pen width only if it is non-cosmetic.
        pen = self.opts['pen']
        spen = self.opts['shadowPen']
        if not pen.isCosmetic():
            b = (b[0] - pen.widthF()*0.7072, b[1] + pen.widthF()*0.7072)
        if spen is not None and not spen.isCosmetic() and spen.style() != QtCore.Qt.NoPen:
            b = (b[0] - spen.widthF()*0.7072, b[1] + spen.widthF()*0.7072)

        self._boundsCache[ax] = [(frac, orthoRange), b]
        return b

    def pixelPadding(self):
        if self._pixelPadding is None:
            pen = self.opts['pen']
            spen = self.opts['shadowPen']
            w = 0
            if pen.isCosmetic():
                w += pen.widthF()*0.7072
            if spen is not None and spen.isCosmetic() and spen.style() != QtCore.Qt.NoPen:
                w = max(w, spen.widthF()*0.7072)
            if self.clickable:
                w = max(w, self.opts['mouseWidth']//2 + 1)
            self._pixelPadding = w
        return self._pixelPadding

    def boundingRect(self):
        if self._boundingRect is None:
            br = self.getPath().boundingRect()

            pxPad = self.pixelPadding()
            if pxPad > 0:
                # determine length of pixel in local x, y directions
                dt = self.deviceTransform()
                px = 1.0/dt.m11() * pxPad
                py = abs(1.0/dt.m22()) * pxPad
                br.adjust(-px, -py, px, py)
            self._boundingRect = br

        return self._boundingRect

    def viewTransformChanged(self):
        self._boundingRect = None
        self.prepareGeometryChange()

    def invalidateBounds(self):
        self._pixelPadding = None
        self._boundingRect = None
        self._boundsCache = [None, None]

    def setPen(self, *args, **kargs):
        """Set the pen used to draw the curve."""
        self.opts['pen'] = fn.mkPen(*args, **kargs)
        self.invalidateBounds()
        self.update()

    def setBrush(self, *args, **kargs):
        """Set the brush used when filling the area under the curve"""
        self.opts['brush'] = fn.mkBrush(*args, **kargs)
        self.invalidateBounds()
        self.update()

    def setFillLevel(self, level):
        """Set the level filled to when filling under the curve"""
        self.opts['fillLevel'] = level
        self.fillPath = None
        self.invalidateBounds()
        self.update()

    def setData(self, *args, **kargs):
        """
        ==============  ========================================================
        **Arguments:**
        x, y            (numpy arrays) Data to show
        pen             Pen to use when drawing. Any single argument accepted by
                        :func:`mkPen <pyqtgraph.mkPen>` is allowed.
        fillLevel       (float or None) Fill the area 'under' the curve to
                        *fillLevel*
        brush           QBrush to use when filling. Any single argument accepted
                        by :func:`mkBrush <pyqtgraph.mkBrush>` is allowed.
        antialias       (bool) Whether to use antialiasing when drawing. This
                        is disabled by default because it decreases performance.
        connect         Argument specifying how vertexes should be connected
                        by line segments. Default is "all", indicating full
                        connection. "pairs" causes only even-numbered segments
                        to be drawn. "finite" causes segments to be omitted if
                        they are attached to nan or inf values. For any other
                        connectivity, specify an array of boolean values.
        ==============  ========================================================

        If non-keyword arguments are used, they will be interpreted as
        setData(y) for a single argument and setData(x, y) for two
        arguments.


        """
        self.updateData(*args, **kargs)

    def updateData(self, *args, **kargs):
        profiler = debug.Profiler()

        if len(args) == 1:
            kargs['y'] = args[0]
        elif len(args) == 2:
            kargs['x'] = args[0]
            kargs['y'] = args[1]

        if 'y' not in kargs or kargs['y'] is None:
            kargs['y'] = np.array([])
        if 'x' not in kargs or kargs['x'] is None:
            kargs['x'] = np.arange(len(kargs['y']))

        for k in ['x', 'y']:
            data = kargs[k]
            if isinstance(data, list):
                data = np.array(data)
                kargs[k] = data
            if not isinstance(data, np.ndarray) or data.ndim > 1:
                raise Exception("Plot data must be 1D ndarray.")
            if 'complex' in str(data.dtype):
                raise Exception("Can not plot complex data types.")

        profiler("data checks")

        self.invalidateBounds()
        self.prepareGeometryChange()
        self.informViewBoundsChanged()
        self.yData = kargs['y'].view(np.ndarray)
        self.xData = kargs['x'].view(np.ndarray)

        profiler('copy')

        if self.xData.shape != self.yData.shape:  ## allow difference of 1 for step mode plots
            raise Exception("X and Y arrays must be the same shape--got %s and %s." % (self.xData.shape, self.yData.shape))

        self.path = None
        self.fillPath = None
        self._mouseShape = None

        if 'name' in kargs:
            self.opts['name'] = kargs['name']
        if 'connect' in kargs:
            self.opts['connect'] = kargs['connect']
        if 'pen' in kargs:
            self.setPen(kargs['pen'])
        if 'fillLevel' in kargs:
            self.setFillLevel(kargs['fillLevel'])
        if 'brush' in kargs:
            self.setBrush(kargs['brush'])
        if 'antialias' in kargs:
            self.opts['antialias'] = kargs['antialias']

        profiler('set')
        self.update()
        profiler('update')
        self.sigPlotChanged.emit(self)
        profiler('emit')

    def generatePath(self, x, y):
        if self.opts['stepMode']:
            # each value in the x/y arrays generates 2 points.
            x2 = np.empty((len(x),2), dtype=x.dtype)
            x2[:] = x[:,np.newaxis]
            if self.opts['fillLevel'] is None:
                x = x2.reshape(x2.size)[1:-1]
                y2 = np.empty((len(y),2), dtype=y.dtype)
                y2[:] = y[:,np.newaxis]
                y = y2.reshape(y2.size)
            else:
                # If we have a fill level, add two extra points at either end
                x = x2.reshape(x2.size)
                y2 = np.empty((len(y)+2,2), dtype=y.dtype)
                y2[1:-1] = y[:,np.newaxis]
                y = y2.reshape(y2.size)[1:-1]
                y[0] = self.opts['fillLevel']
                y[-1] = self.opts['fillLevel']

        path = fn.arrayToQPath(x, y, connect=self.opts['connect'])

        return path

    def getPath(self):
        if self.path is None:
            x,y = self.getData()
            if x is None or len(x) == 0 or y is None or len(y) == 0:
                self.path = QtGui.QPainterPath()
            else:
                self.path = self.generatePath(*self.getData())
            self.fillPath = None
            self._mouseShape = None

        return self.path

    def paint(self, p, opt, widget):
        profiler = debug.Profiler()
        if self.xData is None or len(self.xData) == 0:
            return

        x = None
        y = None
        path = self.getPath()

        profiler('generate path')

        if self._exportOpts is not False:
            aa = self._exportOpts.get('antialias', True)
        else:
            aa = self.opts['antialias']

        p.setRenderHint(p.Antialiasing, aa)

        if self.opts['brush'] is not None and self.opts['fillLevel'] is not None:
            if self.fillPath is None:
                if x is None:
                    x,y = self.getData()
                p2 = QtGui.QPainterPath(self.path)
                p2.lineTo(x[-1], self.opts['fillLevel'])
                p2.lineTo(x[0], self.opts['fillLevel'])
                p2.lineTo(x[0], y[0])
                p2.closeSubpath()
                self.fillPath = p2

            profiler('generate fill path')
            p.fillPath(self.fillPath, self.opts['brush'])
            profiler('draw fill path')

        sp = self.opts['shadowPen']
        if sp is not None and sp.style() != QtCore.Qt.NoPen:
            p.setPen(sp)
            p.drawPath(path)

        p.setPen(self.opts['pen'])
        p.drawPath(path)
        profiler('drawPath')

    def clear(self):
        self.xData = None  # raw values
        self.yData = None
        self.xDisp = None  # display values (after log / fft)
        self.yDisp = None
        self.path = None
        self.fillPath = None
        self._mouseShape = None
        self._mouseBounds = None
        self._boundsCache = [None, None]

    def mouseShape(self):
        """
        Return a QPainterPath representing the clickable shape of the curve

        """
        if self._mouseShape is None:
            view = self.getViewBox()
            if view is None:
                return QtGui.QPainterPath()
            stroker = QtGui.QPainterPathStroker()
            path = self.getPath()
            path = self.mapToItem(view, path)
            stroker.setWidth(self.opts['mouseWidth'])
            mousePath = stroker.createStroke(path)
            self._mouseShape = self.mapFromItem(view, mousePath)
        return self._mouseShape

    def mouseClickEvent(self, ev):
        if not self.clickable or ev.button() != QtCore.Qt.LeftButton:
            return
        if self.mouseShape().contains(ev.pos()):
            ev.accept()
            self.sigClicked.emit(self)


