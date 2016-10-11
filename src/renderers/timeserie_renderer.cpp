#include "timeserie_renderer.h"

#include <QPainter>
#include <QStyleOptionGraphicsItem>
#include <QWidget>


TimeserieRenderer::TimeserieRenderer(QObject* parent) : QObject(parent)
{

}


TimeserieRenderer::~TimeserieRenderer()
{

}


void TimeserieRenderer::setData(const NDArray& x, const NDArray& y) noexcept(false)
{
    mPoints.clear();

    // Ensure x and y have the same shape
    if(x.ndims()!=1 || y.ndims()!=1 || x.shape(0)!=y.shape(0))
        throw std::runtime_error("x and y must be 1D arrays with the same size");

    // Convert data to double to avoid using multiple templates
    const NDArray xd = x.convertTo(NPY_DOUBLE);
    const NDArray yd = y.convertTo(NPY_DOUBLE);

    const size_t size = xd.shape(0);
    mPoints.reserve(size);

    const double* xData = xd.data<double>();
    const double* yData = yd.data<double>();
    for(size_t i=0; i<size; ++i)
        mPoints.push_back(QPointF(xData[i], yData[i]));

    invalidate();
}


void TimeserieRenderer::setPen(const QPen& newPen)
{
    mPen = newPen;
    invalidate();
}


void TimeserieRenderer::render(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{

}


void TimeserieRenderer::invalidate()
{

}