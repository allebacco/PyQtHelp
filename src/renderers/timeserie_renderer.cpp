#include "timeserie_renderer.h"

#include <algorithm>
#include <QTransform>

#include "../painterpath.h"


TimeserieRenderer::TimeserieRenderer(QObject* parent) : QObject(parent)
{
    mPath = QPainterPath();
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

    mConnect.resize(size, 1);
    std::fill(mConnect.begin(), mConnect.end(), 1);

    invalidate();
}


void TimeserieRenderer::setPen(const QPen& newPen)
{
    mPen = newPen;
    invalidate();
}


void TimeserieRenderer::render(QPainter* painter)
{
    // Extract painter transform
    const QTransform transform = painter->combinedTransform();

    // Save transformation and disable it before painting to speedup
    const bool isViewTransformEnabled = painter->viewTransformEnabled();
    const bool isWorldTransformEnabled = painter->worldMatrixEnabled();
    painter->setViewTransformEnabled(false);
    painter->setWorldMatrixEnabled(false);

    const size_t size = mPoints.size();
    if(mPath.isEmpty() && size>0)
    {
        mPath = arrayToQPathOptimized(mPoints.data(), size, mConnect.data(), transform, mPen.width());
    }

    painter->setPen(mPen);
    painter->drawPath(mPath);

    // Restore painter transform
    painter->setViewTransformEnabled(isViewTransformEnabled);
    painter->setWorldMatrixEnabled(isWorldTransformEnabled);
}


void TimeserieRenderer::invalidate()
{
    mPath = QPainterPath();
}