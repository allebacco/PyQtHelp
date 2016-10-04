#include "painterpath.h"
#include "geometry.h"

QPainterPath arrayToQPathOptimized(const double* x, const double* y, const size_t size, const uint8_t* connect,
                                   const QTransform& tr, const double lineWidth)
{
    QPainterPath path;
    if(size==0)
        return path;

    const double minDistance = std::max(1.0, lineWidth);
    QPointF lastPoint(x[0], y[0]);
    lastPoint = tr.map(lastPoint);
    path.moveTo(lastPoint);
    bool forcepoint = true;
    for(size_t i=1; i<size; ++i)
    {
        QPointF currPoint(x[i], y[i]);
        currPoint = tr.map(currPoint);

        if(connect[i-1]==0)
        {
            path.moveTo(currPoint);
            forcepoint = true;
            continue;
        }
        
        if(forcepoint)
        {
            path.lineTo(currPoint);
            lastPoint = currPoint;
            forcepoint = false;
            continue;
        }

        if(sqeuclideanDistance(currPoint, lastPoint) >= minDistance)
        {
            path.lineTo(currPoint);
            lastPoint = currPoint;
        }
    }

    return path;
}