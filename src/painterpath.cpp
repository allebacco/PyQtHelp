#include "painterpath.h"
#include "geometry.h"


PathConnect decodeConnect(const QString& connect, const PathConnect defaultvalue)
{
    if(connect=="all")
        return PathConnect::All;
    if(connect=="finite")
        return PathConnect::Finite;
    if(connect=="pairs")
        return PathConnect::Pairs;

    return defaultvalue;
}



QPainterPath arrayToQPathOptimized(const double* x, const double* y, const size_t size, const uint8_t* connect,
                                   const QTransform& tr, const double lineWidth)
{
    QPainterPath path;
    if(size==0)
        return path;

    const double minDistance = std::max(1.0, lineWidth);
    QPointF lastPoint = tr.map(QPointF(x[0], y[0]));
    path.moveTo(lastPoint);
    bool forcepoint = false;
    const size_t count = size-1;
    for(size_t i=1; i<count; ++i)
    {
        const QPointF currPoint = tr.map(QPointF(x[i], y[i]));

        if(connect[i-1]==0)
        {
            path.moveTo(currPoint);
            forcepoint = true;
        }
        else if(forcepoint || sqeuclideanDistance(currPoint, lastPoint) >= minDistance)
        {
            path.lineTo(currPoint);
            lastPoint = currPoint;
            forcepoint = false;
        }
    }

    // Add the last point
    if(connect[count-1])
        path.lineTo(tr.map(QPointF(x[count], y[count])));

    return path;
}


QPainterPath arrayToQPathOptimized(const QPointF* xy, const size_t size, const uint8_t* connect,
                                   const QTransform& tr, const double lineWidth)
{
    QPainterPath path;
    if(size==0)
        return path;

    const double minDistance = std::max(1.0, lineWidth);
    QPointF lastPoint = tr.map(xy[0]);
    path.moveTo(lastPoint);
    bool forcepoint = false;
    const size_t count = size-1;
    for(size_t i=1; i<count; ++i)
    {
        const QPointF currPoint = tr.map(xy[i]);

        if(connect[i-1]==0)
        {
            path.moveTo(currPoint);
            forcepoint = true;
        }
        else if(forcepoint || sqeuclideanDistance(currPoint, lastPoint) >= minDistance)
        {
            path.lineTo(currPoint);
            lastPoint = currPoint;
            forcepoint = false;
        }
    }

    // Add the last point
    if(connect[count-1])
        path.lineTo(tr.map(xy[count]));

    return path;
}