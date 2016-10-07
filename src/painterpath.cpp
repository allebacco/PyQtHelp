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
    QPointF lastPoint(x[0], y[0]);
    lastPoint = tr.map(lastPoint);
    path.moveTo(lastPoint);
    bool forcepoint = false;
    const size_t count = size-1;
    for(size_t i=1; i<count; ++i)
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

    // Add the last point
    if(connect[count-1])
        path.lineTo(tr.map(QPointF(x[count], y[count])));

    return path;
}