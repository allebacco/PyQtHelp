#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <cmath>
#include <QPointF>


/*!
 * Basically, we are checking that the slopes between point 1 and point 2 and point 1
 * and point 3 match. Slope is change in y divided by change in x, so we have:
 * 
 * y1 - y2     y1 - y3
 * -------  =  --------
 * x1 - x2     x1 - x3
 *
 */
static bool collinear(const double x1, const double y1,
                      const double x2, const double y2,
                      const double x3, const double y3,
                      const double maxError=1e-9)
{
    return std::abs((y1 - y2) * (x1 - x3) - (y1 - y3) * (x1 - x2)) <= maxError;
}


static bool collinear(const QPointF& p1, const QPointF& p2, const QPointF& p3, const double maxError=1e-9)
{
    return std::abs((p1.y() - p2.y()) * (p1.x() - p3.x()) - (p1.y() - p3.y()) * (p1.x() - p2.x())) <= maxError;
}


static double euclideanDistance(const double x1, const double y1, const double x2, const double y2)
{
    const double dx = x2 - x1;
    const double dy = y2 - y1;
    return std::sqrt(dx*dx + dy*dy);
}


static double euclideanDistance(const QPointF& p1, const QPointF& p2)
{
    const QPointF dxy(p1 - p2);
    return std::sqrt(dxy.x()*dxy.x() + dxy.y()*dxy.y());
}


static double sqeuclideanDistance(const double x1, const double y1, const double x2, const double y2)
{
    const double dx = x2 - x1;
    const double dy = y2 - y1;
    return std::abs(dx*dx + dy*dy);
}


static double sqeuclideanDistance(const QPointF& p1, const QPointF& p2)
{
    const QPointF dxy(p1 - p2);
    return std::abs(dxy.x()*dxy.x() + dxy.y()*dxy.y());
}

#endif  // GEOMETRY_H