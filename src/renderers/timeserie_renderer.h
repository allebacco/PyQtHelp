#ifndef TIMESERIE_RENDERER_H
#define TIMESERIE_RENDERER_H

#include <vector>
#include <stdexcept>
#include <QObject>
#include <QPointF>
#include <QPen>
#include <QPainter>

#include "../numpy_wrap/ndarray.h"


class TimeserieRenderer: public QObject
{
    Q_OBJECT

public:
    TimeserieRenderer(QObject* parent=nullptr);
    virtual ~TimeserieRenderer();

    void setData(const NDArray& x, const NDArray& y) noexcept(false);

    void setPen(const QPen& newPen);

    void render(QPainter* painter);

    void invalidate();

protected:

    std::vector<QPointF> mPoints;
    std::vector<uint8_t> mConnect;
    QPen mPen;
    QPainterPath mPath;

};


#endif  // TIMESERIE_RENDERER_H