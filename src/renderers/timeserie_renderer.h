#ifndef TIMESERIE_RENDERER_H
#define TIMESERIE_RENDERER_H

#include <vector>
#include <stdexcept>
#include <QObject>
#include <QPointF>
#include <QPen>

#include "../numpy_wrap/ndarray.h"

class QPainter;
class QStyleOptionGraphicsItem;
class QWidget;


class TimeserieRenderer: public QObject
{
    Q_OBJECT

public:
    TimeserieRenderer(QObject* parent=nullptr);
    virtual ~TimeserieRenderer();

    void setData(const NDArray& x, const NDArray& y) noexcept(false);

    void setPen(const QPen& newPen);

    void render(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget=nullptr);


protected:

    void invalidate();

protected:

    std::vector<QPointF> mPoints;
    QPen mPen;

};


#endif  // TIMESERIE_RENDERER_H