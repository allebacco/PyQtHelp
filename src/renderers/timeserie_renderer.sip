%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip


class TimeserieRenderer: public QObject
{
%TypeHeaderCode
#include "timeserie_renderer.h"
%End

public:
    TimeserieRenderer(QObject* parent /TransferThis/ =nullptr) /HoldGIL/;
    virtual ~TimeserieRenderer() /HoldGIL/;

    void setData(const NDArray& x, const NDArray& y) throw(std::runtime_error) /HoldGIL/;

    void setPen(const QPen& newPen) /HoldGIL/;

    void render(QPainter* painter) /HoldGIL/;

    void invalidate();

};
