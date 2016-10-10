#ifndef PAINTERPATH_H
#define PAINTERPATH_H

#include <QPainterPath>
#include <QTransform>
#include <QString>

#include "geometry.h"


#if defined _MSC_VER
#pragma message("Compiling fpclassify() workaround")
// Workaround for the absence of isfinite for integer types in MSVC
static int fpclassify(int32_t v) { return FP_NORMAL; }
static int fpclassify(uint32_t v) { return FP_NORMAL; }
static int fpclassify(int64_t v) { return FP_NORMAL; }
static int fpclassify(uint64_t v) { return FP_NORMAL; }
#endif



enum class PathConnect
{
    All,    ///< Connect all points
    Finite, ///< Connect only finite points
    Pairs   ///< Connect pairs of points
};

PathConnect decodeConnect(const QString& connect, const PathConnect defaultvalue=PathConnect::All);


template<typename _TpX, typename _TpY>
static void arrayToQPathAll(const _TpX* x, const _TpY* y, const size_t size, QPainterPath& path)
{
    path.moveTo(x[0], y[0]);
    for(size_t i=1; i<size; ++i)
        path.lineTo(x[i], y[i]);
}


/*!
 * \brief Build a path for the input data by connecting only finite points
 * When a not finite point is found, the line is trucated
 * \tparam _TpX Type of the input data
 * \param x Input x data
 * \param y Input y data
 * \param size number of elements of the input data
 * \returns The path for drawing the input data
 */
template<typename _TpX, typename _TpY>
static void arrayToQPathPairs(const _TpX* x, const _TpY* y, const size_t size, QPainterPath& path)
{
    // At least 2 points must be present
    if(size<2)
        return;

    for(size_t i=0; i<size; i+=2)
    {
        path.moveTo(x[i], y[i]);
        path.lineTo(x[i+1], y[i+1]);
    }
}


/*!
 * \brief Build a path for the input data by connecting only finite points
 * When a not finite point is found, the line is trucated
 * \tparam _TpX Type of the input data
 * \param x Input x data
 * \param y Input y data
 * \param size number of elements of the input data
 * \param[out] The path for drawing the input data
 */
template<typename _TpX, typename _TpY>
static void arrayToQPathFinite(const _TpX* x, const _TpY* y, const size_t size, QPainterPath& path)
{
    bool skip = true;
    for(size_t i=0; i<size; ++i)
    {
        if(std::isfinite(x[i]) && std::isfinite(y[i]))
        {
            if(skip)
                path.moveTo(x[i], y[i]);
            else
                path.lineTo(x[i], y[i]);
            skip = false;
        }
        else
            skip = true;
    }
}


/*!
 * \brief Build a path for the input data
 * \tparam _TpX Type of the input data
 * \param x Input x data
 * \param y Input y data
 * \param size number of elements of the input data
 * \param connect Connection between the input points
 * \returns The path for drawing the input data
 */
template<typename _TpX, typename _TpY>
static QPainterPath arrayToQPath(const _TpX* x, const _TpY* y, const size_t size,
                                 const PathConnect connect=PathConnect::All)
{
    QPainterPath path;
    if(size>0)
    {
        switch(connect)
        {
            case PathConnect::All:
                arrayToQPathAll(x, y, size, path);
                break;
            case PathConnect::Finite:
                arrayToQPathFinite(x, y, size, path);
                break;
            case PathConnect::Pairs:
                arrayToQPathPairs(x, y, size, path);
                break;
        }
    }
    return path;
}


/*!
 * \brief Build a path for the input data
 * \tparam _TpX Type of the input data
 * \param x Input x data
 * \param y Input y data
 * \param size number of elements of the input data
 * \param connect Array of connections between input points
 * \returns The path for drawing the input data
 */
template<typename _TpX, typename _TpY>
static QPainterPath arrayToQPath(const _TpX* x, const _TpY* y, const size_t size, const uint8_t* connect)
{
    QPainterPath path;
    if(size>0)
    {
        path.moveTo(x[0], y[0]);
        for(size_t i=1; i<size; ++i)
        {
            if(connect[i-1]!=0)
                path.lineTo(x[i], y[i]);
            else
                path.moveTo(x[i], y[i]);
        }
    }
    return path;
}


/*!
 * \brief Build an optimized path for the input data
 * \param x Input x data
 * \param y Input y data
 * \param size number of elements of the input data
 * \param connect Array of connections between input points
 * \param tr Transform that should be applied to the input data
 * \param lineWidth Width of teh line that will be used to draw the path
 * \returns The optimized path for drawing the input data
 */
QPainterPath arrayToQPathOptimized(const double* x, const double* y, const size_t size, const uint8_t* connect,
                                   const QTransform& tr, const double lineWidth=1.);


#endif  // PAINTERPATH_H