#ifndef PAINTERPATH_H
#define PAINTERPATH_H

#include <cmath>
#include <QPainterPath>


#if defined _MSC_VER
#pragma message("Compiling isfinite() workaround")
// Workaround for the absence of isfinite for integer types in MSVC
int fpclassify(int32_t v) { return FP_NORMAL; }
int fpclassify(uint32_t v) { return FP_NORMAL; }
int fpclassify(int64_t v) { return FP_NORMAL; }
int fpclassify(uint64_t v) { return FP_NORMAL; }
#endif



enum class PathConnect
{
    All,
    Finite,
    Pairs
};


template<typename _TpX, typename _TpY>
static void arrayToQPathAll(const _TpX* x, const _TpY* y, const size_t size, QPainterPath& path)
{
    path.moveTo(x[0], y[0]);
    for(size_t i=1; i<size; ++i)
        path.lineTo(x[i], y[i]);
}


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


template<typename _TpX, typename _TpY>
static void arrayToQPathFinite(const _TpX* x, const _TpY* y, const size_t size, QPainterPath& path)
{
    //using std::isfinite;

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


template<typename _TpX, typename _TpY>
static QPainterPath arrayToQPath(const _TpX* x, const _TpY* y, const size_t size, const uint8_t* connect)
{
    QPainterPath path;
    if(size>0)
    {
        path.moveTo(x[0], y[0]);
        for(size_t i=1; i<size; ++i)
        {
            if(connect[i]!=0)
                path.lineTo(x[i], y[i]);
            else
                path.moveTo(x[i], y[i]);
        }
    }
    return path;
}


#endif  // PAINTERPATH_H