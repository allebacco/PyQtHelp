%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip

%ModuleCode
#include <algorithm>
#include <string>

#include "painterpath.h"
#include "numpy_wrap/ndarray.h"


PathConnect decodeConnect(PyObject* pyConnect, const PathConnect defaultvalue)
{
    std::string connect = PyUnicode_AsUTF8(pyConnect);
    std::transform(connect.begin(), connect.end(), connect.begin(), ::tolower);
    if(connect=="all")
        return PathConnect::All;
    if(connect=="finite")
        return PathConnect::Finite;
    if(connect=="pairs")
        return PathConnect::Pairs;

    return defaultvalue;
}


QPainterPath arrayToQPathImpl(const NDArray& x, const NDArray& y, const PathConnect connect)
{
    if(x.dtype()==NPY_INT64 && y.dtype()==NPY_FLOAT64)
        return arrayToQPath(x.data<int64_t>(), y.data<double>(), x.shape(0), connect);

    if(x.dtype()!=y.dtype())
        throw std::runtime_error("x and y have not the same dtype");

    switch (x.dtype()) {
        case NPY_FLOAT64:
            return arrayToQPath(x.data<double>(), y.data<double>(), x.shape(0), connect);
            break;
        case NPY_FLOAT32:
            return arrayToQPath(x.data<float>(), y.data<float>(), x.shape(0), connect);
            break;
        case NPY_INT32:
            return arrayToQPath(x.data<int32_t>(), y.data<int32_t>(), x.shape(0), connect);
            break;
        case NPY_UINT32:
            return arrayToQPath(x.data<uint32_t>(), y.data<uint32_t>(), x.shape(0), connect);
            break;
        case NPY_INT64:
            return arrayToQPath(x.data<int64_t>(), y.data<int64_t>(), x.shape(0), connect);
            break;
        case NPY_UINT64:
            return arrayToQPath(x.data<uint64_t>(), y.data<uint64_t>(), x.shape(0), connect);
            break;
        default:
            throw std::runtime_error("x and y have unsupported dtype");
    }
}


QPainterPath arrayToQPathImpl(const NDArray& x, const NDArray& y, const NDArray& connect)
{
    if(connect.dtype()!=NPY_UINT8 && connect.dtype()!=NPY_INT8 &&
       connect.dtype()!=NPY_BOOL)
        throw std::runtime_error("connect must be a bool array");

    if(connect.ndims()!=1 || connect.shape(0)!=x.shape(0)-1)
        throw std::runtime_error("connect must be a monodimensional array of size x.size-1");

    if(x.dtype()==NPY_INT64 && y.dtype()==NPY_FLOAT64)
        return arrayToQPath(x.data<int64_t>(), y.data<double>(), x.shape(0), connect.data<uint8_t>());

    if(x.dtype()!=y.dtype())
        throw std::runtime_error("x and y have not the same dtype");

    switch (x.dtype()) {
        case NPY_FLOAT64:
            return arrayToQPath(x.data<double>(), y.data<double>(), x.shape(0), connect.data<uint8_t>());
            break;
        case NPY_FLOAT32:
            return arrayToQPath(x.data<float>(), y.data<float>(), x.shape(0), connect.data<uint8_t>());
            break;
        case NPY_INT32:
            return arrayToQPath(x.data<int32_t>(), y.data<int32_t>(), x.shape(0), connect.data<uint8_t>());
            break;
        case NPY_UINT32:
            return arrayToQPath(x.data<uint32_t>(), y.data<uint32_t>(), x.shape(0), connect.data<uint8_t>());
            break;
        case NPY_INT64:
            return arrayToQPath(x.data<int64_t>(), y.data<int64_t>(), x.shape(0), connect.data<uint8_t>());
            break;
        case NPY_UINT64:
            return arrayToQPath(x.data<uint64_t>(), y.data<uint64_t>(), x.shape(0), connect.data<uint8_t>());
            break;
        default:
            throw std::runtime_error("x and y have unsupported dtype");
    }
}


%End


QPainterPath arrayToQPath(PyObject* x, PyObject* y, PyObject* connect=Py_None);
%MethodCode
    PyObject* pyX = a0;
    PyObject* pyY = a1;
    PyObject* pyConnect = a2;

    try{

        NDArray x(pyX);
        NDArray y(pyY);

        if(x.ndims()!=1 || y.ndims()!=1)
            throw std::runtime_error("x and y must be monodimensional");

        if(x.shape(0)!=y.shape(0))
            throw std::runtime_error("x and y must have the same size");

        if(PyUnicode_Check(pyConnect))
        {
            const PathConnect connect = decodeConnect(pyConnect, PathConnect::All);
            sipRes = new QPainterPath(arrayToQPathImpl(x, y, connect));
        }
        else
        {
            NDArray connectArray(pyConnect);
            sipRes = new QPainterPath(arrayToQPathImpl(x, y, connectArray));
        }

    } catch(const std::runtime_error& err) {
        PyErr_SetString(PyExc_RuntimeError, err.what());
        sipRes =  nullptr;
    }

%End



QPainterPath arrayToQPathOptimized(const NDArray& x, const NDArray& y, const NDArray& connect, const QTransform& tr, const double lineWidth=1.);
%MethodCode
    try {
        const NDArray* x = a0;
        const NDArray* y = a1;
        const NDArray* connect = a2;
        const QTransform* tr = a3;
        const double lineWidth = a4;

        if(x->ndims()!=1 || y->ndims()!=1)
            throw std::runtime_error("x and y must be monodimensional");

        if(x->shape(0)!=y->shape(0))
            throw std::runtime_error("x and y must have the same size");

        if(connect->ndims()!=1 || connect->shape(0)!=x->shape(0)-1)
            throw std::runtime_error("connect must be a monodimensional array of size x.size-1");

        if(connect->shape(0)!=x->shape(0)-1 && connect->shape(0)!=x->shape(0))
        throw std::runtime_error("connect must be a monodimensional array of size x.size-1 or x.size");

        if(x->dtype()!=NPY_FLOAT64 || y->dtype()!=NPY_FLOAT64)
            throw std::runtime_error("x and y must be float64 arrays");
            
        if(connect->dtype()!=NPY_UINT8 && connect->dtype()!=NPY_INT8 &&
           connect->dtype()!=NPY_BOOL)
            throw std::runtime_error("connect must be a bool array");

        const size_t size = x->shape(0);

        sipRes = new QPainterPath(arrayToQPathOptimized(x->data<double>(), y->data<double>(), size, connect->data<uint8_t>(), *tr, lineWidth));

    } catch(const std::runtime_error& err) {
        PyErr_SetString(PyExc_RuntimeError, err.what());
        sipRes = nullptr;
    }
%End

