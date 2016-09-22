%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip

%ModuleCode
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL numpy_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include "painterpath.h"
#include "internal/Numpy.h"
%End

/*
QPainterPath arrayToQPath(PyObject* x, PyObject* y, PyObject* connect=Py_None);
%MethodCode
    PyObject* pyX = a0;
    PyObject* pyY = a1;
    PyObject* pyConnect = a2;

    try{

        NDArray x(pyX);
        NDArray y(pyY);
        QString connect;
        if(PyString_Check(pyConnect))
            connect = PyString_AsString(pyConnect);

        //if(x.dtype()!=NPY_FLOAT64 || y.dtype()!=NPY_FLOAT64)
        //    throw std::runtime_error("x and y has not dtype float64");

        if(x.ndims()!=1 || y.ndims()!=1)
            throw std::runtime_error("x and y has ndims != 1");

        if(x.shape(0)!=y.shape(0))
            throw std::runtime_error("x and y different shape");

        if(x.dtype()==NPY_INT64 && y.dtype()==NPY_FLOAT64)
        {
            sipRes = new QPainterPath(arrayToQPath(x.data<int64_t>(), y.data<double>(), x.shape(0), connect));
        }
        else
        {
            if(x.dtype()!=y.dtype())
                throw std::runtime_error("x and y has not same dtype");

            switch (x.dtype()) {
                case NPY_FLOAT64:
                    sipRes = new QPainterPath(arrayToQPath(x.data<double>(), y.data<double>(), x.shape(0), connect));
                    break;
                case NPY_FLOAT32:
                    sipRes = new QPainterPath(arrayToQPath(x.data<float>(), y.data<float>(), x.shape(0), connect));
                    break;
                case NPY_INT8:
                    sipRes = new QPainterPath(arrayToQPath(x.data<int8_t>(), y.data<int8_t>(), x.shape(0), connect));
                    break;
                case NPY_UINT8:
                    sipRes = new QPainterPath(arrayToQPath(x.data<uint8_t>(), y.data<uint8_t>(), x.shape(0), connect));
                    break;
                case NPY_INT16:
                    sipRes = new QPainterPath(arrayToQPath(x.data<int16_t>(), y.data<int16_t>(), x.shape(0), connect));
                    break;
                case NPY_UINT16:
                    sipRes = new QPainterPath(arrayToQPath(x.data<uint16_t>(), y.data<uint16_t>(), x.shape(0), connect));
                    break;
                case NPY_INT32:
                    sipRes = new QPainterPath(arrayToQPath(x.data<int32_t>(), y.data<int32_t>(), x.shape(0), connect));
                    break;
                case NPY_UINT32:
                    sipRes = new QPainterPath(arrayToQPath(x.data<uint32_t>(), y.data<uint32_t>(), x.shape(0), connect));
                    break;
                case NPY_INT64:
                    sipRes = new QPainterPath(arrayToQPath(x.data<int64_t>(), y.data<int64_t>(), x.shape(0), connect));
                    break;
                case NPY_UINT64:
                    sipRes = new QPainterPath(arrayToQPath(x.data<uint64_t>(), y.data<uint64_t>(), x.shape(0), connect));
                    break;
                default:
                    throw std::runtime_error("x and y have unsupported dtype");
            }
        }
        //sipRes = new QPainterPath(arrayToQPath(x.data<double>(), y.data<double>(), x.shape(0), connect));

    } catch(const std::runtime_error& err) {
        PyErr_SetString(PyExc_RuntimeError, err.what());
        return nullptr;
    }

%End
*/