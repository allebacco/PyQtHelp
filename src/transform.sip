%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip

%ModuleCode
#include <stdexcept>
#include "transform.h"
#include "numpy_wrap/ndarray.h"

template <typename _Tp>
void arrayToTransformImpl(const _Tp arrayData, QTransform& outTr)
{
    outTr.setMatrix(arrayData[0], arrayData[1], arrayData[2],
                    arrayData[3], arrayData[4], arrayData[5],
                    arrayData[6], arrayData[7], arrayData[8]);
}


%End

QTransform invertTransform(const QTransform& tr, bool *invertible=nullptr) throw(std::runtime_error);


QTransform arrayToTransform(const NDArray& array);
%MethodCode
    try
    {
        const NDArray* x = a0;

        if(x->ndims()!=2 || x->shape(0)!=3  || x->shape(1)!=3)
            throw std::runtime_error("Input array must be a 3x3 matrix");

        sipRes = new QTransform();

        switch (x->dtype()) {
            case NPY_FLOAT64:
                arrayToTransformImpl(x->data<double>(), *sipRes);
                break;
            case NPY_FLOAT32:
                arrayToTransformImpl(x->data<float>(), *sipRes);
                break;
            case NPY_INT32:
                arrayToTransformImpl(x->data<int32_t>(), *sipRes); 
                break;
            case NPY_UINT32:
                arrayToTransformImpl(x->data<uint32_t>(), *sipRes);   
                break;
            case NPY_INT64:
                arrayToTransformImpl(x->data<int64_t>(), *sipRes); 
                break;
            case NPY_UINT64:
                arrayToTransformImpl(x->data<uint64_t>(), *sipRes);   
                break;
            default:
                throw std::runtime_error("x has unsupported dtype");
        }

    } catch(const std::runtime_error& err) {
        PyErr_SetString(PyExc_RuntimeError, err.what());
        if(sipRes!=nullptr)
            delete sipRes;
        sipRes = nullptr;
    }

%End



NDArray transformToArray(const QTransform& transform);
%MethodCode
    try
    {
        const QTransform* transform = a0;

        sipRes = new NDArray(NDArray::empty(3, 3, NPY_DOUBLE));

        double* data = sipRes->data<double>();
        data[0] = transform->m11();
        data[1] = transform->m12();
        data[2] = transform->m13();
        data[3] = transform->m21();
        data[4] = transform->m22();
        data[5] = transform->m23();
        data[6] = transform->m31();
        data[7] = transform->m32();
        data[8] = transform->m33();

    } catch(const std::runtime_error& err) {
        PyErr_SetString(PyExc_RuntimeError, err.what());
        sipRes = nullptr;
    }

%End

