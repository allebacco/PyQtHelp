
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL  numpy_ARRAY_API
#ifdef NO_IMPORT_ARRAY:
    #undef NO_IMPORT_ARRAY
#endif
#include <numpy/arrayobject.h>

#include "ndarray.h"

#include <vector>


int NDArray::import_numpy()
{
    int ret = _import_array();
    return ret;
}

void NDArray::acquire(PyObject* ndarray)
{
    release(); // release the previous data
    
    if(ndarray==nullptr)
        return;

    if(!PyArray_Check(ndarray))
        throw std::runtime_error("Object is not Numpy Array");

    if(PyArray_IS_C_CONTIGUOUS((PyArrayObject*)ndarray)==0)
        throw std::runtime_error("Numpy array must be C contiguous");

    mNDims = PyArray_NDIM((PyArrayObject*)ndarray);
    mDtype = PyArray_TYPE((PyArrayObject*)ndarray);
    mData = PyArray_DATA((PyArrayObject*)ndarray);

    mNdArray = ndarray;
    Py_INCREF(mNdArray);
}

NDArray NDArray::convertTo(const int typenum) const
{
    if(mDtype==typenum)
        return *this;

    NDArray out = empty_like(*this, typenum);

    int ok = PyArray_CopyInto((PyArrayObject*)out.mNdArray, (PyArrayObject*)mNdArray);
    return out;
}

NDArray NDArray::empty_like(const NDArray& other, int typenum)
{
    if(typenum==NPY_VOID)
        typenum = other.dtype();

    std::vector<npy_intp> dims;
    for(size_t i=0; i<other.ndims(); ++i)
        dims.push_back(other.shape(i));

    PyObject* py_out = PyArray_EMPTY(static_cast<int>(other.ndims()), dims.data(), typenum, 0);
    NDArray out(py_out);
    // release a reference because the NDArray has already taken one
    Py_XDECREF(py_out);

    return out;
}


bool NDArray::is_valid_array(PyObject* ndarray)
{
    if(ndarray==nullptr)
        return true;

    if(ndarray==Py_None)
        return true;

    if(!PyArray_Check(ndarray))
        return false;

    if(PyArray_IS_C_CONTIGUOUS((PyArrayObject*)ndarray)==0)
        return false;

    return true;
}