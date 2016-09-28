
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL  numpy_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include "ndarray.h"

#include <vector>

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

    PyObject* py_out = PyArray_EMPTY(other.ndims(), dims.data(), typenum, 0);
    NDArray out(py_out);
    // release a reference because the NDArray has already taken one
    Py_XDECREF(py_out);

    return out;
}