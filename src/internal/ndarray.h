#ifndef NUMPY_H
#define NUMPY_H

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <stdexcept>

class NDArray
{
public:
    NDArray():
        mDtype(NPY_VOID),
        mNdArray(nullptr),
        mNDims(0),
        mData(nullptr)
    {}

    NDArray(PyObject* ndarray) throw(std::runtime_error):
        mDtype(NPY_VOID),
        mNdArray(nullptr),
        mNDims(0),
        mData(nullptr)
    {
        acquire(ndarray);
    }

    ~NDArray()
    {
        release();
    }

    void operator=(PyObject* ndarray) throw(std::runtime_error)
    {
        acquire(ndarray);
    }

    void release()
    {
        Py_CLEAR(mNdArray);
        mNdArray = nullptr;
        mData = nullptr;
        mNDims = 0;
    }

    size_t ndims() const
    {
        return mNDims;
    }

    size_t shape(const size_t i=0) const throw(std::runtime_error)
    {
        if(i>=mNDims)
             throw std::runtime_error("Index error in ndarray.shape");
        return PyArray_DIM((PyArrayObject*)mNdArray, i);
    }

    int dtype() const
    {
        return mDtype;
    }

    template<typename _Tp>
    _Tp* data()
    {
        return static_cast<_Tp*>(mData);
    }

    template<typename _Tp>
    const _Tp* data() const
    {
        return static_cast<const _Tp*>(mData);
    }

protected:

    void acquire(PyObject* ndarray) throw(std::runtime_error)
    {
        if(!PyArray_Check(ndarray))
            throw std::runtime_error("Object is not Numpy Array");

        release(); // release the previous data

        if(PyArray_IS_C_CONTIGUOUS((PyArrayObject*)ndarray)==0)
            throw std::runtime_error("Numpy array must be contiguous");

        mNDims = PyArray_NDIM((PyArrayObject*)ndarray);
        mDtype = PyArray_TYPE((PyArrayObject*)ndarray);
        mData = PyArray_DATA((PyArrayObject*)ndarray);

        mNdArray = ndarray;
        Py_INCREF(mNdArray);
    }


protected:
    int mDtype;
    PyObject* mNdArray;
    size_t mNDims;
    void* mData;
};

#endif // NUMPY_H
