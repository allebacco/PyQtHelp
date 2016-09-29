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

    NDArray(PyObject* ndarray, const bool acquireRef=true) :
        mDtype(NPY_VOID),
        mNdArray(nullptr),
        mNDims(0),
        mData(nullptr)
    {
        acquire(ndarray);
    }

    NDArray(const NDArray& other):
        mDtype(other.mDtype),
        mNdArray(other.mNdArray),
        mNDims(other.mNDims),
        mData(other.mData)
    {
        Py_XINCREF(mNdArray);
    }

    NDArray(NDArray&& other):
        mDtype(other.mDtype),
        mNdArray(other.mNdArray),
        mNDims(other.mNDims),
        mData(other.mData)
    {
        mDtype = other.mDtype;
        mNdArray = other.mNdArray;
        mNDims = other.mNDims;
        mData = other.mData;

        other.mDtype = NPY_VOID;
        other.mNdArray = nullptr;
        other.mNDims = 0;
        other.mData = nullptr;
    }

    ~NDArray()
    {
        release();
    }

    void operator=(PyObject* ndarray)
    {
        acquire(ndarray);
    }

    void operator=(const NDArray& other)
    {
        acquire(other.mNdArray);
    }

    void operator=(NDArray&& other)
    {
        mDtype = other.mDtype;
        mNdArray = other.mNdArray;
        mNDims = other.mNDims;
        mData = other.mData;

        other.mDtype = NPY_VOID;
        other.mNdArray = nullptr;
        other.mNDims = 0;
        other.mData = nullptr;
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

    size_t shape(const size_t i=0) const
    {
        if(i>=mNDims)
             throw std::runtime_error("Index error in ndarray.shape");
        return PyArray_DIM(reinterpret_cast<PyArrayObject*>(mNdArray), static_cast<int>(i));
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

    NDArray convertTo(const int typenum) const;

    static NDArray empty_like(const NDArray& other, int typenum=NPY_VOID); 

protected:

    void acquire(PyObject* ndarray);

protected:
    int mDtype;
    PyObject* mNdArray;
    size_t mNDims;
    void* mData;
};

#endif // NUMPY_H
