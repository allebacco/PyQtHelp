#include "ndarray.h"

#include <vector>

NDArray NDArray::convertTo(const int typenum) const
{
    if(mDtype==typenum)
        return *this;

    NDArray out = empty_like(*this, typenum);
    
    int ok = PyArray_CopyInto(out.mNdArray, mNdArray);
    return out;
}

NDArray empty_like(const NDArray& other, int typenum)
{
    if(typenum==NPY_VOID)
        typenum = other.dtype();

    std::vector<npy_intp> dims;
    for(size_t i=0; i<other.ndims(); ++i)
        dims.push_back(other.shape(i));

    NDArray out(PyArray_EMPTY(other.ndims(), dims.data(), dtype, 0));
    // release a reference because the NDArray has already taken one
    PY_DECREF(out.mNdArray);

    return out;
}