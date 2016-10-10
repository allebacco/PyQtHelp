#ifndef NUMPY_H
#define NUMPY_H

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL  numpy_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <stdexcept>
#include <vector>

/*!
 * \brief C++ wrapper around the numpy.ndarray object
 */
class NDArray
{
public:
    /*!
     * \brief Construct a null array
     */
    NDArray():
        mDtype(NPY_VOID),
        mNdArray(nullptr),
        mNDims(0),
        mData(nullptr)
    {}

    /*!
     * \brief Wrap a numpy arary and optionally increment its reference counter
     * \param ndarray Numpy array object or None or nullptr for creating null arrays
     * \param acquireRef true for incrementing the reference count of the numpy.ndarray object
     */
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

    /*!
     * \brief Release the reference to the Python Object
     */
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

    /*!
     * \brief Release the reference to the Python Object
     */
    void release()
    {
        Py_CLEAR(mNdArray);
        mNdArray = nullptr;
        mData = nullptr;
        mNDims = 0;
    }

    /*!
     * \brief Number of dimensions
     * \returns The number of dimensions
     */
    size_t ndims() const
    {
        return mNDims;
    }

    /*!
     * \brief Shape of the array
     * \param i Index of the dimension
     * \returns The shape of teh array along the dimension i
     */
    size_t shape(const size_t i=0) const
    {
        if(i>=mNDims)
             throw std::runtime_error("Index error in ndarray.shape");
        return PyArray_DIM(reinterpret_cast<PyArrayObject*>(mNdArray), static_cast<int>(i));
    }

    /*!
     * \brief Data type of the array
     * \returns The data type of the array
     */
    int dtype() const
    {
        return mDtype;
    }

    /*!
     * \brief Pointer to the inner array memory
     * \tparam _Tp Type of teh data
     * \returns Pointer to the wrapped data
     */
    template<typename _Tp>
    _Tp* data()
    {
        return static_cast<_Tp*>(mData);
    }

    /*!
     * \brief Pointer to the inner array memory
     * \tparam _Tp Type of teh data
     * \returns Pointer to the wrapped data
     */
    template<typename _Tp>
    const _Tp* data() const
    {
        return static_cast<const _Tp*>(mData);
    }

    /*!
     * \brief Convert the wrapped array to a different type
     * If the data has already the requested type, a reference to the same array is returned
     * \param typenum Type to convert to
     * \returns NDArray of the requested type
     */
    NDArray convertTo(const int typenum) const;

    /*!
     * \brief Wrapped Python object
     * \returns The wrapped Python object
     */
    PyObject* handle() const { return mNdArray; }

public:

    /*!
     * \brief Create a new numpy array with the same shape (and dtype) of other
     * \param other Other array
     * \param typenum Requested dtype. NPY_VOID for using other dtype
     * \returns Array with the same shape of the input array and the requested type
     */
    static NDArray empty_like(const NDArray& other, int typenum=NPY_VOID);

    static NDArray empty(const size_t dim0, const size_t dim1, const int typenum);

    static NDArray empty(const size_t dim0, const size_t dim1, const size_t dim2, const int typenum=NPY_DOUBLE);

    static NDArray empty(const std::vector<npy_intp> dims, const int typenum=NPY_DOUBLE);

    /*!
     * \brief Check if input is a valid array
     * \returns false if the input argument can't construct a NDArray instance
     */
    static bool is_valid_array(PyObject* ndarray);

    static int import_numpy();

protected:

    void acquire(PyObject* ndarray);

protected:
    int mDtype;
    PyObject* mNdArray;
    size_t mNDims;
    void* mData;
};

#endif // NUMPY_H
