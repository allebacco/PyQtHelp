
%MappedType NDArray
{
%TypeHeaderCode

#include "ndarray.h"

%End

%ConvertFromTypeCode
    PyObject* ret = sipCpp->handle();

    if(ret==nullptr)
        ret = Py_None;

    Py_INCREF(ret);

    return ret;
%End


%ConvertToTypeCode
    // See if we are just being asked to check the type of the Python
    // object.
    if (!sipIsErr)
    {
        if(NDArray::is_valid_array(sipPy))
            return 1;
        // The type is not valid.
        return 0;
    }

    PyObject* obj = sipPy;

    if(sipPy==Py_None)
        *sipCppPtr = new NDArray();
    else
        *sipCppPtr = new NDArray(sipPy);

    // The type is valid.
    return sipGetState(sipTransferObj);
%End
};
