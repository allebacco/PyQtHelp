%Module native

%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip


%ModuleCode
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL numpy_ARRAY_API
#include <numpy/arrayobject.h>

void init_package()
{
    int ret = _import_array();
}

%End

%PostInitialisationCode
init_package();
%End



%Include src/exceptions.sip
%Include src/transform.sip
%Include src/painterpath.sip
