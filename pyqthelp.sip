%Module(name=native, keyword_arguments="Optional")

%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip


%ModuleCode

#include "numpy_wrap/ndarray.h"

%End

%PostInitialisationCode

NDArray::import_numpy();

%End



%Include src/exceptions.sip
%Include src/numpy_wrap/ndarray.sip
%Include src/transform.sip
%Include src/painterpath.sip
%Include src/renderers/timeserie_renderer.sip
