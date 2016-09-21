%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip

%ModuleCode
#include "transform.h"
#include <stdexcept>
%End

QTransform inverted(const QTransform& tr, bool *invertible=nullptr) throw(std::runtime_error);
