%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip

%ModuleCode
#include "transform.h"
%End

QTransform inverted(const QTransform& tr, bool *invertible=nullptr);