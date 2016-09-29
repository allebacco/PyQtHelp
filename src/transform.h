#ifndef TRANSFORM_H
#define TRANSFORM_H

#include <stdexcept>
#include <QtGui/QTransform>


QTransform invertTransform(const QTransform& tr, bool* invertible);


#endif  // TRANSFORM_H
