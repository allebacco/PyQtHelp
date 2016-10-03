#ifndef TRANSFORM_H
#define TRANSFORM_H

#include <stdexcept>
#include <QTransform>


QTransform invertTransform(const QTransform& tr, bool* invertible=nullptr);


#endif  // TRANSFORM_H
