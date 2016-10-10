#ifndef TRANSFORM_H
#define TRANSFORM_H

#include <stdexcept>
#include <QTransform>


/*!
 * \brief Invert a transform without precision loss
 * \param tr Transform to invert
 * \param[out] invertible Output flag that notify if teh matrix is invertible or not
 * \returns Inverted transform or identity of the tr is not invertible
 */
QTransform invertTransform(const QTransform& tr, bool* invertible=nullptr);


#endif  // TRANSFORM_H
