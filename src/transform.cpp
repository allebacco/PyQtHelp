#include "transform.h"


QTransform inverted(const QTransform& tr, bool* invertible=nullptr)
{
    if(invertible!=nullptr)
        *invertible = false;

    const double in_m[3][3] { {tr.m11(), tr.m12(), tr.m13()},
                              {tr.m21(), tr.m22(), tr.m23()},
                              {tr.m31(), tr.m32(), tr.m33()} };

    const double det = in_m[0][0] * (in_m[1][1] * in_m[2][2] - in_m[2][1] * in_m[1][2]) -
                       in_m[0][1] * (in_m[1][0] * in_m[2][2] - in_m[1][2] * in_m[2][0]) +
                       in_m[0][2] * (in_m[1][0] * in_m[2][1] - in_m[1][1] * in_m[2][0]);

    if(det==0.0)
        return QTransform();

    if(invertible!=nullptr)
        *invertible = true;

    const double invdet = 1 / det;

    double out_m[3][3]; // inverse of matrix m
    out_m[0][0] = (in_m[1][1] * in_m[2][2] - in_m[2][1] * in_m[1][2]) * invdet;
    out_m[0][1] = (in_m[0][2] * in_m[2][1] - in_m[0][1] * in_m[2][2]) * invdet;
    out_m[0][2] = (in_m[0][1] * in_m[1][2] - in_m[0][2] * in_m[1][1]) * invdet;
    out_m[1][0] = (in_m[1][2] * in_m[2][0] - in_m[1][0] * in_m[2][2]) * invdet;
    out_m[1][1] = (in_m[0][0] * in_m[2][2] - in_m[0][2] * in_m[2][0]) * invdet;
    out_m[1][2] = (in_m[1][0] * in_m[0][2] - in_m[0][0] * in_m[1][2]) * invdet;
    out_m[2][0] = (in_m[1][0] * in_m[2][1] - in_m[2][0] * in_m[1][1]) * invdet;
    out_m[2][1] = (in_m[2][0] * in_m[0][1] - in_m[0][0] * in_m[2][1]) * invdet;
    out_m[2][2] = (in_m[0][0] * in_m[1][1] - in_m[1][0] * in_m[0][1]) * invdet;

    return QTransform(out_m[0][0], out_m[0][1], out_m[0][2],
                      out_m[1][0], out_m[1][1], out_m[1][2],
                      out_m[2][0], out_m[2][1], out_m[2][2]);
}