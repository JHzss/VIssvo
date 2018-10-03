//
// Created by jh on 18-9-4.
//

#ifndef SSVO_POSEPVR_H
#define SSVO_POSEPVR_H

#include "global.hpp"
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "so3.h"
#include "frame.hpp"

namespace ssvo
{

    class PosePVR : public ceres::LocalParameterization
    {
        //todo 设置成0
        virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;

        virtual bool ComputeJacobian(const double *x, double *jacobian) const;

        virtual int GlobalSize() const { return 9; };
        virtual int LocalSize() const { return 9; };


    };




}




#endif //SSVO_POSEPVR_H
