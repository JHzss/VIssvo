//
// Created by jh on 18-6-14.
//

#ifndef SSVO_IMU_VISION_ALIGN_H
#define SSVO_IMU_VISION_ALIGN_H

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "global.hpp"
#include "frame.hpp"
#include "new_parameters.h"
#include "optimizer.hpp"
#include <opencv2/core/eigen.hpp>

///估计陀螺仪bias
namespace ssvo
{

Vector3d EstimateGyrBias(deque<Frame::Ptr> &initilization_frame_buffer_);

///估计重力、速度、尺度
bool EstimateGVS(deque<Frame::Ptr> &initilization_frame_buffer_, VectorXd &x);

///重力求精
bool RefineGravity(deque<Frame::Ptr> &initilization_frame_buffer_, VectorXd &x);

MatrixXd TangentBasis(Vector3d &g0);

}

#endif //SSVO_IMU_VISION_ALIGN_H
