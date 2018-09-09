//
// Created by jh on 18-6-4.
//

#ifndef SSVO_NEW_PARAMETERS_H
#define SSVO_NEW_PARAMETERS_H

#include "global.hpp"

extern double acc_n,acc_w;
extern double gyr_n,gyr_w;
extern cv::Mat Rc2b,tc2b;//Rotation from camera frame to imu frame
extern Eigen::Matrix3d eigen_Rc2b;
extern Eigen::Vector3d eigen_tc2b;
extern Eigen::Vector3d G;

extern SE3d Tc2b;

void LoadParameters(ros::NodeHandle &n);
#endif //SSVO_NEW_PARAMETERS_H
