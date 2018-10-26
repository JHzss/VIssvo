#ifndef _GLOBAL_HPP_
#define _GLOBAL_HPP_

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Imu.h>
#include <cv_bridge/cv_bridge.h>

#include <cstdlib>
#include <stdint.h>
#include <assert.h>
#include <cmath>

#include <iostream>
#include <iomanip>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <random>
#include <deque>
#include <list>
#include <vector>
#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>

#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/calib3d/calib3d.hpp"


#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/so3.hpp>
#include "so3.h"
#include <sophus/se3.hpp>
#include <glog/logging.h>

#include<Eigen/StdVector>
//EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix2d)
//EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix3d)
//EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Vector3d)

#include <DBoW3/DBoW3.h>
#include <DBoW3/DescManip.h>
#include "brief.hpp"


using namespace Eigen;
using namespace std;
using Sophus::SE3d;

#ifndef MIN
    #define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
    #define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

typedef std::vector<cv::Mat> ImgPyr;

namespace ssvo{

static std::mt19937_64 rd;
static std::uniform_real_distribution<double> distribution(0.0, std::nextafter(1, std::numeric_limits<double>::max()));

inline double Rand(double min, double max)
{ return (((double)distribution(rd) * (max - min + 1))) + min;}

inline int Rand(int min, int max)
{ return (((double)distribution(rd) * (max - min + 1))) + min;}

class noncopyable
{
protected:
    noncopyable() = default;
    ~noncopyable() = default;

    noncopyable(const noncopyable&) = delete;
    noncopyable &operator=(const noncopyable&) = delete;
};

}

#endif