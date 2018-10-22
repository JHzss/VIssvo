//
// Created by jh on 18-6-5.
//
#include "global.hpp"
#include "new_parameters.h"
#include <opencv2/core/eigen.hpp>

//int WINDOW_SIZE = 10;
double acc_n;
double acc_w;
double gyr_n;
double gyr_w;
int vio_init_frames;
double vision_weight;
int SlideWindow_size;
cv::Mat Rc2b,tc2b;//Rotation from camera frame to imu frame
Eigen::Matrix3d eigen_Rc2b;
Eigen::Vector3d eigen_tc2b;

SE3d Tc2b;

Eigen::Vector3d G = Eigen::Vector3d(0.0, 0.0, 9.8107);

template <typename T>
T readParam(ros::NodeHandle &n, string name)
{
    T para;
    if(n.getParam(name,para))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << para);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return para;
}
void LoadParameters(ros::NodeHandle &n)
{
    string config_file;
    config_file=readParam<std::string>(n,"config_file");
    cv::FileStorage filename(config_file,cv::FileStorage::READ);
    if(!filename.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }
    else
    {
        cout<<"load config file successfully"<<endl;
    }
//    filename["imu_topic"]>> IMU_TOPIC;
//    filename["image_topic"]>> IMAGE_TOPIC;
    filename["extrinsicRotation"]>> Rc2b;
    filename["extrinsicTranslation"]>> tc2b;

    cv::cv2eigen(Rc2b,eigen_Rc2b);
    cv::cv2eigen(tc2b,eigen_tc2b);

    Tc2b = SE3d(eigen_Rc2b,eigen_tc2b);

//    Tc2b.translation() = eigen_tc2b;
//    Tc2b.rotationMatrix() = eigen_Rc2b;
//
//    camera_fx=filename["camera.fx"];
//    camera_fy=filename["camera.fy"];
//    camera_cx=filename["camera.cx"];
//    camera_cy=filename["camera.cy"];
//
//    camera_k1=filename["camera.k1"];
//    camera_k2=filename["camera.k2"];
//    camera_p1=filename["camera.p1"];
//    camera_p2=filename["camera.p2"];

    acc_n=filename["acc_n"];
    acc_w=filename["acc_w"];
    gyr_n=filename["gyr_n"];
    gyr_w=filename["gyr_w"];
    vio_init_frames = filename["vio_init_frames"];
    vision_weight = filename["vision_weight"];
//    SlideWindow_size = filename["slidewindow_size"];

//    number_of_features=filename["number_of_features"];
//    init_dist=filename["init_dist"];
//
//
//
//    image_width=filename["image.width"];
//    image_height=filename["image.height"];
//    slideWindowsize=filename["slideWindowsize"];
//    camera_k=(Mat_<double>(3,3)<< camera_fx,0,camera_cx,0,camera_fy,camera_cy,0,0,1.0);
    cout<<"load finished"<<endl;
}

