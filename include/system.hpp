#ifndef _SSVO_SYSTEM_HPP_
#define _SSVO_SYSTEM_HPP_

#include "global.hpp"
#include "frame.hpp"
#include "keyframe.hpp"
#include "map.hpp"
#include "initializer.hpp"
#include "feature_detector.hpp"
#include "feature_tracker.hpp"
#include "local_mapping.hpp"
#include "depth_filter.hpp"
#include "viewer.hpp"
#include "preintegration.h"

namespace ssvo {

class System: public noncopyable
{
public:
    enum Stage{
        STAGE_INITALIZE,
        STAGE_NORMAL_FRAME,
        STAGE_RELOCALIZING
    };

    enum Status {
        STATUS_INITAL_RESET,
        STATUS_INITAL_PROCESS,
        STATUS_INITAL_SUCCEED,
        STATUS_TRACKING_BAD,
        STATUS_TRACKING_GOOD,
    };

    System(std::string config_file);

    void saveTrajectoryTUM(const std::string &file_name);

    ~System();

    void process(pair<vector<sensor_msgs::ImuPtr>,sensor_msgs::ImageConstPtr> &measure);

    Preintegration::Ptr Imu_process(vector<sensor_msgs::ImuPtr> &imus, Vector3d &ba, Vector3d &bg);

    std::string IMAGE_TOPIC_s;
    std::string IMU_TOPIC_s;

    double systemScale;
    vector<Vector3d> image_ba_s,image_bg_s;
    vector<Preintegration::Ptr> Preintegrations; //所有的预积分变量（还不确定是全部的还是本图像帧对应的）


private:

    void processFrame();

    Status tracking();

    Status initialize();

    Status reinitialize();

    Status relocalize();

    bool createNewKeyFrame();

    void finishFrame();

    void calcLightAffine();

    void drowTrackedPoints(const Frame::Ptr &frame, cv::Mat &dst);

    //! jh
    bool vio_process();

//    bool SFM();

private:

    struct Option{
        double min_kf_disparity;
        double min_ref_track_rate;

    } options_;

    Stage stage_;
    Status status_;

    AbstractCamera::Ptr camera_;
    FastDetector::Ptr fast_detector_;
    FeatureTracker::Ptr feature_tracker_;
    Initializer::Ptr initializer_;
    DepthFilter::Ptr depth_filter_;

    LocalMapper::Ptr mapper_;

    Viewer::Ptr viewer_;

    std::thread viewer_thread_;

    cv::Mat rgb_;
    Frame::Ptr last_frame_;
    Frame::Ptr current_frame_;
    KeyFrame::Ptr reference_keyframe_;
    KeyFrame::Ptr last_keyframe_;

    double time_;

    std::list<double > frame_timestamp_buffer_;
    std::list<Sophus::SE3d> frame_pose_buffer_;
    std::list<KeyFrame::Ptr> reference_keyframe_buffer_;

    //! jh
    //初始化的两帧的id
    int firstframe_id;
    int secondframe_id;

    bool RT_success;
    bool vio_init;
    bool correctScale;


    std::deque<Frame::Ptr> initilization_frame_buffer_;
    std::vector<Frame::Ptr> all_frame_buffer_;


};

}// namespce ssvo

#endif //SSVO_SYSTEM_HPP
