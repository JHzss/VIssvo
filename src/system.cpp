#include "config.hpp"
#include "system.hpp"
#include "optimizer.hpp"
#include "image_alignment.hpp"
#include "feature_alignment.hpp"
#include "time_tracing.hpp"
#include "imu_vision_align.h"
#include "new_parameters.h"

using namespace cv;
namespace ssvo{

std::string Config::FileName;

TimeTracing::Ptr sysTrace = nullptr;

System::System(std::string config_file) :
    stage_(STAGE_INITALIZE), status_(STATUS_INITAL_RESET),
    last_frame_(nullptr), current_frame_(nullptr), reference_keyframe_(nullptr),firstframe_id(-1),vio_init(false),RT_success(false),correctScale(false),filescale(false)
{
    systemScale = -1.0;
    LOG_ASSERT(!config_file.empty()) << "Empty Config file input!!!";
    Config::FileName = config_file;

    double fps = Config::cameraFps();
    if(fps < 1.0) fps = 1.0;
    //! image
    IMAGE_TOPIC_s=Config::getIMAGE_TOPIC();
    IMU_TOPIC_s=Config::getIMU_TOPIC();
    //todo 把imu的参数添加进去
    const int width = Config::imageWidth();
    const int height = Config::imageHeight();
    const int level = Config::imageTopLevel();
    const int image_border = AlignPatch::Size;
    //! camera
    const cv::Mat K = Config::cameraIntrinsic();
    const cv::Mat DistCoef = Config::cameraDistCoefs(); //! for pinhole
    const double s = Config::cameraDistCoef();  //! for atan
    //! corner detector
    const int grid_size = Config::gridSize();
    const int grid_min_size = Config::gridMinSize();
    const int fast_max_threshold = Config::fastMaxThreshold();
    const int fast_min_threshold = Config::fastMinThreshold();

    if(Config::cameraModel() == Config::CameraModel::PINHOLE)
    {
        PinholeCamera::Ptr pinhole_camera = PinholeCamera::create(width, height, K, DistCoef);
        camera_ = std::static_pointer_cast<AbstractCamera>(pinhole_camera); //!转换成对应的模版类型
    }
    else if(Config::cameraModel() == Config::CameraModel::ATAN)
    {
        AtanCamera::Ptr atan_camera = AtanCamera::create(width, height, K, s);
        camera_ = std::static_pointer_cast<AbstractCamera>(atan_camera);
    }

    fast_detector_ = FastDetector::create(width, height, image_border, level+1, grid_size, grid_min_size, fast_max_threshold, fast_min_threshold);
    feature_tracker_ = FeatureTracker::create(width, height, 20, image_border, true);
    initializer_ = Initializer::create(fast_detector_, true);
    mapper_ = LocalMapper::create(true, false);
    //! https://www.cnblogs.com/bencai/p/9124654.html
    DepthFilter::Callback depth_fliter_callback = std::bind(&LocalMapper::createFeatureFromSeed, mapper_, std::placeholders::_1);
    depth_filter_ = DepthFilter::create(fast_detector_, depth_fliter_callback, true);

//    depth_filter_new = DepthFilter::create(fast_detector_, depth_fliter_callback, true);


    viewer_ = Viewer::create(mapper_->map_, cv::Size(width, height));

    mapper_->startMainThread();

    depth_filter_->startMainThread();

//    depth_filter_new->startMainThread();

    time_ = 1000.0/fps;

    options_.min_kf_disparity = 100;//MIN(Config::imageHeight(), Config::imageWidth())/5;
    options_.min_ref_track_rate = 0.7;

    //! LOG and timer for system;
    TimeTracing::TraceNames time_names;
    time_names.push_back("total");
    time_names.push_back("processing");
    time_names.push_back("frame_create");
    time_names.push_back("img_align");
    time_names.push_back("feature_reproj");
    time_names.push_back("motion_ba");
    time_names.push_back("light_affine");
    time_names.push_back("per_depth_filter");
    time_names.push_back("finish");

    TimeTracing::TraceNames log_names;
    log_names.push_back("frame_id");
    log_names.push_back("num_feature_reproj");
    log_names.push_back("stage");

    string trace_dir = Config::timeTracingDirectory();
    sysTrace.reset(new TimeTracing("ssvo_trace_system", trace_dir, time_names, log_names));


}

System::~System()
{
    sysTrace.reset();

    viewer_->setStop();
    depth_filter_->stopMainThread();
//    depth_filter_new->stopMainThread();
    mapper_->stopMainThread();

    viewer_->waitForFinish();
}
Preintegration::Ptr System::Imu_process(vector<sensor_msgs::ImuPtr> &imus, Vector3d &ba, Vector3d &bg)
{
    double last_time = -1;
    Preintegration::Ptr pre_integration_tmp = Preintegration::creat(ba, bg);

    for (auto imu:imus)
    {
        double t = imu->header.stamp.toSec();
        if(last_time<0)
            pre_integration_tmp->dt_buf.emplace_back(0);
        else
        {
            double dt=t-last_time;
            pre_integration_tmp->sum_t+=dt;
            pre_integration_tmp->dt_buf.emplace_back(dt);
        }
        last_time=t;

        ROS_ASSERT(pre_integration_tmp->sum_t=(imus.back()->header.stamp.toSec()-imus.front()->header.stamp.toSec()));

        double ax = imu->linear_acceleration.x;//todo 这里重传播的时候还需要检查一遍
        double ay = imu->linear_acceleration.y;
        double az = imu->linear_acceleration.z;
        double gx = imu->angular_velocity.x;
        double gy = imu->angular_velocity.y;
        double gz = imu->angular_velocity.z;
        pre_integration_tmp->acc_buf.emplace_back(Vector3d(ax, ay, az));
        pre_integration_tmp->gyr_buf.emplace_back(Vector3d(gx, gy, gz));
    }
    LOG_ASSERT(pre_integration_tmp->dt_buf.size()==pre_integration_tmp->acc_buf.size()) << "wrong in copy imu measurement";
    /*
        //todo 需要检查一下这里定义成double结果对不对
        double t = imu->header.stamp.toSec();
        if (last_time < 0)
        {
            last_time = t;
            double ax = imu->linear_acceleration.x - ba[0];//todo 这里重传播的时候还需要检查一遍
            double ay = imu->linear_acceleration.y - ba[1];
            double az = imu->linear_acceleration.z - ba[2];
            double gx = imu->angular_velocity.x - bg[0];
            double gy = imu->angular_velocity.y - bg[1];
            double gz = imu->angular_velocity.z - bg[2];
            pre_integration_tmp->acc_0 = Vector3d(ax, ay, az);
            pre_integration_tmp->gyr_0 = Vector3d(gx, gy, gz);
            continue;
        }

        //设置时间
        double dt = t - last_time;
        last_time = t;

        double ax = imu->linear_acceleration.x - ba[0];//todo 这里重传播的时候还需要检查一遍
        double ay = imu->linear_acceleration.y - ba[1];
        double az = imu->linear_acceleration.z - ba[2];
        double gx = imu->angular_velocity.x - bg[0];
        double gy = imu->angular_velocity.y - bg[1];
        double gz = imu->angular_velocity.z - bg[2];
        pre_integration_tmp->dt = dt;
        pre_integration_tmp->sum_t += dt;

        pre_integration_tmp->acc_1 = Vector3d(ax, ay, az);
        pre_integration_tmp->gyr_1 = Vector3d(gx, gy, gz);
        //todo 开始对预积分进行操作
        pre_integration_tmp->run();
        //todo 将IMU和图像对应上，需要定义frame_count,预积分
        pre_integration_tmp->img_stamp = imus.back()->header.stamp.toSec();
        pre_imu_txt<<fixed;
        pre_imu_txt<<preIntegrations[frame_count]->img_stamp<<"  "<<preIntegrations[frame_count]->sum_t<<endl;
        cout<<"p:"<<endl<<preIntegrations[frame_count]->dp<<endl;
        cout<<"v:"<<endl<<preIntegrations[frame_count]->dv<<endl;
        cout<<"q:"<<endl<<preIntegrations[frame_count]->dq.toRotationMatrix()<<endl;
        pre_imu_txt<<"dp: "<<preIntegrations[frame_count]->dp(0)<<" "<<preIntegrations[frame_count]->dp(1)<<" "<<preIntegrations[frame_count]->dp(2)<<endl;
        pre_imu_txt<<"dv: "<<preIntegrations[frame_count]->dv(0)<<" "<<preIntegrations[frame_count]->dv(1)<<" "<<preIntegrations[frame_count]->dv(2)<<endl;
        pre_imu_txt<<"dq: "<<preIntegrations[frame_count]->dq.w()<<" "<<preIntegrations[frame_count]->dq.x()<<" "<<preIntegrations[frame_count]->dq.y()<<" "<<preIntegrations[frame_count]->dq.z()<<endl;
        ROS_INFO("finish preintegration %d",frame_count);
        */
    pre_integration_tmp->run();
    return pre_integration_tmp;
}

void System::process(pair<vector<sensor_msgs::ImuPtr>,sensor_msgs::ImageConstPtr> &measure)
{
    if(!vio_init)
    {
//        bool ok1=false;
        //! jh 把用于视觉初始化之间的帧用来跟踪，计算位姿，增加关键帧的数量
        if(initilization_frame_buffer_.size()==(secondframe_id-firstframe_id+1))
        {

            cout<<"firstframe_id"<<firstframe_id<<endl;
            cout<<"secondframe_id"<<secondframe_id<<endl;
            last_frame_=initilization_frame_buffer_.front();
            KeyFrame::Ptr kf0 = mapper_->map_->getKeyFrame(0);
            reference_keyframe_ = kf0;
            for(auto frame:initilization_frame_buffer_)
            {
                if(frame->id_!=firstframe_id&&frame->id_!=secondframe_id)
                {
                    current_frame_=frame;
                    status_=tracking();
                    last_frame_=current_frame_;
                }
            }
            reference_keyframe_= initilization_frame_buffer_.back()->getRefKeyFrame();
        }
        /*
        //check imu observibility
        if(STAGE_NORMAL_FRAME == stage_)
        {
            std::deque<Frame::Ptr>::iterator frame_it;
            Vector3d sum_g;
            for (frame_it = initilization_frame_buffer_.begin(), frame_it++; frame_it != initilization_frame_buffer_.end(); frame_it++)
            {
                double dt = (*frame_it)->preintegration->sum_t;
                Vector3d tmp_g = (*frame_it)->preintegration->dv / dt;
                sum_g += tmp_g;
            }
            Vector3d aver_g;
            aver_g = sum_g * 1.0 / ((int)initilization_frame_buffer_.size() - 1);
            double var = 0;
            for (frame_it = initilization_frame_buffer_.begin(), frame_it++; frame_it != initilization_frame_buffer_.end(); frame_it++)
            {
                double dt = (*frame_it)->preintegration->sum_t;
                Vector3d tmp_g = (*frame_it)->preintegration->dv / dt;
                var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
                //cout << "frame g " << tmp_g.transpose() << endl;
            }
            var = sqrt(var / ((int)initilization_frame_buffer_.size() - 1));
            //ROS_WARN("IMU variation %f!", var);
            if(var < 0.25)
            {
                ROS_INFO("IMU excitation not enouth!");
                ok1=false;
                //return false;
            }
            else
                ok1=true;
        }
        if( ok1&&initilization_frame_buffer_.size()>50)
         */
        //! 50更利于imu和相机之间的初始化 20 100的效果都不如50，但是应该会根据数据集变化， 50的话是 3.95  3.96
        if(initilization_frame_buffer_.size()>=50)
        {
            vio_init = vio_process();

            if(!vio_init)
            {
                initilization_frame_buffer_.pop_front();
            }
            else
            {
                viewer_->setTraScale(systemScale);
                mapper_->map_->applyScaleCorrect(systemScale);
                last_frame_ =initilization_frame_buffer_.back();
                cout<<"initilization_frame_buffer_.back() frame id:"<<current_frame_->id_<<endl;
            }

        }
    }


    sensor_msgs::ImageConstPtr ros_image;
    ros_image=measure.second;
    cv_bridge::CvImagePtr cv_img_ptr = cv_bridge::toCvCopy(ros_image,sensor_msgs::image_encodings::BGR8);
    cv::Mat image=cv_img_ptr->image;
    double timestamp=ros_image->header.stamp.toSec();
    //! 以上 by jh


    sysTrace->startTimer("total");
    sysTrace->startTimer("frame_create");
    //! get gray image
    double t0 = (double)cv::getTickCount();
    rgb_ = image;
    cv::Mat gray = image.clone();
    if(gray.channels() == 3)
        cv::cvtColor(gray, gray, cv::COLOR_RGB2GRAY);

    Vector3d ba_last,bg_last,v_last;
    ba_last.setZero();
    bg_last.setZero();
    if(last_frame_ != nullptr)
    {
        ba_last=last_frame_->preintegration->ba;
        bg_last=last_frame_->preintegration->bg;
        v_last = last_frame_->v;
    }
    Preintegration::Ptr preintegration_frame;
    //! save imus to preintegration and send it to the frame
    preintegration_frame = Imu_process(measure.first,ba_last,bg_last);
    current_frame_ = Frame::create(gray, timestamp, camera_, ba_last, bg_last,preintegration_frame);
    current_frame_->v = v_last;

    all_frame_buffer_.emplace_back(current_frame_);

    double t1 = (double)cv::getTickCount();
    LOG(WARNING) << "[System] Frame " << current_frame_->id_ << " create time: " << (t1-t0)/cv::getTickFrequency();
    sysTrace->log("frame_id", current_frame_->id_);
    sysTrace->stopTimer("frame_create");

    sysTrace->startTimer("processing");
    if(STAGE_NORMAL_FRAME == stage_)
    {
        status_ = tracking();
    }
    else if(STAGE_INITALIZE == stage_)
    {
        status_ = initialize();
    }
    else if(STAGE_RELOCALIZING == stage_)
    {

        if(vio_init)
        {
            cout<<"STAGE_RELOCALIZING == stage_"<<endl;
            depth_filter_->stopMainThread();
            mapper_->stopMainThread();
            waitKey(0);
        }

        status_ = relocalize();
    }
    sysTrace->stopTimer("processing");

    finishFrame();
}

System::Status System::initialize()
{

        const Initializer::Result result = initializer_->addImage(current_frame_);

        if(result == Initializer::RESET)
            return STATUS_INITAL_RESET;
        else if(result == Initializer::FAILURE || result == Initializer::READY)
            return STATUS_INITAL_PROCESS;

        //! SUCCESS 之后进行下面这一段
        std::vector<Vector3d> points;
        initializer_->createInitalMap(Config::mapScale());
        mapper_->createInitalMap(initializer_->getReferenceFrame(), current_frame_);

        LOG(WARNING) << "number of init---------------------------------------------------------"<<current_frame_->id_;
        LOG(WARNING) << "[System] Start two-view BA";


        KeyFrame::Ptr kf0 = mapper_->map_->getKeyFrame(0);
        KeyFrame::Ptr kf1 = mapper_->map_->getKeyFrame(1);

        firstframe_id=kf0->frame_id_;
        secondframe_id=kf1->frame_id_;

        LOG_ASSERT(kf0 != nullptr && kf1 != nullptr) << "Can not find intial keyframes in map!";

        Optimizer::twoViewBundleAdjustment(kf0, kf1, true);

        LOG(WARNING) << "[System] End of two-view BA";


        current_frame_->setPose(kf1->pose());
        current_frame_->setRefKeyFrame(kf1);
        reference_keyframe_ = kf1;
        last_keyframe_ = kf1;

//    cout<<"kf1:"<<kf1->frame_id_<<endl;
//    cout<<"kf1 pose:"<<kf1->Tcw().translation()<<endl;
//    waitKey(0);

        depth_filter_->insertFrame(current_frame_, kf1);

    for(auto frame:all_frame_buffer_)
    {
        if(frame->id_>=firstframe_id)
            initilization_frame_buffer_.emplace_back(frame);
    }

    initializer_->reset();

//    ros::shutdown();



//    waitKey(0);

    return STATUS_INITAL_SUCCEED;
}


System::Status System::tracking()
{
    //! jh 输出
    /*
    cout<<"last_frame_ id "<<last_frame_->id_<<endl;
    cout<<"current_frame_ id "<<current_frame_->id_<<endl;
    cout<<"reference_keyframe_ id "<<reference_keyframe_->frame_id_<<endl;
    cout<<"last_frame_ pose "<<(last_frame_->pose()).rotationMatrix()<<endl;
    cout<<"current_frame_ pose "<<(current_frame_->pose()).rotationMatrix()<<endl;
     */


    current_frame_->setRefKeyFrame(reference_keyframe_);


    //! track seeds

    //todo 没有种子点怎么办?
    depth_filter_->trackFrame(last_frame_, current_frame_);//追踪上一帧的种子点，添加当前帧的种子点

    // TODO 先验信息怎么设置？

    current_frame_->setPose(last_frame_->pose());

    ///粗略估计当前帧的位姿，而不是像ORB那样很粗略
    //! alignment by SE3
    AlignSE3 align;
    sysTrace->startTimer("img_align");
    align.run(last_frame_, current_frame_, Config::alignTopLevel(), Config::alignBottomLevel(), 30, 1e-8);
    sysTrace->stopTimer("img_align");

    //! track local map
    sysTrace->startTimer("feature_reproj");

    //todo  here!
    int matches = feature_tracker_->reprojectLoaclMap(current_frame_);


    cout<<"---------------------------------------- matches when tracking -------------------------------------"<<endl;

    cout<<last_frame_->featureNumber()<<endl;
    cout<<current_frame_->featureNumber()<<endl;

    sysTrace->stopTimer("feature_reproj");
    sysTrace->log("num_feature_reproj", matches);
    LOG(WARNING) << "[System] Track with " << matches << " points";

    cout<<"matches:"<<matches<<endl;

    // TODO tracking status
    if(matches < Config::minQualityFts())
        return STATUS_TRACKING_BAD;
    //! motion-only BA
    sysTrace->startTimer("motion_ba");

    //todo by jh 最主要的就是改这个函数了,注意使用上面的 vio_init
//    cout<<"test 3"<<endl;
    Optimizer::motionOnlyBundleAdjustment(last_frame_,current_frame_, false ,vio_init, true, true);
    sysTrace->stopTimer("motion_ba");

    sysTrace->startTimer("per_depth_filter");

    if(createNewKeyFrame(matches) )
    {

        depth_filter_->insertFrame(current_frame_, reference_keyframe_);

        mapper_->insertKeyFrame(reference_keyframe_);
    }
    else
    {
        depth_filter_->insertFrame(current_frame_, nullptr);
    }
    sysTrace->stopTimer("per_depth_filter");

    sysTrace->startTimer("light_affine");
    calcLightAffine();
    sysTrace->stopTimer("light_affine");

    //！ save frame pose
    frame_timestamp_buffer_.push_back(current_frame_->timestamp_);
    reference_keyframe_buffer_.push_back(current_frame_->getRefKeyFrame());
    frame_pose_buffer_.push_back(current_frame_->pose());//current_frame_->getRefKeyFrame()->Tcw() * current_frame_->pose());

    if(!vio_init&&current_frame_->id_>secondframe_id)initilization_frame_buffer_.emplace_back(current_frame_);

//    waitKey(0);

    return STATUS_TRACKING_GOOD;
}

System::Status System::relocalize()
{
    Corners corners_new;
    Corners corners_old;
    fast_detector_->detect(current_frame_->images(), corners_new, corners_old, Config::minCornersPerKeyFrame());

    reference_keyframe_ = mapper_->relocalizeByDBoW(current_frame_, corners_new);

    if(reference_keyframe_ == nullptr)
        return STATUS_TRACKING_BAD;

    current_frame_->setPose(reference_keyframe_->pose());

    //! alignment by SE3
    AlignSE3 align;
    int matches = align.run(reference_keyframe_, current_frame_, Config::alignTopLevel(), Config::alignBottomLevel(), 30, 1e-8);

    if(matches < 30)
        return STATUS_TRACKING_BAD;

    current_frame_->setRefKeyFrame(reference_keyframe_);
    matches = feature_tracker_->reprojectLoaclMap(current_frame_);

    if(matches < 30)
        return STATUS_TRACKING_BAD;

    Optimizer::motionOnlyBundleAdjustment(last_frame_,current_frame_, true, true, true);

    if(current_frame_->featureNumber() < 30)
        return STATUS_TRACKING_BAD;

    return STATUS_TRACKING_GOOD;
}

void System::calcLightAffine()
{
    std::vector<Feature::Ptr> fts_last;
    last_frame_->getFeatures(fts_last);

    const cv::Mat img_last = last_frame_->getImage(0);
    const cv::Mat img_curr = current_frame_->getImage(0).clone() * 1.3;

    const int size = 4;
    const int patch_area = size*size;
    const int N = (int)fts_last.size();
    cv::Mat patch_buffer_last = cv::Mat::zeros(N, patch_area, CV_32FC1);
    cv::Mat patch_buffer_curr = cv::Mat::zeros(N, patch_area, CV_32FC1);

    int count = 0;
    for(int i = 0; i < N; ++i)
    {
        const Feature::Ptr ft_last = fts_last[i];
        const Feature::Ptr ft_curr = current_frame_->getFeatureByMapPoint(ft_last->mpt_);

        if(ft_curr == nullptr)
            continue;

        utils::interpolateMat<uchar, float, size>(img_last, patch_buffer_last.ptr<float>(count), ft_last->px_[0], ft_last->px_[1]);
        utils::interpolateMat<uchar, float, size>(img_curr, patch_buffer_curr.ptr<float>(count), ft_curr->px_[0], ft_curr->px_[1]);

        count++;
    }

    patch_buffer_last.resize(count);
    patch_buffer_curr.resize(count);

    if(count < 20)
    {
        Frame::light_affine_a_ = 1;
        Frame::light_affine_b_ = 0;
        return;
    }

    float a=1;
    float b=0;
    calculateLightAffine(patch_buffer_last, patch_buffer_curr, a, b);
    Frame::light_affine_a_ = a;
    Frame::light_affine_b_ = b;

//    std::cout << "a: " << a << " b: " << b << std::endl;
}

bool System::createNewKeyFrame(int matches)
{
    std::map<KeyFrame::Ptr, int> overlap_kfs = current_frame_->getOverLapKeyFrames();

    std::vector<Feature::Ptr> fts;
    current_frame_->getFeatures(fts);
    std::map<MapPoint::Ptr, Feature::Ptr> mpt_ft;
    for(const Feature::Ptr &ft : fts)
    {
        mpt_ft.emplace(ft->mpt_, ft);
    }

    KeyFrame::Ptr max_overlap_keyframe;
    int max_overlap = 0;
    for(const auto &olp_kf : overlap_kfs)
    {
        if(olp_kf.second < max_overlap || (olp_kf.second == max_overlap && olp_kf.first->id_ < max_overlap_keyframe->id_))
            continue;

        max_overlap_keyframe = olp_kf.first;
        max_overlap = olp_kf.second;
    }

    //! check distance
    bool c1 = true;
    bool cjh = true;
    double median_depth = std::numeric_limits<double>::max();
    double min_depth = std::numeric_limits<double>::max();
    current_frame_->getSceneDepth(median_depth, min_depth);
//    for(const auto &ovlp_kf : overlap_kfs)
//    {
//        SE3d T_cur_from_ref = current_frame_->Tcw() * ovlp_kf.first->pose();
//        Vector3d tran = T_cur_from_ref.translation();
//        double dist1 = tran.dot(tran);
//        double dist2 = 0.1 * (T_cur_from_ref.rotationMatrix() - Matrix3d::Identity()).norm();
//        double dist = dist1 + dist2;
////        std::cout << "d1: " << dist1 << ". d2: " << dist2 << std::endl;
//        if(dist  < 0.10 * median_depth)
//        {
//            c1 = false;
//            break;
//        }
//    }

    SE3d T_cur_from_ref = current_frame_->Tcw() * last_keyframe_->pose();
    Vector3d tran = T_cur_from_ref.translation();
    double dist1 = tran.dot(tran);
    double dist2 = 0.01 * (T_cur_from_ref.rotationMatrix() - Matrix3d::Identity()).norm();
    if(dist1+dist2  < /*0.01*/0.01 * median_depth)
        c1 = false;

    //! jh 放宽初始化时的条件
    if(dist1+dist2  < 0.001 * median_depth)
        cjh = false;

    //! check disparity
    std::list<float> disparities;
    const int threahold = int (max_overlap * 0.6);
    for(const auto &ovlp_kf : overlap_kfs)
    {
        if(ovlp_kf.second < threahold)
            continue;

        std::vector<float> disparity;
        disparity.reserve(ovlp_kf.second);
        MapPoints mpts;
        ovlp_kf.first->getMapPoints(mpts);
        for(const MapPoint::Ptr &mpt : mpts)
        {
            Feature::Ptr ft_ref = mpt->findObservation(ovlp_kf.first);
            if(ft_ref == nullptr) continue;

            if(!mpt_ft.count(mpt)) continue;
            Feature::Ptr ft_cur = mpt_ft.find(mpt)->second;

            const Vector2d px(ft_ref->px_ - ft_cur->px_);
            disparity.push_back(px.norm());
        }

        std::sort(disparity.begin(), disparity.end());
        float disp = disparity.at(disparity.size()/2);
        disparities.push_back(disp);
    }
    disparities.sort();

    if(!disparities.empty())
        current_frame_->disparity_ = *std::next(disparities.begin(), disparities.size()/2);

    LOG(INFO) << "[System] Max overlap: " << max_overlap << " min disaprity " << disparities.front() << ", median: " << current_frame_->disparity_;

//    int all_features = current_frame_->featureNumber() + current_frame_->seedNumber();
    bool c2 = disparities.front() > options_.min_kf_disparity;
    bool c3 = current_frame_->featureNumber() < reference_keyframe_->featureNumber() * options_.min_ref_track_rate;
//    bool c4 = current_frame_->featureNumber() < reference_keyframe_->featureNumber() * 0.9;

    //! create new keyFrame
    //! 这里好像没有什么用
    if(c1 && (c2 || c3 )/* || (cjh && !vio_init) || matches < 150*/)
    {
        //! create new keyframe
        KeyFrame::Ptr new_keyframe = KeyFrame::create(current_frame_);

        for(const Feature::Ptr &ft : fts)
        {
            if(ft->mpt_->isBad())
            {
                current_frame_->removeFeature(ft);
                continue;
            }

            ft->mpt_->addObservation(new_keyframe, ft);
            ft->mpt_->updateViewAndDepth();
//            mapper_->addOptimalizeMapPoint(ft->mpt_);
        }
        new_keyframe->updateConnections();
        reference_keyframe_ = new_keyframe;
        last_keyframe_ = new_keyframe;
        cout<<"here 0 "<<reference_keyframe_->id_<<endl;
//        LOG(ERROR) << "C: (" << c1 << ", " << c2 << ", " << c3 << ") cur_n: " << current_frame_->N() << " ck: " << reference_keyframe_->N();
        return true;
    }
        //! change reference keyframe
    else
    {
        cout<<"here1"<<endl;
        if(overlap_kfs[reference_keyframe_] < max_overlap * 0.85)
            reference_keyframe_ = max_overlap_keyframe;
        return false;
    }
}

void System::finishFrame()
{
    sysTrace->startTimer("finish");
    cv::Mat image_show;
//    Stage last_stage = stage_;
    if(STAGE_NORMAL_FRAME == stage_)
    {
        if(STATUS_TRACKING_BAD == status_)
        {
            stage_ = STAGE_RELOCALIZING;
            current_frame_->setPose(last_frame_->pose());
        }
    }
    else if(STAGE_INITALIZE == stage_)
    {
        if(STATUS_INITAL_SUCCEED == status_)
            stage_ = STAGE_NORMAL_FRAME;
        else if(STATUS_INITAL_RESET == status_)
            initializer_->reset();

        initializer_->drowOpticalFlow(image_show);
    }
    else if(STAGE_RELOCALIZING == stage_)
    {
        if(STATUS_TRACKING_GOOD == status_)
            stage_ = STAGE_NORMAL_FRAME;
        else
            current_frame_->setPose(last_frame_->pose());
    }

    //! update
    last_frame_ = current_frame_;

    //! display 主要是显示轨迹用的
    viewer_->setCurrentFrame(current_frame_, image_show);

    sysTrace->log("stage", stage_);
    sysTrace->stopTimer("finish");
    sysTrace->stopTimer("total");
    const double time = sysTrace->getTimer("total");
    LOG(WARNING) << "[System] Finish Current Frame with Stage: " << stage_ << ", total time: " << time;

    sysTrace->writeToFile();

}

void System::saveTrajectoryTUM(const std::string &file_name)
{
    std::ofstream f;
    f.open(file_name.c_str());
    f << std::fixed;

    std::list<double>::iterator frame_timestamp_ptr = frame_timestamp_buffer_.begin();
    std::list<Sophus::SE3d>::iterator frame_pose_ptr = frame_pose_buffer_.begin();
    std::list<KeyFrame::Ptr>::iterator reference_keyframe_ptr = reference_keyframe_buffer_.begin();
    const std::list<double>::iterator frame_timestamp = frame_timestamp_buffer_.end();
    for(; frame_timestamp_ptr!= frame_timestamp; frame_timestamp_ptr++, frame_pose_ptr++, reference_keyframe_ptr++)
    {
        Sophus::SE3d frame_pose = (*frame_pose_ptr);//(*reference_keyframe_ptr)->Twc() * (*frame_pose_ptr);
        Vector3d t = frame_pose.translation();
        Quaterniond q = frame_pose.unit_quaternion();

        f << std::setprecision(6) << *frame_timestamp_ptr << " "
          << std::setprecision(9) << t[0] << " " << t[1] << " " << t[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    }
    f.close();
    LOG(INFO) << " Trajectory saved!";
}

bool System::vio_process()
    {
        string file_name ="/home/jh/trajectory_ssvo.txt";
        std::ofstream f;
        f.open(file_name.c_str());
        f << std::fixed;

        for(auto frame:initilization_frame_buffer_)
            {
                Sophus::SE3d frame_pose = frame->pose();//(*reference_keyframe_ptr)->Twc() * (*frame_pose_ptr);
                Vector3d t = frame_pose.translation();
                Quaterniond q = frame_pose.unit_quaternion();

                f << std::setprecision(6) << frame->timestamp_ << " "
                  << std::setprecision(9) << t[0] << " " << t[1] << " " << t[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
            }
        f.close();
        cout << " Trajectory_ssvo saved!"<<endl;


        Vector3d GyrBais=EstimateGyrBais(initilization_frame_buffer_);

//        GyrBais=EstimateGyrBais(initilization_frame_buffer_);

        VectorXd x;
        bool GVSSuccess=EstimateGVS(initilization_frame_buffer_,x);
        if(!GVSSuccess)
        {
            return false;
        }
        /*
    string file_name_imu="/home/jh/trajectory_imu.txt";
    f.open(file_name_imu.c_str());
    f << std::fixed;
    SE3d T1= SE3d(Matrix3d::Identity(),Vector3d(0,0,0));
    Frame::Ptr last_frame= initilization_frame_buffer_.front();
    int i=0;
    Matrix3d eigen_Rb2c=eigen_Rc2b.transpose();
    for(auto frame:initilization_frame_buffer_)
    {
        i++;
        Matrix3d R=(last_frame->pose().rotationMatrix())*frame->preintegration->dR;

        Vector3d t = (last_frame->pose().rotationMatrix())*(frame->preintegration->dp)+last_frame->pose().translation()*last_frame->s-last_frame->pose().rotationMatrix()*eigen_tc2b
                     +(last_frame->Twc().rotationMatrix()*eigen_Rb2c)*(last_frame->v)*last_frame->preintegration->sum_t
                     +0.5*Vector3d(0,0,9.8)*last_frame->preintegration->sum_t*last_frame->preintegration->sum_t;
//        Vector3d t = last_frame->pose().translation()*last_frame->s-last_frame->pose().rotationMatrix()*eigen_tc2b;

        Quaterniond q(R);
        SE3d T_(R,(t+last_frame->pose().rotationMatrix()*eigen_tc2b));
        t=T_.translation();
        q=T_.unit_quaternion();
        last_frame=frame;
        if(i==1)continue;

        f << std::setprecision(6) << frame->timestamp_ << " "
          << std::setprecision(9) << t[0] << " " << t[1] << " " << t[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;

    }
    f.close();
    cout << " Trajectory_imu saved!"<<endl;
         */
        bool RefineGSuccess=RefineGravity(initilization_frame_buffer_,x);

        /*
        string file_name_new="/home/jh/trajectory_ssvo_new.txt";
        f.open(file_name_new.c_str());
        f << std::fixed;

        for(auto frame:initilization_frame_buffer_)
        {
            Sophus::SE3d frame_pose = frame->pose();//(*reference_keyframe_ptr)->Twc() * (*frame_pose_ptr);
            Vector3d t = frame->s*frame_pose.translation();
//            Vector3d t = 4*frame_pose.translation();
            Quaterniond q = frame_pose.unit_quaternion();

            f << std::setprecision(6) << frame->timestamp_  << " "
              << std::setprecision(9) << t[0] << " " << t[1] << " " << t[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
        }
        f.close();
        cout << " Trajectory_ssvo_new saved!"<<endl;
        waitKey(0);
        */

        //! jh change the state

        systemScale=(x.tail<1>())(0)/100;
//        systemScale = 1;


//    waitKey(0);

        for(auto &frame:initilization_frame_buffer_)
        {

                frame->optimal_Tcw_=frame->Tcw();
                frame->optimal_Tcw_.translation() *= systemScale;
                frame->setTcw(frame->optimal_Tcw_);

            /*
//            MapPoints mpts;
//            frame->getMapPoints(mpts);
//
//            for(auto &mpt : mpts)
//            {
//                if(find(ids_mp.begin(),ids_mp.end(),mpt->id_)==ids_mp.end())
//                {
//                    mpt->setPose((mpt->pose())*scale);
//                    mpt->correctscale(scale);
//                    ids_mp.push_back(mpt->id_);
//
//                    KeyFrame::Ptr kf=mpt->getReferenceKeyFrame();
//                    if(find(ids.begin(),ids.end(),kf->id_)==ids.end())
//                    {
//                        kf->optimal_Tcw_=kf->Tcw();
//                        kf->optimal_Tcw_.translation()*=scale;
//                        kf->setTcw(kf->optimal_Tcw_);
//                        ids.push_back(kf->id_);
//                    }
//                }
//            }
             */
//            frame->removeAllSeed();
        }




        return true;
    }

}

