#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <include/config.hpp>

#include "frame.hpp"
#include "keyframe.hpp"
#include "utils.hpp"

namespace ssvo {

uint64_t Frame::next_id_ = 0;
const cv::Size Frame::optical_win_size_ = cv::Size(21,21);
float Frame::light_affine_a_ = 1.0f;
float Frame::light_affine_b_ = 0.0f;

Frame::Frame(const cv::Mat &img, const double timestamp, const AbstractCamera::Ptr &cam, Vector3d ba_, Vector3d bg_, Preintegration::Ptr& preint) :
    id_(next_id_++), timestamp_(timestamp), cam_(cam), max_level_(Config::imageTopLevel()), preintegration(preint)
{
    v = Vector3d(0,0,0);
    gray_image=img;
    Tcw_ = SE3d(Matrix3d::Identity(), Vector3d::Zero());
    Twc_ = Tcw_.inverse();

//    utils::createPyramid(img, img_pyr_, nlevels_);
    //! create pyramid for optical flow
    cv::buildOpticalFlowPyramid(img, optical_pyr_, optical_win_size_, max_level_, false);
    LOG_ASSERT(max_level_ == (int) optical_pyr_.size()-1) << "The pyramid level is unsuitable! maxlevel should be " << optical_pyr_.size()-1;

    //! copy to image pyramid
    img_pyr_.resize(optical_pyr_.size());
    for(size_t i = 0; i < optical_pyr_.size(); i++)
        optical_pyr_[i].copyTo(img_pyr_[i]);

}

Frame::Frame(const ImgPyr &img_pyr, const uint64_t id, const double timestamp, const AbstractCamera::Ptr &cam) :
    id_(id), timestamp_(timestamp), cam_(cam), max_level_(Config::imageTopLevel()), img_pyr_(img_pyr),
    Tcw_(SE3d(Matrix3d::Identity(), Vector3d::Zero())), Twc_(Tcw_.inverse())
{}

const ImgPyr Frame::images() const
{
    return img_pyr_;
}

const ImgPyr Frame::opticalImages() const
{
    return optical_pyr_;
}

const cv::Mat Frame::getImage(int level) const
{
    LOG_ASSERT(level < (int) img_pyr_.size()) << "Error level: " << level;
    return img_pyr_[level];
}



    SE3d Frame::Tbw()
    {
        std::lock_guard<std::mutex> lock(mutex_pose_);
        return Tbw_;
    }

    SE3d Frame::Twb()
    {
        std::lock_guard<std::mutex> lock(mutex_pose_);
        return Twb_;
    }

SE3d Frame::Tcw()
{
    std::lock_guard<std::mutex> lock(mutex_pose_);
    return Tcw_;
}

SE3d Frame::Twc()
{
    std::lock_guard<std::mutex> lock(mutex_pose_);
    return Twc_;
}

SE3d Frame::pose()
{
    std::lock_guard<std::mutex> lock(mutex_pose_);
    return Twc_;
}

Vector3d Frame::ray()
{
    std::lock_guard<std::mutex> lock(mutex_pose_);
    return Dw_;
}

void Frame::setPose(const SE3d& pose)
{
    std::lock_guard<std::mutex> lock(mutex_pose_);
    Twc_ = pose;
    Tcw_ = Twc_.inverse();
    Dw_ = Tcw_.rotationMatrix().determinant() * Tcw_.rotationMatrix().col(2);

    Tbw_ = Tc2b * Tcw_;
    Twb_ = Tbw_.inverse();

    PVR[0] = (Twb_.translation()).x();
    PVR[1] = (Twb_.translation()).y();
    PVR[2] = (Twb_.translation()).z();
    PVR[3] = v.x();
    PVR[4] = v.y();
    PVR[5] = v.z();
    Vector3d phii = static_cast<Sophus_new::SO3>(Twb_.rotationMatrix()).log();
    PVR[6] = phii.x();
    PVR[7] = phii.y();
    PVR[8] = phii.z();

//    this->getRefKeyFrame()->setPose(pose);
}

void Frame::setPose(const Matrix3d& R, const Vector3d& t)
{
    std::lock_guard<std::mutex> lock(mutex_pose_);
    Twc_ = SE3d(R, t);
    Tcw_ = Twc_.inverse();
    Dw_ = Tcw_.rotationMatrix().determinant() * Tcw_.rotationMatrix().col(2);

    Tbw_ = Tc2b * Tcw_;
    Twb_ = Tbw_.inverse();

    PVR[0] = (Twb_.translation()).x();
    PVR[1] = (Twb_.translation()).y();
    PVR[2] = (Twb_.translation()).z();
    PVR[3] = v.x();
    PVR[4] = v.y();
    PVR[5] = v.z();
    Vector3d phii = static_cast<Sophus_new::SO3>(Twb_.rotationMatrix()).log();
    PVR[6] = phii.x();
    PVR[7] = phii.y();
    PVR[8] = phii.z();

//    this->getRefKeyFrame()->setPose(Twc_);
}

void Frame::setTcw(const SE3d &Tcw)
{
    std::lock_guard<std::mutex> lock(mutex_pose_);
    Tcw_ = Tcw;
    Twc_ = Tcw_.inverse();
    Dw_ = Tcw_.rotationMatrix().determinant() * Tcw_.rotationMatrix().col(2);

    Tbw_ = Tc2b * Tcw_;
    Twb_ = Tbw_.inverse();

    PVR[0] = (Twb_.translation()).x();
    PVR[1] = (Twb_.translation()).y();
    PVR[2] = (Twb_.translation()).z();
    PVR[3] = v.x();
    PVR[4] = v.y();
    PVR[5] = v.z();
    Vector3d phii = static_cast<Sophus_new::SO3>(Twb_.rotationMatrix()).log();
    PVR[6] = phii.x();
    PVR[7] = phii.y();
    PVR[8] = phii.z();
//    this->getRefKeyFrame()->setTcw(Twc_);
}

    void Frame::setTwb(const SE3d &Twb)
    {
        std::lock_guard<std::mutex> lock(mutex_pose_);
        Twb_ = Twb;
        Tbw_ = Twb_.inverse();
        Tcw_ = Tc2b.inverse() * Tbw_;
        Twc_ = Tcw_.inverse();
        Dw_ = Tcw_.rotationMatrix().determinant() * Tcw_.rotationMatrix().col(2);

        optimal_Twb_ = Twb;

        PVR[0] = (Twb_.translation()).x();
        PVR[1] = (Twb_.translation()).y();
        PVR[2] = (Twb_.translation()).z();
        PVR[3] = v.x();
        PVR[4] = v.y();
        PVR[5] = v.z();
        Vector3d phii = static_cast<Sophus_new::SO3>(Twb_.rotationMatrix()).log();
        PVR[6] = phii.x();
        PVR[7] = phii.y();
        PVR[8] = phii.z();
    }

bool Frame::isVisiable(const Vector3d &xyz_w, const int border)
{
    SE3d Tcw;
    {
        std::lock_guard<std::mutex> lock(mutex_pose_);
        Tcw = Tcw_;
    }
    const Vector3d xyz_c = Tcw * xyz_w;
    if(xyz_c[2] < 0.0f)
        return false;

    Vector2d px = cam_->project(xyz_c);
    return cam_->isInFrame(px.cast<int>(), border);
}

std::unordered_map<MapPoint::Ptr, Feature::Ptr> Frame::features()
{
    std::lock_guard<std::mutex> lock(mutex_feature_);
    return mpt_fts_;
}

int Frame::featureNumber()
{
    std::lock_guard<std::mutex> lock(mutex_feature_);
    return (int)mpt_fts_.size();
}

void Frame::getFeatures(std::vector<Feature::Ptr>& fts)
{
    if(!fts.empty()) fts.clear();

    std::lock_guard<std::mutex> lock(mutex_feature_);
    fts.reserve(mpt_fts_.size());
    for(const auto &it : mpt_fts_)
        fts.push_back(it.second);
}

void Frame::getMapPoints(std::list<MapPoint::Ptr> &mpts)
{
    if(!mpts.empty()) mpts.clear();

    std::lock_guard<std::mutex> lock(mutex_feature_);
    for(const auto &it : mpt_fts_)
        mpts.push_back(it.first);
}

bool Frame::addFeature(const Feature::Ptr &ft)
{
    LOG_ASSERT(ft->mpt_ != nullptr) << " The feature is invalid with empty mappoint!";
    std::lock_guard<std::mutex> lock(mutex_feature_);
    if(mpt_fts_.count(ft->mpt_))
    {
        LOG(ERROR) << " The mappoint is already be observed! Frame: " << id_ << " Mpt: " << ft->mpt_->id_
            << ", px: " << mpt_fts_.find(ft->mpt_)->second->px_.transpose() << ", " << ft->px_.transpose();
        return false;
    }

    mpt_fts_.emplace(ft->mpt_, ft);

    return true;
}

bool Frame::removeFeature(const Feature::Ptr &ft)
{
    std::lock_guard<std::mutex> lock(mutex_feature_);
    return (bool)mpt_fts_.erase(ft->mpt_);
}

bool Frame::removeMapPoint(const MapPoint::Ptr &mpt)
{
    std::lock_guard<std::mutex> lock(mutex_feature_);
    return (bool)mpt_fts_.erase(mpt);
}

Feature::Ptr Frame::getFeatureByMapPoint(const MapPoint::Ptr &mpt)
{
    std::lock_guard<std::mutex> lock(mutex_feature_);
    const auto it = mpt_fts_.find(mpt);
    if(it != mpt_fts_.end())
        return it->second;
    else
        return nullptr;
}

int Frame::seedNumber()
{
    std::lock_guard<std::mutex> lock(mutex_seed_);
    return (int)seed_fts_.size();
}

void Frame::getSeeds(std::vector<Feature::Ptr> &fts)
{
    if(!fts.empty()) fts.clear();
    fts.reserve(seed_fts_.size());

    std::lock_guard<std::mutex> lock(mutex_seed_);
    for(const auto &it : seed_fts_)
        fts.push_back(it.second);
}


bool Frame::addSeed(const Feature::Ptr &ft)
{
    LOG_ASSERT(ft->seed_ != nullptr) << " The feature is invalid with empty mappoint!";

    {
        std::lock_guard<std::mutex> lock(mutex_seed_);
        if(seed_fts_.count(ft->seed_))
        {
            LOG(ERROR) << " The seed is already exited ! Frame: " << id_ << " Seed: " << ft->seed_->id;
            return false;
        }

        seed_fts_.emplace(ft->seed_, ft);
    }

    return true;
}

bool Frame::removeSeed(const Seed::Ptr &seed)
{
    std::lock_guard<std::mutex> lock(mutex_seed_);
    return (bool) seed_fts_.erase(seed);
}

    //! by jh
    void Frame::removeAllSeed()
    {
        std::lock_guard<std::mutex> lock(mutex_seed_);
        seed_fts_.clear();
    }

    void Frame::removeAllFeatures()
    {
        std::lock_guard<std::mutex> lock(mutex_feature_);
        mpt_fts_.clear();
    }

bool Frame::hasSeed(const Seed::Ptr &seed)
{
    std::lock_guard<std::mutex> lock(mutex_seed_);
    return (bool) seed_fts_.count(seed);
}

bool Frame::getSceneDepth(double &depth_mean, double &depth_min)
{
    SE3d Tcw;
    {
        std::lock_guard<std::mutex> lock(mutex_pose_);
        Tcw = Tcw_;
    }
    Features fts;
    {
        std::lock_guard<std::mutex> lock(mutex_feature_);
        for(const auto &it : mpt_fts_)
            fts.push_back(it.second);
    }

    std::vector<double> depth_vec;
    depth_vec.reserve(fts.size());

    depth_min = std::numeric_limits<double>::max();
    for(const Feature::Ptr &ft : fts)
    {
        if(ft->mpt_ == nullptr)
            continue;

        const Vector3d p =  Tcw * ft->mpt_->pose();
        depth_vec.push_back(p[2]);
        depth_min = fmin(depth_min, p[2]);
    }

    if(depth_vec.empty())
        return false;

    depth_mean = utils::getMedian(depth_vec);
    return true;
}

std::map<KeyFrame::Ptr, int> Frame::getOverLapKeyFrames()
{
    std::list<MapPoint::Ptr> mpts;
    getMapPoints(mpts);

    std::map<KeyFrame::Ptr, int> overlap_kfs;

    for(const MapPoint::Ptr &mpt : mpts)
    {
        const std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();
        for(const auto &item : obs)
        {
            auto it = overlap_kfs.find(item.first);
            if(it != overlap_kfs.end())
                it->second++;
            else
                overlap_kfs.insert(std::make_pair(item.first, 1));
        }
    }

    return overlap_kfs;
}

    void Frame::updatePoseAndBias()
    {
        Vector3d t = Vector3d(PVR[0],PVR[1],PVR[2]);
        Vector3d w = Vector3d(PVR[6],PVR[7],PVR[8]);

        Matrix3d ttttt = Sophus::SO3d::exp(w).matrix();
        SE3d tmp(ttttt,t);
        setTwb(tmp);

        v = Vector3d(PVR[3],PVR[4],PVR[5]);

        preintegration->bg.x() = bgba[0];
        preintegration->bg.y() = bgba[1];
        preintegration->bg.z() = bgba[2];
        preintegration->ba.x() = bgba[3];
        preintegration->ba.y() = bgba[4];
        preintegration->ba.z() = bgba[5];
    }

    void Frame::updatePose()
    {
        Vector3d t = Vector3d(PVR[0],PVR[1],PVR[2]);
        Vector3d w = Vector3d(PVR[6],PVR[7],PVR[8]);

        Matrix3d ttttt = Sophus::SO3d::exp(w).matrix();
        SE3d tmp(ttttt,t);
        setTwb(tmp);
        v = Vector3d(PVR[3],PVR[4],PVR[5]);

    }

}