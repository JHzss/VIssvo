#include <opencv2/opencv.hpp>
#include "global.hpp"
#include "utils.hpp"
#include "initializer.hpp"

namespace ssvo{

int FrameCandidate::size = 1000;

FrameCandidate::FrameCandidate(const Frame::Ptr &frame) :
    frame(frame)
{
    level.resize(size, -1);
    pts.resize(size, cv::Point2f(0, 0));
    idx.resize(size, -1);
}

FrameCandidate::FrameCandidate(const Frame::Ptr &frame, const FrameCandidate::Ptr &cand) :
    frame(frame), pts(cand->pts), level(cand->level), idx(cand->idx)
{}

void FrameCandidate::createFts()
{
    std::vector<cv::Point2f> temp_udist;
    frame->cam_->undistortPoints(pts, temp_udist);
    fts.resize(size);
    for(int i = 0; i < FrameCandidate::size; ++i)
    {
        fts[i].x = (double)temp_udist[i].x;
        fts[i].y = (double)temp_udist[i].y;
    }
}

int FrameCandidate::getInliers(std::vector<bool> &inliers)
{
    int count = 0;
    inliers.resize(size);
    std::fill(inliers.begin(), inliers.end(), false);
    for(int i = 0; i < size; ++i)
    {
        if(idx[i] < 0)
            continue;

        inliers[i] = true;
    }
    return count;
}

int FrameCandidate::updateInliers(const std::vector<bool> &inliers)
{
    int count = 0;
    for(int i = 0; i < size; ++i)
    {
        if(inliers[i])
        {
            count++;
            continue;
        }

        idx[i] = -1;
        pts[i].x = 0;
        pts[i].y = 0;
        level[i] = -1;
    }
    return count;
}

void FrameCandidate::getMatch(std::vector<bool> &mask, const int ref_idx)
{
    mask.resize(size);
    for(int i = 0; i < size; ++i)
    {
        if(idx[i] != -1 && idx[i] <= ref_idx)
            mask[i] = true;
        else
            mask[i] = false;
    }
}

int FrameCandidate::checkTracking(const int min_idx, const int max_idx, const int min_track)
{
    assert(min_idx <= max_idx);

    const int N = max_idx-min_idx+1;
    std::vector<int> count(N, 0);
    for(int i = 0; i < size; ++i)
    {
        if(idx[i] < 0)
            continue;

        if(idx[i] < min_idx)
            idx[i] = min_idx;

        count[idx[i]-min_idx]++;
    }

//    std::cout << " [";
//    for(int j = 0; j < N; ++j)
//    {
//        std::cout << " " << count[j];
//    }
//    std::cout << " ]" << std::endl;

    //! | ref [] ... [] cur |
    //!    0  1  ... n-1 n
    int delta = N;
    int corners_count = 0;
    for(int i = 0; i < N; ++i)
    {
        corners_count += count[i];
        if(corners_count < min_track)
            continue;

        delta = i;
        break;
    }

    return delta;
}

Initializer::Initializer(const FastDetector::Ptr &fast_detector, bool verbose):
    fast_detector_(fast_detector), cand_ref_(nullptr), cand_cur_(nullptr), cand_last_(nullptr), finished_(false), verbose_(verbose)
{
    FrameCandidate::size = Config::initMinCorners();
};


void Initializer::reset()
{

    frame_buffer_.clear();

    p3ds_.clear();
    disparities_.clear();
    inliers_.clear();
    finished_ = false;
}

Initializer::Result Initializer::createNewCorners(const FrameCandidate::Ptr &candidate)
{

    ///jh 旧的角点
    ///目的：不在旧的特征点附近提取特征点，原理应该与VINS相同
    Corners old_corners;
    for(int i = 0; i < FrameCandidate::size; ++i)  ///1000
    {
        if(candidate->idx[i] < 0)
            continue;

        old_corners.emplace_back(Corner(candidate->pts[i].x, candidate->pts[i].y, 0, 0));
    }

    Corners new_corners;
    fast_detector_->detect(candidate->frame->images(), new_corners, old_corners, FrameCandidate::size/0.85, Config::fastMinEigen());

    //! check corner number of first image
    if(old_corners.size() + new_corners.size() < FrameCandidate::size)
    {
        LOG_IF(WARNING, verbose_) << "[INIT][0] No enough corners detected!!! "
                                  << new_corners.size() << "(new) + " << old_corners.size() << "(old) < "
                                  << FrameCandidate::size;
        return RESET;
    }

    const int needed = FrameCandidate::size - old_corners.size();
    std::nth_element(new_corners.begin(), new_corners.begin()+needed, new_corners.end(),
                     [](const Corner &c1, const Corner &c2){ return c1.score > c2.score; });

    new_corners.resize(needed);
    auto new_corners_itr = new_corners.begin();
    for(int i = 0; i < FrameCandidate::size && new_corners_itr != new_corners.end(); i++)
    {
        if(candidate->idx[i] >= 0) //特征点被检测到的帧的id
            continue;

        const Corner &corner = *new_corners_itr;
        candidate->idx[i] = candidate->frame->id_;
        candidate->level[i] = corner.level;
        candidate->pts[i].x = corner.x;
        candidate->pts[i].y = corner.y;
        new_corners_itr++;
    }

    LOG_IF(INFO, verbose_) << "[INIT][0] New Detect corners: " << new_corners.size() << " for frame: " << candidate->frame->id_;

    return SUCCESS;
}

bool Initializer::changeReference(int buffer_offset)
{
    if(buffer_offset <= 0)
        return false;

    const int buffer_size = frame_buffer_.size();
    buffer_offset = MIN(buffer_offset, buffer_size-1);
    for(int i = 0; i < buffer_offset; i++)
        frame_buffer_.pop_front();

    cand_ref_ = frame_buffer_.front();
    cand_ref_->createFts();
    LOG_IF(INFO, verbose_) << "[INIT][*] Change reference frame to " << cand_ref_->frame->id_
                           << " offset: " << buffer_offset << " / " << buffer_size;

    //! change reference index in current
    const int64_t ref_index = cand_ref_->frame->id_;
    std::vector<int64_t> &index = cand_cur_->idx;
    for(int i = 0; i < FrameCandidate::size; i++)
    {
        if(index[i] != -1 && index[i] < ref_index)
            index[i] = ref_index;
    }

    return true;
}

Initializer::Result Initializer::addImage(Frame::Ptr frame_cur)
{

    //! jh 添加帧，设置参考帧，从参考帧中提取角点，创建pts，

    //! create first candidate
    if(frame_buffer_.empty())
    {
        frame_buffer_.push_back(FrameCandidate::create(frame_cur)); //当前帧作为候选帧
        cand_ref_ = frame_buffer_.front();
        createNewCorners(cand_ref_);///提取角点 pts
        cand_ref_->createFts();///去除畸变 fts
        cand_last_ = frame_buffer_.back();
        cand_cur_ = frame_buffer_.back();

        return READY;
    }

    //! finish last frame corner detect
    cand_last_ = frame_buffer_.back();
    int outlier_num = std::count(cand_last_->idx.begin(), cand_last_->idx.end(), -1);/// 如果上一帧的角点不足1000，则提取特征点，保证特征点的数目
    if(outlier_num > 0)
        createNewCorners(cand_last_);
    //! create current candidate from last
    cand_cur_ = FrameCandidate::create(frame_cur, cand_last_);
    frame_buffer_.push_back(cand_cur_);


    LOG_IF(INFO, verbose_) << "[INIT][*] Ref: " << cand_ref_->frame->id_
                           << ", Lst: " << cand_last_->frame->id_
                           << ", Cur: " << cand_cur_->frame->id_;


    double t1 = (double)cv::getTickCount();

    //! [1] KLT tracking
    const bool backward_check = true;
    cand_cur_->getInliers(inliers_); ///检测角点
    static cv::TermCriteria termcrit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.001);
    utils::kltTrack(cand_last_->frame->opticalImages(), cand_cur_->frame->opticalImages(), Frame::optical_win_size_,
                    cand_last_->pts, cand_cur_->pts, inliers_, termcrit, backward_check, true);
    cand_cur_->updateInliers(inliers_);
    //! if track too little corners in reference, then change reference
    const int offset = cand_cur_->checkTracking(cand_ref_->frame->id_, cand_cur_->frame->id_, Config::initMinTracked());
    //! return if the buffer is empty
    changeReference(offset);
    if(cand_ref_->frame->id_ == cand_cur_->frame->id_)
        return READY;

    cand_cur_->getMatch(inliers_, cand_ref_->frame->id_);
    calcDisparity(cand_ref_->pts, cand_cur_->pts, inliers_, disparities_);
    double aver=0;
    for(auto it:disparities_)
    {
        aver+=it.second;
    }
    aver/=disparities_.size();
    LOG_IF(INFO, verbose_) << "[INIT][1] KLT tracking points: " << disparities_.size();
//    if(disparities_.size() < Config::initMinTracked()) return RESET;

    //!BY JH 如果刚开始视差太小，视为刚开始时不动，就reset
    if(aver<5||aver>50) return RESET;

    //! 如果10帧的视差的还不满足要求，就reset，越小越严格
    if(frame_buffer_.size()>10) return RESET;


    double t2 = (double)cv::getTickCount();

    //! [2] calculate disparities
    std::vector<std::pair<int, float> > disparities_temp = disparities_;
    std::sort(disparities_temp.begin(), disparities_temp.end(),
              [](const std::pair<int, float> &disp1, const std::pair<int, float> &disp2){return disp1.second > disp2.second;});
    float disparity = disparities_temp.at(disparities_temp.size()/2).second;
    //! remove outliers
    int outliers = 0;
    if(!backward_check)
    {
        float max_disparity = disparities_temp.at(disparities_temp.size() * 1 / 5).second * 2;
        for(size_t i = 0; i < disparities_.size(); ++i)
        {
            if(disparities_[i].second > max_disparity)
            {
                const int id = disparities_[i].first;
                inliers_[id] = false;
                cand_cur_->idx[id] = -1;
                outliers++;
            }
        }

        disparity = disparities_temp.at((disparities_temp.size()-outliers)/2).second;
        cand_cur_->updateInliers(inliers_);
    }

    LOG_IF(INFO, verbose_) << "[INIT][2] Avage disparity: " << disparity << " with outliers: " << outliers;
    if(disparity < Config::initMinDisparity()) return FAILURE;

    double t3 = (double)cv::getTickCount();

    //todo jh 根据深度的方差判断是计算F矩阵还是E矩阵

    //! [3] geometry check by F matrix
    //! find fundamental matrix
    Matrix3d E;
    cand_cur_->createFts(); //! get undistorted points
    bool succeed = utils::Fundamental::findFundamentalMat(cand_ref_->fts, cand_cur_->fts, E, inliers_,
                                                          Config::imagePixelUnSigma2(), Config::initMaxRansacIters(), true);

    cand_cur_->updateInliers(inliers_);
    int inliers_count = std::count(inliers_.begin(), inliers_.end(), true);
    LOG_IF(INFO, verbose_) << "[INIT][3] Inliers after epipolar geometry check: " << inliers_count;
    if(!succeed || inliers_count < Config::initMinInliers()) return FAILURE;

    double t4 = (double)cv::getTickCount();

    //! [4] cheirality check
    Matrix3d R1, R2;
    Vector3d t;
    utils::Fundamental::decomposeEssentialMat(E, R1, R2, t);

    Matrix3d K = Matrix3d::Identity(3,3);
    succeed = findBestRT(R1, R2, t, K, K, cand_ref_->fts, cand_cur_->fts, inliers_, p3ds_, T_);
    if(!succeed) return FAILURE;

    /// by jh
    if(frame_buffer_.size()<3) return FAILURE;


    cand_cur_->updateInliers(inliers_);
    LOG_IF(INFO, verbose_) << "[INIT][4] Inliers after cheirality check: " << std::count(inliers_.begin(), inliers_.end(), true);

    double t5 = (double)cv::getTickCount();

    //! [5] reprojective check
    inliers_count = checkReprejectErr(cand_ref_->pts, cand_cur_->pts, cand_ref_->fts, cand_cur_->fts, T_, inliers_, p3ds_,
                                      Config::imagePixelUnSigma2()*4);
    cand_cur_->updateInliers(inliers_);
    LOG_IF(INFO, verbose_) << "[INIT][5] Inliers after reprojective check: " << inliers_count;
    if(inliers_count < Config::initMinInliers()) return FAILURE;

    double t6 = (double)cv::getTickCount();
    LOG_IF(WARNING, verbose_) << "[INIT][*] Time: " << (t2-t1)/cv::getTickFrequency() << " "
                              << (t3-t2)/cv::getTickFrequency() << " "
                              << (t4-t3)/cv::getTickFrequency() << " "
                              << (t5-t4)/cv::getTickFrequency() << " "
                              << (t6-t5)/cv::getTickFrequency();

    finished_ = true;

    return SUCCESS;
}

    Initializer::Result Initializer::reAddImage(Frame::Ptr frame_cur)
    {

        //! create first candidate
        if(frame_buffer_.empty())
        {
            frame_buffer_.push_back(FrameCandidate::create(frame_cur)); //当前帧作为候选帧
            cand_ref_ = frame_buffer_.front();
            createNewCorners(cand_ref_);///提取角点 pts
            cand_ref_->createFts();///去除畸变 fts
            cand_last_ = frame_buffer_.back();
            cand_cur_ = frame_buffer_.back();

            return READY;
        }

        //! finish last frame corner detect
        cand_last_ = frame_buffer_.back();
        int outlier_num = std::count(cand_last_->idx.begin(), cand_last_->idx.end(), -1);/// 如果上一帧的角点不足1000，则提取特征点，保证特征点的数目
        if(outlier_num > 0)
            createNewCorners(cand_last_);
        //! create current candidate from last
        cand_cur_ = FrameCandidate::create(frame_cur, cand_last_);
        frame_buffer_.push_back(cand_cur_);


        LOG_IF(INFO, verbose_) << "[INIT][*] Ref: " << cand_ref_->frame->id_
                               << ", Lst: " << cand_last_->frame->id_
                               << ", Cur: " << cand_cur_->frame->id_;


        double t1 = (double)cv::getTickCount();

        //! [1] KLT tracking
        const bool backward_check = true;
        cand_cur_->getInliers(inliers_); ///检测角点
        static cv::TermCriteria termcrit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.001);
        utils::kltTrack(cand_last_->frame->opticalImages(), cand_cur_->frame->opticalImages(), Frame::optical_win_size_,
                        cand_last_->pts, cand_cur_->pts, inliers_, termcrit, backward_check, true);
        cand_cur_->updateInliers(inliers_);
        //! if track too little corners in reference, then change reference
        const int offset = cand_cur_->checkTracking(cand_ref_->frame->id_, cand_cur_->frame->id_, Config::initMinTracked());
        //! return if the buffer is empty
        changeReference(offset);
        if(cand_ref_->frame->id_ == cand_cur_->frame->id_)
            return READY;

        cand_cur_->getMatch(inliers_, cand_ref_->frame->id_);
        calcDisparity(cand_ref_->pts, cand_cur_->pts, inliers_, disparities_);
        double aver=0;
        for(auto it:disparities_)
        {
            aver+=it.second;
        }
        aver/=disparities_.size();
        LOG_IF(INFO, verbose_) << "[INIT][1] KLT tracking points: " << disparities_.size();
//    if(disparities_.size() < Config::initMinTracked()) return RESET;

        //!BY JH 如果刚开始视差太小，视为刚开始时不动，就reset
        if(aver<5||aver>50) return RESET;

        //! 如果10帧的视差的还不满足要求，就reset，越小越严格
        if(frame_buffer_.size()>10) return RESET;


        double t2 = (double)cv::getTickCount();

        //! [2] calculate disparities
        std::vector<std::pair<int, float> > disparities_temp = disparities_;
        std::sort(disparities_temp.begin(), disparities_temp.end(),
                  [](const std::pair<int, float> &disp1, const std::pair<int, float> &disp2){return disp1.second > disp2.second;});
        float disparity = disparities_temp.at(disparities_temp.size()/2).second;
        //! remove outliers
        int outliers = 0;
        if(!backward_check)
        {
            float max_disparity = disparities_temp.at(disparities_temp.size() * 1 / 5).second * 2;
            for(size_t i = 0; i < disparities_.size(); ++i)
            {
                if(disparities_[i].second > max_disparity)
                {
                    const int id = disparities_[i].first;
                    inliers_[id] = false;
                    cand_cur_->idx[id] = -1;
                    outliers++;
                }
            }

            disparity = disparities_temp.at((disparities_temp.size()-outliers)/2).second;
            cand_cur_->updateInliers(inliers_);
        }

        LOG_IF(INFO, verbose_) << "[INIT][2] Avage disparity: " << disparity << " with outliers: " << outliers;
        if(disparity < Config::initMinDisparity()) return FAILURE;

        double t3 = (double)cv::getTickCount();

        //todo jh 根据深度的方差判断是计算F矩阵还是E矩阵

        //! [3] geometry check by F matrix
        //! find fundamental matrix
        Matrix3d E;
        cand_cur_->createFts(); //! get undistorted points
        bool succeed = utils::Fundamental::findFundamentalMat(cand_ref_->fts, cand_cur_->fts, E, inliers_,
                                                              Config::imagePixelUnSigma2(), Config::initMaxRansacIters(), true);

        cand_cur_->updateInliers(inliers_);
        int inliers_count = std::count(inliers_.begin(), inliers_.end(), true);
        LOG_IF(INFO, verbose_) << "[INIT][3] Inliers after epipolar geometry check: " << inliers_count;
        if(!succeed || inliers_count < Config::initMinInliers()) return FAILURE;

        double t4 = (double)cv::getTickCount();

        //! [4] cheirality check
        Matrix3d R1, R2;
        Vector3d t;
        utils::Fundamental::decomposeEssentialMat(E, R1, R2, t);

        Matrix3d K = Matrix3d::Identity(3,3);
        succeed = findBestRT(R1, R2, t, K, K, cand_ref_->fts, cand_cur_->fts, inliers_, p3ds_, T_);
        if(!succeed) return FAILURE;

        /// by jh
        if(frame_buffer_.size()<3) return FAILURE;


        cand_cur_->updateInliers(inliers_);
        LOG_IF(INFO, verbose_) << "[INIT][4] Inliers after cheirality check: " << std::count(inliers_.begin(), inliers_.end(), true);

        double t5 = (double)cv::getTickCount();

        //! [5] reprojective check
        inliers_count = checkReprejectErr(cand_ref_->pts, cand_cur_->pts, cand_ref_->fts, cand_cur_->fts, T_, inliers_, p3ds_,
                                          Config::imagePixelUnSigma2()*4);
        cand_cur_->updateInliers(inliers_);
        LOG_IF(INFO, verbose_) << "[INIT][5] Inliers after reprojective check: " << inliers_count;
        if(inliers_count < Config::initMinInliers()) return FAILURE;


        finished_ = true;

        return SUCCESS;
    }

void Initializer::createInitalMap(double map_scale)
{
    LOG_ASSERT(finished_) << "[INIT][6] Initialization is not finished!";

    //! [6] create inital map
    const size_t N = cand_ref_->pts.size();

    //! calculate mean depth
    double mean_depth = 0.0;
    size_t count = 0;
    for(size_t i = 0; i < N; ++i)
    {
        if(inliers_[i])
        {
            mean_depth += p3ds_[i][2];
            count++;
        }
    }
    double scale = map_scale*count/mean_depth; /// 地图的平均深度为1

    //! rescale frame pose
    SE3d T_cur_from_ref(T_.topLeftCorner(3,3), T_.rightCols(1));
    SE3d T_ref_from_cur = T_cur_from_ref.inverse();
    T_ref_from_cur.translation() = T_ref_from_cur.translation() * scale;
//    T_ref_from_cur.translation() = T_ref_from_cur.translation();
    cand_ref_->frame->setPose(Matrix3d::Identity(), Vector3d::Zero()); ///首帧作为世界坐标系
    cand_cur_->frame->setPose(T_ref_from_cur);

    //! create and rescale map points
    for(size_t i = 0; i < N; ++i)
    {
        if(!inliers_[i])
            continue;

        Vector2d px_ref(cand_ref_->pts[i].x, cand_ref_->pts[i].y);
        Vector2d px_cur(cand_cur_->pts[i].x, cand_cur_->pts[i].y);
        Vector3d ft_ref(cand_ref_->fts[i].x, cand_ref_->fts[i].y, 1);
        Vector3d ft_cur(cand_cur_->fts[i].x, cand_cur_->fts[i].y, 1);

        MapPoint::Ptr mpt = ssvo::MapPoint::create(p3ds_[i]*scale);

        LOG_ASSERT(cand_cur_->level[i] >= 0 && cand_cur_->level[i] <= cand_cur_->frame->max_level_) << "Error in level, index:" << i;
        LOG_ASSERT(cand_ref_->level[i] >= 0 && cand_ref_->level[i] <= cand_ref_->frame->max_level_) << "Error in level, index:" << i;
        Feature::Ptr feature_ref = Feature::create(px_ref, ft_ref, cand_ref_->level[i], mpt);
        Feature::Ptr feature_cur = Feature::create(px_cur, ft_cur, cand_cur_->level[i], mpt);

        cand_ref_->frame->addFeature(feature_ref);
        cand_cur_->frame->addFeature(feature_cur);
    }
}

    void Initializer::reCreateInitalMap(double scale)
    {
        const size_t N = cand_ref_->pts.size();

        //! rescale frame pose
        SE3d T_cur_from_ref(T_.topLeftCorner(3,3), T_.rightCols(1));
        SE3d T_ref_from_cur = T_cur_from_ref.inverse();
        T_ref_from_cur.translation() = T_ref_from_cur.translation() * scale;

        cand_ref_->frame->setPose(Matrix3d::Identity(), Vector3d::Zero()); ///首帧作为世界坐标系
        cand_cur_->frame->setPose(T_ref_from_cur);

        //! create and rescale map points
        for(size_t i = 0; i < N; ++i)
        {
            if(!inliers_[i])
                continue;

            Vector2d px_ref(cand_ref_->pts[i].x, cand_ref_->pts[i].y);
            Vector2d px_cur(cand_cur_->pts[i].x, cand_cur_->pts[i].y);
            Vector3d ft_ref(cand_ref_->fts[i].x, cand_ref_->fts[i].y, 1);
            Vector3d ft_cur(cand_cur_->fts[i].x, cand_cur_->fts[i].y, 1);

            MapPoint::Ptr mpt = ssvo::MapPoint::create(p3ds_[i]*scale);

            LOG_ASSERT(cand_cur_->level[i] >= 0 && cand_cur_->level[i] <= cand_cur_->frame->max_level_) << "Error in level, index:" << i;
            LOG_ASSERT(cand_ref_->level[i] >= 0 && cand_ref_->level[i] <= cand_ref_->frame->max_level_) << "Error in level, index:" << i;
            Feature::Ptr feature_ref = Feature::create(px_ref, ft_ref, cand_ref_->level[i], mpt);
            Feature::Ptr feature_cur = Feature::create(px_cur, ft_cur, cand_cur_->level[i], mpt);

            cand_ref_->frame->addFeature(feature_ref);
            cand_cur_->frame->addFeature(feature_cur);
        }
    }

void Initializer::getTrackedPoints(std::vector<cv::Point2f>& pts_ref, std::vector<cv::Point2f>& pts_cur) const
{
    const int ref_id = cand_ref_->frame->id_;
    const size_t track_points = std::count(cand_cur_->idx.begin(), cand_cur_->idx.end(), ref_id);

    pts_ref.clear();
    pts_cur.clear();
    pts_ref.reserve(track_points);
    pts_cur.reserve(track_points);

    const std::vector<int64_t> &index = cand_cur_->idx;
    for(int i = 0; i < FrameCandidate::size; ++i)
    {
        if(index[i] < 0 || index[i] > ref_id)
            continue;

        pts_ref.push_back(cand_ref_->pts[i]);
        pts_cur.push_back(cand_cur_->pts[i]);
    }
}

void Initializer::drowOpticalFlow(cv::Mat &dst) const
{
    std::vector<cv::Point2f> pts_ref, pts_cur;
    getTrackedPoints(pts_ref, pts_cur);

    const cv::Mat src = cand_cur_->frame->getImage(0);
    cv::cvtColor(src, dst, CV_GRAY2RGB);
    for(size_t i=0; i<pts_ref.size();i++)
    {
        cv::circle(dst, pts_cur[i], 2, cv::Scalar(0, 255, 0));
        cv::line(dst, 2*pts_cur[i]-pts_ref[i], pts_cur[i], cv::Scalar(0, 255, 0));
    }
}

void Initializer::drowOpticalFlowMatch(cv::Mat &dst) const
{
    const cv::Mat &cur = cand_cur_->frame->getImage(0);
    const cv::Mat &ref = cand_ref_->frame->getImage(0);
    const int cols = cur.cols;
    const int rows = cur.rows;
    dst = cv::Mat(rows, cols*2, CV_8UC3);
    cv::cvtColor(cur, dst.colRange(0, cols), CV_GRAY2RGB);
    cv::cvtColor(ref, dst.colRange(cols, 2*cols), CV_GRAY2RGB);

    const int N = FrameCandidate::size;
    const int ref_id = cand_ref_->frame->id_;
    const std::vector<int64_t> &index = cand_cur_->idx;
    for(int i = 0; i < N; ++i)
    {
        if(index[i] < 0 || index[i] > ref_id)
            continue;

        cv::RNG rng(i);
        cv::Scalar color = cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
        cv::line(dst, cand_cur_->pts[i], cand_ref_->pts[i]+cv::Point2f(cols, 0), color);
    }
}

void Initializer::calcDisparity(const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2,
                                const std::vector<bool> &mask, std::vector<std::pair<int, float> >& disparities)
{
    const size_t N = pts1.size();
    assert(N == pts2.size());

    const size_t out_size= std::count(mask.begin(), mask.end(), true);
    disparities.clear();
    disparities.reserve(out_size);
    for(size_t i = 0; i < N; i++)
    {
        if(!mask[i])
            continue;

        float dx = pts2[i].x - pts1[i].x;
        float dy = pts2[i].y - pts1[i].y;
        disparities.emplace_back(std::make_pair(i, std::sqrt(dx*dx + dy*dy)));
    }
}

bool Initializer::findBestRT(const Matrix3d& R1, const Matrix3d& R2, const Vector3d& t,
                             const Matrix3d& K1, const Matrix3d& K2,
                             const std::vector<cv::Point2d>& fts1, const std::vector<cv::Point2d>& fts2,
                             std::vector<bool>& mask, std::vector<Vector3d>& P3Ds,
                             Matrix<double, 3, 4>& T)
{
    const size_t N = fts1.size();
    assert(N == fts2.size());
    if(mask.empty())
        mask.resize(fts1.size(), true);

    //! P = K[R|t]
    MatrixXd P0(3, 4), P1(3, 4), P2(3, 4), P3(3, 4), P4(3, 4);
    MatrixXd T1(3, 4), T2(3, 4), T3(3, 4), T4(3, 4);

    //! P0 = K[I|0]
    P0 = K1 * MatrixXd::Identity(3,4);

    //! P1 = K[R1|t]
    T1.block(0,0,3,3) = R1; T1.col(3) = t;
    P1 = K2 * T1;

    //! P2 = K[R2|t]
    T2.block(0,0,3,3) = R2; T2.col(3) = t;
    P2 = K2 * T2;

    //! P3 = K[R1|-t]
    T3.block(0,0,3,3) = R1; T3.col(3) = -t;
    P3 = K2 * T3;

    //! P4 = K[R2|-t]
    T4.block(0,0,3,3) = R2; T4.col(3) = -t;
    P4 = K2 * T4;

    //! Do the cheirality check, and remove points too far away
    const double max_dist = 50.0;
    std::vector<Vector3d> P3Ds1(N, Vector3d(0,0,0));
    std::vector<Vector3d> P3Ds2(N, Vector3d(0,0,0));
    std::vector<Vector3d> P3Ds3(N, Vector3d(0,0,0));
    std::vector<Vector3d> P3Ds4(N, Vector3d(0,0,0));
    std::vector<bool> mask1 = mask;
    std::vector<bool> mask2 = mask;
    std::vector<bool> mask3 = mask;
    std::vector<bool> mask4 = mask;

    int nGood0 = std::count(mask.begin(), mask.end(), true);
    int nGood1 = 0;
    int nGood2 = 0;
    int nGood3 = 0;
    int nGood4 = 0;

    Vector4d P3D;
    Vector3d T_P3D;
    for(size_t i = 0; i < N; ++i)
    {
        if(!mask[i])
            continue;

        const cv::Point2d ft1 = fts1[i];
        const cv::Point2d ft2 = fts2[i];

        //! P0 & P1
        triangulate(P0, P1, ft1, ft2, P3D);
        P3Ds1[i] = P3D.head(3);
        T_P3D = T1 * P3D;
        if(P3D[2] < 0 || T_P3D[2] < 0 || P3D[2] > max_dist || T_P3D[2] > max_dist)
            mask1[i] = 0;
        else
            nGood1++;

        //! P0 & P2
        triangulate(P0, P2, ft1, ft2, P3D);
        P3Ds2[i] = P3D.head(3);
        T_P3D = T2 * P3D;
        if(P3D[2] < 0 || T_P3D[2] < 0 || P3D[2] > max_dist || T_P3D[2] > max_dist)
            mask2[i] = 0;
        else
            nGood2++;

        //! P0 & P3
        triangulate(P0, P3, ft1, ft2, P3D);
        P3Ds3[i] = P3D.head(3);
        T_P3D = T3 * P3D;
        if(P3D[2] < 0 || T_P3D[2] < 0 || P3D[2] > max_dist || T_P3D[2] > max_dist)
            mask3[i] = 0;
        else
            nGood3++;

        //! P0 & P4
        triangulate(P0, P4, ft1, ft2, P3D);
        P3Ds4[i] = P3D.head(3);
        T_P3D = T4 * P3D;
        if(P3D[2] < 0 || T_P3D[2] < 0 || P3D[2] > max_dist || T_P3D[2] > max_dist)
            mask4[i] = 0;
        else
            nGood4++;
    }

    int maxGood = MAX(MAX(nGood1, nGood2), MAX(nGood3, nGood4));

    int goodCount = 0;
    if(nGood1 > 0.7*maxGood)
        goodCount++;
    if(nGood2 > 0.7*maxGood)
        goodCount++;
    if(nGood3 > 0.7*maxGood)
        goodCount++;
    if(nGood4 > 0.7*maxGood)
        goodCount++;

    //! for normal, other count is 0
    if(maxGood < 0.9*nGood0 || goodCount > 1)
    {
        LOG(WARNING) << "[INIT] Cheirality check failed! " << nGood1 << " " << nGood2 << " " << nGood3 << " " << nGood4
                     << ", " << nGood0;
        return false;
    }

    if(maxGood == nGood1)
    {
        mask = mask1;
        P3Ds = P3Ds1;
        T = T1;
    }
    else if(maxGood == nGood2)
    {
        mask = mask2;
        P3Ds = P3Ds2;
        T = T2;
    }
    else if(maxGood == nGood3)
    {
        mask = mask3;
        P3Ds = P3Ds3;
        T = T3;
    }
    else if(maxGood == nGood4)
    {
        mask = mask4;
        P3Ds = P3Ds4;
        T = T4;
    }

    cout<<"T:"<<T<<endl;
//    waitKey(0);
    return true;
}

int Initializer::checkReprejectErr(const std::vector<cv::Point2f>& pts_ref, const std::vector<cv::Point2f>& pts_cur,
                                   const  std::vector<cv::Point2d>& fts_ref, const std::vector<cv::Point2d>& fts_cur,
                                   const Matrix<double, 3, 4>& T, std::vector<bool>& mask, std::vector<Vector3d>& p3ds,
                                   const double sigma2)
{
    const int size = pts_ref.size();

    int inliers_count = 0;

    for(int i = 0; i < size; ++i)
    {
        if(!mask[i])
            continue;

        const Vector3d p3d1 = p3ds[i];
        const double X = p3d1[0];
        const double Y = p3d1[1];
        const double Z = p3d1[2];
        double x1 = X / Z;
        double y1 = Y / Z;
        double dx1 = fts_ref[i].x - x1;
        double dy1 = fts_ref[i].y - y1;
        double err1 = dx1*dx1 + dy1*dy1;

        if(err1 > sigma2)
        {
            mask[i] = false;
            continue;
        }

        Vector3d p3d2 = T * Vector4d(X,Y,Z,1);
        double x2 = p3d2[0] / p3d2[2];
        double y2 = p3d2[1] / p3d2[2];
        double dx2 = fts_cur[i].x - x2;
        double dy2 = fts_cur[i].y - y2;
        double err2 = dx2*dx2 + dy2*dy2;

        if(err2 > sigma2)
        {
            mask[i] = false;
            continue;
        }

        inliers_count++;
        p3ds[i] = Vector3d(X, Y, Z);
    }

    return inliers_count;
}

void Initializer::triangulate(const Matrix<double, 3, 4>& P1, const Matrix<double, 3, 4>& P2,
                              const cv::Point2d& ft1, const cv::Point2d& ft2, Vector4d& P3D)
{
    MatrixXd A(4,4);
    A.row(0) = ft1.x*P1.row(2)-P1.row(0);
    A.row(1) = ft1.y*P1.row(2)-P1.row(1);
    A.row(2) = ft2.x*P2.row(2)-P2.row(0);
    A.row(3) = ft2.y*P2.row(2)-P2.row(1);

    JacobiSVD<MatrixXd> svd(A, ComputeThinV);
    MatrixXd V = svd.matrixV();

    P3D = V.col(3);
    P3D = P3D/P3D(3);
}

}


