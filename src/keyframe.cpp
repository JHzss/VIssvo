#include "config.hpp"
#include "map.hpp"
#include "keyframe.hpp"

namespace ssvo{

uint64_t KeyFrame::next_id_ = 0;

KeyFrame::KeyFrame(const Frame::Ptr frame):
    Frame(frame->images(), next_id_++, frame->timestamp_, frame->cam_), frame_id_(frame->id_), isBad_(false)
{
    preintegration = frame->preintegration;
    std::memcpy(bgba,frame->bgba,6);
    mpt_fts_ = frame->features();
    setRefKeyFrame(frame->getRefKeyFrame());
    setPose(frame->pose());
}
void KeyFrame::updateConnections()
{
    if(isBad())
        return;

    Features fts;
    {
        std::lock_guard<std::mutex> lock(mutex_feature_);
        for(const auto &it : mpt_fts_)
            fts.push_back(it.second);
    }

    std::map<KeyFrame::Ptr, int> connection_counter;

    for(const Feature::Ptr &ft : fts)
    {
        const MapPoint::Ptr &mpt = ft->mpt_;

        if(mpt->isBad())
        {
            removeFeature(ft);
            continue;
        }

        const std::map<KeyFrame::Ptr, Feature::Ptr> observations = mpt->getObservations();
        for(const auto &obs : observations)
        {
            if(obs.first->id_ == id_)
                continue;
            connection_counter[obs.first]++;
        }
    }

    if(connection_counter.empty())
    {
        setBad();
        return;
    }

    // TODO how to select proper connections
    int connection_threshold = Config::minConnectionObservations();

    KeyFrame::Ptr best_unfit_keyframe;
    int best_unfit_connections = 0;
    std::vector<std::pair<int, KeyFrame::Ptr> > weight_connections;
    for(const auto &obs : connection_counter)
    {
        if(obs.second < connection_threshold)
        {
            best_unfit_keyframe = obs.first;
            best_unfit_connections = obs.second;
        }
        else
        {
            obs.first->addConnection(shared_from_this(), obs.second);
            weight_connections.emplace_back(std::make_pair(obs.second, obs.first));
        }
    }

    if(weight_connections.empty())
    {
        best_unfit_keyframe->addConnection(shared_from_this(), best_unfit_connections);
        weight_connections.emplace_back(std::make_pair(best_unfit_connections, best_unfit_keyframe));
    }

    //! sort by weight
    std::sort(weight_connections.begin(), weight_connections.end(),
              [](const std::pair<int, KeyFrame::Ptr> &a, const std::pair<int, KeyFrame::Ptr> &b){ return a.first > b.first; });

    //! update
    {
        std::lock_guard<std::mutex> lock(mutex_connection_);

        connectedKeyFrames_.clear();
        for(const auto &item : weight_connections)
        {
            connectedKeyFrames_.insert(std::make_pair(item.second, item.first));
        }

        orderedConnectedKeyFrames_ = std::multimap<int, KeyFrame::Ptr>(weight_connections.begin(), weight_connections.end());
    }
}

std::set<KeyFrame::Ptr> KeyFrame::getConnectedKeyFrames(int num, int min_fts)
{
    std::lock_guard<std::mutex> lock(mutex_connection_);

    std::set<KeyFrame::Ptr> connected_keyframes;
    if(num == -1) num = (int) orderedConnectedKeyFrames_.size();

    int count = 0;
    const auto end = orderedConnectedKeyFrames_.rend();
    for(auto it = orderedConnectedKeyFrames_.rbegin(); it != end && it->first >= min_fts && count < num; it++, count++)
    {
        connected_keyframes.insert(it->second);
    }

    return connected_keyframes;
}

std::set<KeyFrame::Ptr> KeyFrame::getSubConnectedKeyFrames(int num)
{
    std::set<KeyFrame::Ptr> connected_keyframes = getConnectedKeyFrames();

    std::map<KeyFrame::Ptr, int> candidate_keyframes;
    for(const KeyFrame::Ptr &kf : connected_keyframes)
    {
        std::set<KeyFrame::Ptr> sub_connected_keyframe = kf->getConnectedKeyFrames();
        for(const KeyFrame::Ptr &sub_kf : sub_connected_keyframe)
        {
            if(connected_keyframes.count(sub_kf) || sub_kf == shared_from_this())
                continue;

            if(candidate_keyframes.count(sub_kf))
                candidate_keyframes.find(sub_kf)->second++;
            else
                candidate_keyframes.emplace(sub_kf, 1);
        }
    }

    std::set<KeyFrame::Ptr> sub_connected_keyframes;
    if(num == -1)
    {
        for(const auto &item : candidate_keyframes)
            sub_connected_keyframes.insert(item.first);

        return sub_connected_keyframes;
    }

    //! stort by order
    std::map<int, KeyFrame::Ptr, std::greater<int> > ordered_candidate_keyframes;
    for(const auto &item : candidate_keyframes)
    {
        ordered_candidate_keyframes.emplace(item.second, item.first);
    }

    //! get best (num) keyframes
    for(const auto &item : ordered_candidate_keyframes)
    {
        sub_connected_keyframes.insert(item.second);
        if(sub_connected_keyframes.size() >= num)
            break;
    }

    return sub_connected_keyframes;
}

void KeyFrame::setBad()
{
    if(id_ == 0)
        return;

    std::cout << "The keyframe " << id_ << " was set to be earased." << std::endl;

    std::unordered_map<MapPoint::Ptr, Feature::Ptr> mpt_fts;
    {
        std::lock_guard<std::mutex> lock(mutex_feature_);
        mpt_fts = mpt_fts_;
    }

    for(const auto &it : mpt_fts)
    {
        it.first->removeObservation(shared_from_this());
    }

    {
        std::lock_guard<std::mutex> lock(mutex_connection_);

        isBad_ = true;

        for(const auto &connect : connectedKeyFrames_)
        {
            connect.first->removeConnection(shared_from_this());
        }

        connectedKeyFrames_.clear();
        orderedConnectedKeyFrames_.clear();
        mpt_fts_.clear();
        seed_fts_.clear();
    }
    // TODO change refKF
}

bool KeyFrame::isBad()
{
    std::lock_guard<std::mutex> lock(mutex_connection_);
    return isBad_;
}

void KeyFrame::addConnection(const KeyFrame::Ptr &kf, const int weight)
{
    {
        std::lock_guard<std::mutex> lock(mutex_connection_);

        if(!connectedKeyFrames_.count(kf))
            connectedKeyFrames_[kf] = weight;
        else if(connectedKeyFrames_[kf] != weight)
            connectedKeyFrames_[kf] = weight;
        else
            return;
    }

    updateOrderedConnections();
}

void KeyFrame::updateOrderedConnections()
{
    std::lock_guard<std::mutex> lock(mutex_connection_);
    orderedConnectedKeyFrames_.clear();
    for(const auto &connect : connectedKeyFrames_)
    {
        auto it = orderedConnectedKeyFrames_.lower_bound(connect.second);
        orderedConnectedKeyFrames_.insert(it, std::pair<int, KeyFrame::Ptr>(connect.second, connect.first));
    }
}

void KeyFrame::removeConnection(const KeyFrame::Ptr &kf)
{
    {
        std::lock_guard<std::mutex> lock(mutex_connection_);
        if(connectedKeyFrames_.count(kf))
        {
            connectedKeyFrames_.erase(kf);
        }
    }

    updateOrderedConnections();
}

//    void KeyFrame::conputeDescriptor(const BRIEF::Ptr &brief)
//    {
//        std::vector<cv::KeyPoint> kps; kps.reserve(mpt_fts_.size());
//        for(auto mpt_ft : mpt_fts_)
//        {
//            const Feature::Ptr ft = mpt_ft.second;
//            kps.emplace_back(cv::KeyPoint(ft->corner_.x, ft->corner_.y, 31, -1, 0, ft->corner_.level));
//        }
//
//        cv::Mat _descriptors;
//        brief->compute(images(), kps, _descriptors);
//
//        descriptors_.reserve(_descriptors.rows);
//        for(int i = 0; i < _descriptors.rows; i++)
//            descriptors_.push_back(_descriptors.row(i));
//    }
//
//    void KeyFrame::computeBoW(const DBoW3::Vocabulary& vocabulary)
//    {
//        LOG_ASSERT(!descriptors_.empty()) << "Please use conputeDescriptor first!";
//        if(bow_vec_.empty())
//            vocabulary.transform(descriptors_, bow_vec_, feat_vec_, 4);
//    }

}