#ifndef _KEYFRAME_HPP_
#define _KEYFRAME_HPP_

#include "global.hpp"
#include "frame.hpp"

namespace ssvo
{

class Map;

class KeyFrame: public Frame, public std::enable_shared_from_this<KeyFrame>
{
public:

    typedef std::shared_ptr<KeyFrame> Ptr;

    void updateConnections();

    void setBad();

    bool isBad();

    std::set<KeyFrame::Ptr> getConnectedKeyFrames(int num=-1, int min_fts = 0);

    std::set<KeyFrame::Ptr> getSubConnectedKeyFrames(int num=-1);

    std::set<KeyFrame::Ptr> getOrderedSubConnectedKeyFrames();

    const ImgPyr opticalImages() const = delete;    //! disable this function

    inline static KeyFrame::Ptr create(const Frame::Ptr frame)
    { return Ptr(new KeyFrame(frame)); }

private:

    KeyFrame(const Frame::Ptr frame);

    void addConnection(const KeyFrame::Ptr &kf, const int weight);

    void updateOrderedConnections();

    void removeConnection(const KeyFrame::Ptr &kf);

public:

    static uint64_t next_id_;

    const uint64_t frame_id_;

    std::vector<Feature::Ptr> dbow_fts_;
    cv::Mat descriptors_;
    unsigned int dbow_Id_;

    DBoW3::BowVector bow_vec_;

    DBoW3::FeatureVector feat_vec_;

private:

    float grid_col_inv_;
    float grid_row_inv_;

    size_t N_;
    std::unordered_set<size_t> seeds_created_;

//    std::vector<std::size_t> grid_[GRID_ROWS][GRID_COLS];



    std::map<KeyFrame::Ptr, int> connectedKeyFrames_;

    std::multimap<int, KeyFrame::Ptr> orderedConnectedKeyFrames_;

    bool isBad_;

    std::mutex mutex_connection_;

public:
//    void conputeDescriptor(const BRIEF::Ptr &brief);

//    void computeBoW(const DBoW3::Vocabulary& vocabulary);

};

}

#endif