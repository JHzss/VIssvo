#ifndef _MAP_HPP_
#define _MAP_HPP_

#include "map_point.hpp"
#include "keyframe.hpp"
#include "global.hpp"

namespace ssvo{

class LocalMapper;
class DepthFilter;

class Map: public noncopyable
{
    friend class LocalMapper;
    friend class DepthFilter;

public:
    typedef std::shared_ptr<Map> Ptr;

    KeyFrame::Ptr getKeyFrame(uint64_t id);

    std::vector<KeyFrame::Ptr> getAllKeyFrames();

    std::vector<MapPoint::Ptr> getAllMapPoints();

    uint64_t KeyFramesInMap();

    uint64_t MapPointsInMap();


    void applyScaleCorrect(double scale);

private:

    void clear();

    bool insertKeyFrame(const KeyFrame::Ptr &kf);

    void removeKeyFrame(const KeyFrame::Ptr &kf);

    void insertMapPoint(const MapPoint::Ptr &mpt);

//    void removeMapPoint(const MapPoint::Ptr &mpt);

    inline static Map::Ptr create() {return Map::Ptr(new Map());}

public:

        void removeMapPoint(const MapPoint::Ptr &mpt);

    std::set<MapPoint::Ptr> removed_mpts_;

private:

    std::unordered_map<uint64_t, KeyFrame::Ptr> kfs_;

    std::unordered_map<uint64_t, MapPoint::Ptr> mpts_;

    std::mutex mutex_kf_;
    std::mutex mutex_mpt_;
};

}

#endif