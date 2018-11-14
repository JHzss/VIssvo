#ifndef _FEATURE_TRACKER_HPP_
#define _FEATURE_TRACKER_HPP_


#include "global.hpp"
#include "feature_detector.hpp"
#include "map.hpp"

namespace ssvo
{

class FeatureTracker : public noncopyable
{
//    struct Candidate {
//        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//        MapPoint::Ptr pt;    //!< 3D point.
//        Vector2d px;         //!< projected 2D pixel location.
//        Candidate(MapPoint::Ptr pt, Vector2d& px) : pt(pt), px(px) {}
//        Candidate(){}
//    };
//
//    struct Grid {
//        typedef std::list<Candidate, aligned_allocator<Candidate> > Cell;
//        int grid_size;
//        int grid_n_cols;
//        int grid_n_rows;
//        std::vector<Cell*> cells;
//        std::vector<int> grid_order;
//        std::vector<bool> occupied;
//    };

public:
    typedef std::shared_ptr<FeatureTracker> Ptr;

    int reprojectLoaclMap(const Frame::Ptr &frame);


        int reprojectLoaclMap(const Frame::Ptr &frame, std::set<KeyFrame::Ptr> &local_keyframes);

    static int reprojectMapPoint(const Frame::Ptr &frame, const MapPoint::Ptr& mpt, Vector2d &px_cur, int &level_cur,
                                  const int max_iterations = 30, const double epslion = 0.01, const double threshold = 4.0, bool verbose = false);

    static bool trackFeature(const Frame::Ptr &frame_ref, const Frame::Ptr &frame_cur, const Feature::Ptr &ft_ref,
                             Vector2d &px_cur, int &level_cur, const int max_iterations = 30, const double epslion = 0.01, const double threshold = 4.0, bool verbose = false);

    inline static FeatureTracker::Ptr create(int width, int height, int grid_size, int border, bool report = false, bool verbose = false)
    {return FeatureTracker::Ptr(new FeatureTracker(width, height, grid_size, border, report, verbose));}

private:

    FeatureTracker(int width, int height, int grid_size, int border, bool report = false, bool verbose = false);

    bool reprojectMapPointToCell(const Frame::Ptr &frame, const MapPoint::Ptr &point);

    bool matchMapPointsFromCell(const Frame::Ptr &frame, Grid<Feature::Ptr>::Cell &cell);

    int matchMapPointsFromLastFrame(const Frame::Ptr &frame_cur, const Frame::Ptr &frame_last);

private:

    struct Option{
        int border;
        int max_matches;
        int max_track_kfs;
        int num_align_iter;
        double max_align_epsilon;
        double max_align_error2;
    } options_;

    Grid<Feature::Ptr> grid_;
    std::vector<size_t> grid_order_;

    bool report_;
    bool verbose_;
    int total_project_;
};

}//! end of ssvo
#endif