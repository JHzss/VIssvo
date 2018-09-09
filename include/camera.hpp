#ifndef _CAMERA_HPP_
#define _CAMERA_HPP_

#include <vector>
#include <memory>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

#include "global.hpp"

using namespace Eigen;
using namespace cv;

namespace ssvo {

// once created, never changed
class AbstractCamera : public noncopyable
{
public:

    enum Type {
        ABSTRACT    = -1,
        PINHOLE     = 0,
        ATAN        = 1
    };

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef std::shared_ptr<AbstractCamera> Ptr;

    AbstractCamera() {}

    AbstractCamera(int width, int height, Type type = ABSTRACT);

    AbstractCamera(int width, int height, double fx, double fy, double cx, double cy, Type type = ABSTRACT);

    virtual ~AbstractCamera() {};

    inline const int width() const { return width_; }

    inline const int height() const { return height_; }

    inline const double fx() const { return fx_; };

    inline const double fy() const { return fy_; };

    inline const double cx() const { return cx_; };

    inline const double cy() const { return cy_; };

    inline const Type type() const { return type_; }

    virtual Vector3d lift(const Vector2d& px) const;

    virtual Vector3d lift(double x, double y) const;

    virtual Vector2d project(const Vector3d& xyz) const;

    virtual Vector2d project(double x, double y) const;

    virtual void undistortPoints(const std::vector<cv::Point2f> &pts_dist, std::vector<cv::Point2f> &pts_udist) const;

    inline bool isInFrame(const Vector2i &obs, int boundary=0) const
    {
        if(obs[0] >= boundary && obs[0] < width() - boundary
            && obs[1] >= boundary && obs[1] < height() - boundary)
            return true;
        return false;
    }

    inline bool isInFrame(const Vector2i &obs, int boundary, int level) const
    {
        if(obs[0] >= boundary && obs[0] < (width() >> level) - boundary
            && obs[1] >= boundary && obs[1] < (height() >> level) - boundary)
            return true;
        return false;
    }

protected:
    int width_;
    int height_;
    double fx_, fy_, cx_, cy_;
    bool distortion_;
    Type type_;
};

class PinholeCamera : public AbstractCamera
{

public:

    typedef std::shared_ptr<PinholeCamera> Ptr;

    virtual Vector3d lift(const Vector2d& px) const;

    virtual Vector3d lift(double x, double y) const;

    virtual Vector2d project(const Vector3d& xyz) const;

    virtual Vector2d project(double x, double y) const;

    //! all undistort points are in the normlized plane
    virtual void undistortPoints(const std::vector<cv::Point2f> &pts_dist, std::vector<cv::Point2f> &pts_udist) const;

    inline static PinholeCamera::Ptr create(int width, int height, double fx, double fy, double cx, double cy, double k1 = 0.0, double k2 = 0.0, double p1 = 0.0, double p2 = 0.0)
    {return PinholeCamera::Ptr(new PinholeCamera(width, height, fx, fy, cx, cy, k1, k2, p1, p2));}

    inline static PinholeCamera::Ptr create(int width, int height, const cv::Mat& K, const cv::Mat& D)
    {return PinholeCamera::Ptr(new PinholeCamera(width, height, K, D));}

private:

    PinholeCamera(int width, int height, double fx, double fy, double cx, double cy,
           double k1 = 0.0, double k2 = 0.0, double p1 = 0.0, double p2 = 0.0);

    PinholeCamera(int width, int height, const cv::Mat& K, const cv::Mat& D);

private:

    double k1_, k2_, p1_, p2_;
    cv::Mat cvK_, cvD_;


};

class AtanCamera : public AbstractCamera
{

public:

    typedef std::shared_ptr<AtanCamera> Ptr;

    virtual Vector3d lift(const Vector2d& px) const;

    virtual Vector3d lift(double x, double y) const;

    virtual Vector2d project(const Vector3d& xyz) const;

    virtual Vector2d project(double x, double y) const;

    virtual void undistortPoints(const std::vector<cv::Point2f> &pts_dist, std::vector<cv::Point2f> &pts_udist) const;

    inline static AtanCamera::Ptr create(int width, int height, double fx, double fy, double cx, double cy, double s = 0.0)
    {return AtanCamera::Ptr(new AtanCamera(width, height, fx, fy, cx, cy, s));}

    inline static AtanCamera::Ptr create(int width, int height, const cv::Mat& K, const double s = 0.0)
    {return AtanCamera::Ptr(new AtanCamera(width, height, K, s));}

private:

    AtanCamera(int width, int height, double fx, double fy, double cx, double cy, double s = 0.0);

    AtanCamera(int width, int height, const cv::Mat& K, const double s = 0.0);

private:

    double s_;
    double tans_;

};

    class Camera {

    public:
        static Point2d uv2camera(Point2f& point_uv_,Mat K_);//todo,还有一些函数没写
        static Point2f  removeDistort(Point2f &pre,double k1,double k2,double p1,double p2,Mat &k_);
        static Point2d world2camera(Point3d point_3d_);
        static Point2f camera2uv(Point2d point_camera_,Mat K_);
        static Point   world2uv(Point3d point_3d_);
        static Point3d camera2world(Point2d point_camera_);
        static Point3d uv2world(Point point_uv_);

    };
}

#endif