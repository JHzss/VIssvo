//
// Created by jh on 18-9-4.
//

#include "PosePVR.h"
//#include ""
/*
bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    Eigen::Map<const Eigen::Vector3d> _p(x);//用指针构造Eigen里的类
    Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

    Eigen::Map<const Eigen::Vector3d> dp(delta);

    Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

    Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
    Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

    p = _p + dp;
    q = (_q * dq).normalized();

    return true;
}
bool PoseLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
    j.topRows<6>().setIdentity();
    j.bottomRows<1>().setZero();

    return true;
}
 */

namespace ssvo
{
    PosePVR::PosePVR() //!顶层const 指针不能变，指向的内容可以变
    {}

    //在一个类里定义了一个const成员函数后，则此函数不能修改类中的成员变量，
    bool PosePVR::Plus(const double *x, const double *delta, double *x_plus_delta) const
    {

        Eigen::Map<const Eigen::Vector3d > p_dt(delta);
        Eigen::Map<const Eigen::Vector3d > v_dt(delta+3);
        Eigen::Map<const Eigen::Vector3d > phi_dt(delta+6);


        Eigen::Map<Eigen::Vector3d > p_re(x_plus_delta);
        Eigen::Map<Eigen::Vector3d > v_re(x_plus_delta+3);
        Eigen::Map<Eigen::Vector3d > phi_re(x_plus_delta+6);

        p_re = p_dt;
        v_re = v_dt;
        phi_re = phi_dt;

        /*
        frame_->v += v_dt;

        Sophus_new::SO3 dR = Sophus_new::SO3::exp(phi_dt);
        Sophus::SE3d pose_;
        pose_ = frame_->optimal_Tcw_;
        pose_.rotationMatrix() *= dR;
        pose_.translation() += p_dt;
        frame_->setTcw(pose_);
         */

        return true;
    }

    bool PosePVR::ComputeJacobian(const double *x, double *jacobian) const
    {

        Eigen::Map<Eigen::Matrix<double, 9, 9, Eigen::RowMajor>> j(jacobian);

        Eigen::Matrix<double ,9,9> I9x9=Eigen::Matrix<double ,9,9>::Identity();

        j=I9x9;

    }


}