#ifndef _OPTIMIZER_HPP_
#define _OPTIMIZER_HPP_

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "PosePVR.h"

#include "map_point.hpp"
#include "keyframe.hpp"
#include "map.hpp"
#include "global.hpp"
#include "new_parameters.h"
#include "preintegration.h"
#include "utility.h"


namespace ssvo {

class Optimizer: public noncopyable
{
public:

    //! 仅初始化的时候使用
    static void twoViewBundleAdjustment(const KeyFrame::Ptr &kf1, const KeyFrame::Ptr &kf2, bool report=false, bool verbose=false);

    static void motionOnlyBundleAdjustment(const Frame::Ptr &last_frame,const Frame::Ptr &frame, bool use_seeds, bool vio, bool reject=false, bool report=false, bool verbose=false);

    static void localBundleAdjustment(const KeyFrame::Ptr &keyframe, std::list<MapPoint::Ptr> &bad_mpts, int size=10, int min_shared_fts=50, bool report=false, bool verbose=false);

//    static void localBundleAdjustmentWithInvDepth(const KeyFrame::Ptr &keyframe, std::list<MapPoint::Ptr> &bad_mpts, int size=10, bool report=false, bool verbose=false);

    static void refineMapPoint(const MapPoint::Ptr &mpt, int max_iter, bool report=false, bool verbose=false);

    static Vector2d reprojectionError(const ceres::Problem &problem, ceres::ResidualBlockId id);

    static void reportInfo(const ceres::Problem &problem, const ceres::Solver::Summary summary, bool report=false, bool verbose=false);
};

namespace ceres_slover {
// https://github.com/strasdat/Sophus/blob/v1.0.0/test/ceres/local_parameterization_se3.hpp
class SE3Parameterization : public ceres::LocalParameterization {
public:
    virtual ~SE3Parameterization() {}

    virtual bool Plus(double const *T_raw, double const *delta_raw,
                      double *T_plus_delta_raw) const {
        Eigen::Map<Sophus::SE3d const> const T(T_raw);
        Eigen::Map<Sophus::Vector6d const> const delta(delta_raw);
        Eigen::Map<Sophus::SE3d> T_plus_delta(T_plus_delta_raw);
        T_plus_delta = Sophus::SE3d::exp(delta) * T;
        return true;
    }

    // Set to Identity, for we have computed in ReprojectionErrorSE3::Evaluate
    virtual bool ComputeJacobian(double const *T_raw,
                                 double *jacobian_raw) const {
        Eigen::Map<Eigen::Matrix<double, 6, 7> > jacobian(jacobian_raw);
        jacobian.block<6,6>(0, 0).setIdentity();
        jacobian.rightCols<1>().setZero();
        return true;
    }

    virtual int GlobalSize() const { return Sophus::SE3d::num_parameters; }

    virtual int LocalSize() const { return Sophus::SE3d::DoF; }
};

struct ReprojectionError {
    ReprojectionError(double observed_x, double observed_y)
        : observed_x_(observed_x), observed_y_(observed_y) {}

    template<typename T>
    bool operator()(const T *const camera, const T *const point, T *residuals) const {
        Sophus::SE3<T> pose = Eigen::Map<const Sophus::SE3<T> >(camera);
        Eigen::Matrix<T, 3, 1> p = Eigen::Map<const Eigen::Matrix<T, 3, 1> >(point);

        Eigen::Matrix<T, 3, 1> p1 = pose.rotationMatrix() * p + pose.translation();

        T predicted_x = (T) p1[0] / p1[2];
        T predicted_y = (T) p1[1] / p1[2];
        residuals[0] = predicted_x - T(observed_x_);
        residuals[1] = predicted_y - T(observed_y_);
        return true;
    }

    static ceres::CostFunction *Create(const double observed_x, const double observed_y) {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, Sophus::SE3d::num_parameters, 3>(
            new ReprojectionError(observed_x, observed_y)));
    }

    double observed_x_;
    double observed_y_;
};

//class ReprojectionErrorSE3 : public ceres::SizedCostFunction<2, 7, 3>
class ReprojectionErrorSE3 : public ceres::SizedCostFunction<2, 9, 3>
{
public:

    ReprojectionErrorSE3(double observed_x, double observed_y, double weight)
        : observed_x_(observed_x), observed_y_(observed_y), weight_(weight) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        //! In Sophus, stored in the form of [q, t]
//        Eigen::Map<const Eigen::Quaterniond> q(parameters[0]);
//        Eigen::Map<const Eigen::Vector3d> t(parameters[0] + 4);
//        Eigen::Map<const Eigen::Vector3d> p(parameters[1]);


        //todo 从参数块读取数据
        Eigen::Map<const Eigen::Vector3d> Pwb(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> Vw(parameters[0] + 3);
        Eigen::Map<const Eigen::Vector3d> PHIwb(parameters[0] + 6);

        //Mappoint Pw
        Eigen::Map<const Eigen::Vector3d> Pw(parameters[1]);
        //todo 求Pc 即 p1
        Matrix3d Rcb = eigen_Rc2b.transpose();
        Vector3d Pbc = eigen_tc2b;

        Matrix3d Rwb = Sophus_new::SO3::exp(PHIwb).matrix();
        Vector3d p1 = Rcb * Rwb.transpose() * (Pw - Pwb) - Rcb * Pbc;

        //TODO 设置雅克比矩阵

//        Eigen::Vector3d p1 = q * p + t;

        const double predicted_x = p1[0] / p1[2];
        const double predicted_y = p1[1] / p1[2];


        //计算的是归一化平面中的误差？
        residuals[0] = predicted_x - observed_x_;
        residuals[1] = predicted_y - observed_y_;

//        cout<<"residuals[0]:"<<residuals[0]<<endl;
//        cout<<"residuals[1]:"<<residuals[1]<<endl;

        residuals[0] *= weight_;
        residuals[1] *= weight_;

        if(!jacobians) return true;
        double* jacobian0 = jacobians[0];
        double* jacobian1 = jacobians[1];

        //! The point observed is in the normalized plane
        Eigen::Matrix<double, 2, 3, Eigen::RowMajor> jacobian;

        const double z_inv = 1.0 / p1[2];
        const double z_inv2 = z_inv*z_inv;
        jacobian << z_inv, 0.0, -p1[0]*z_inv2,
                    0.0, z_inv, -p1[1]*z_inv2;

        jacobian.array() *= weight_;

        if(jacobian0 != nullptr)
        {
//            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > Jse3(jacobian0);
//            Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor> > Jse3(jacobian0);
//            Jse3.setZero();
//            //! In the order of Sophus::Tangent
//            Jse3.block<2,3>(0,0) = jacobian;
//            Jse3.block<2,3>(0,3) = jacobian*Sophus::SO3d::hat(-p1);

            Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor> > Jse3(jacobian0);
            Jse3.setZero();
            //! In the order of Sophus::Tangent
            Jse3.block<2,3>(0,0) =  jacobian * (-Rcb*Rwb.transpose());
            Jse3.block<2,3>(0,6) =  jacobian * (Sophus::SO3d::hat(Rcb*Rwb.transpose()*(Pw-Pwb)) * Rcb);
//            cout<<"Jse3-------------------------------"<<Jse3<<endl;
        }
        if(jacobian1 != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > Jpoint(jacobian1);
//            Jpoint = jacobian * q.toRotationMatrix();
            Jpoint =  jacobian * Rcb*Rwb.transpose();
//            cout<<"Jpoint-------------------------------"<<Jpoint<<endl;
        }
        return true;
    }

    static inline ceres::CostFunction *Create(const double observed_x, const double observed_y, const double weight = 1.0) {
        return (new ReprojectionErrorSE3(observed_x, observed_y, weight));
    }

private:

    double observed_x_;
    double observed_y_;
    double weight_;

}; // class ReprojectionErrorSE3

class ReprojectionErrorSE3InvDepth : public ceres::SizedCostFunction<2, 7, 7, 1>
{
public:

    ReprojectionErrorSE3InvDepth(double observed_x_ref, double observed_y_ref, double observed_x_cur, double observed_y_cur)
        : observed_x_ref_(observed_x_ref), observed_y_ref_(observed_y_ref),
          observed_x_cur_(observed_x_cur), observed_y_cur_(observed_y_cur) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {

        Eigen::Map<const Sophus::SE3d> T_ref(parameters[0]);
        Eigen::Map<const Sophus::SE3d> T_cur(parameters[1]);
        const double inv_z = parameters[2][0];

        const Eigen::Vector3d p_ref(observed_x_ref_/inv_z, observed_y_ref_/inv_z, 1.0/inv_z);
        const Sophus::SE3d T_cur_ref = T_cur * T_ref.inverse();
        const Eigen::Vector3d p_cur = T_cur_ref * p_ref;

        const double predicted_x =  p_cur[0] / p_cur[2];
        const double predicted_y =  p_cur[1] / p_cur[2];
        residuals[0] = predicted_x - observed_x_cur_;
        residuals[1] = predicted_y - observed_y_cur_;

        if(!jacobians) return true;
        double* jacobian0 = jacobians[0];
        double* jacobian1 = jacobians[1];
        double* jacobian2 = jacobians[2];

        //! The point observed is in the normalized plane
        Eigen::Matrix<double, 2, 3, Eigen::RowMajor> Jproj;

        const double z_inv = 1.0 / p_cur[2];
        const double z_inv2 = z_inv*z_inv;
        Jproj << z_inv, 0.0, -p_cur[0]*z_inv2,
            0.0, z_inv, -p_cur[1]*z_inv2;

        if(jacobian0 != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > JRse3(jacobian0);
            JRse3.setZero();
            Eigen::Matrix<double, 2, 3, Eigen::RowMajor> JRP = Jproj*T_cur_ref.rotationMatrix();
            JRse3.block<2,3>(0,0) = -JRP;
            JRse3.block<2,3>(0,3) = JRP*Sophus::SO3d::hat(p_ref);
        }
        if(jacobian1 != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > JCse3(jacobian1);
            JCse3.setZero();
            JCse3.block<2,3>(0,0) = Jproj;
            JCse3.block<2,3>(0,3) = Jproj*Sophus::SO3d::hat(-p_cur);
        }
        if(jacobian2 != nullptr)
        {
//            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > Jp(jacobian2);
//            Eigen::Matrix3d Jpp(T_cur_ref.rotationMatrix());
//            Jpp.col(2) = T_cur_ref.rotationMatrix() * (-p_ref);
//            Jp.noalias() = Jproj * Jpp * p_ref[2];
            Eigen::Map<Eigen::RowVector2d> Jp(jacobian2);
            Jp = Jproj * T_cur_ref.rotationMatrix() * p_ref * (-1.0/inv_z);
        }
        return true;
    }

    static inline ceres::CostFunction *Create(double observed_x_ref, double observed_y_ref,
                                              double observed_x_cur, double observed_y_cur) {
        return (new ReprojectionErrorSE3InvDepth(observed_x_ref, observed_y_ref, observed_x_cur, observed_y_cur));
    }

private:

    double observed_x_ref_;
    double observed_y_ref_;
    double observed_x_cur_;
    double observed_y_cur_;

};

class ReprojectionErrorSE3InvPoint : public ceres::SizedCostFunction<2, 7, 7, 3>
{
public:

    ReprojectionErrorSE3InvPoint(double observed_x, double observed_y)
        : observed_x_(observed_x), observed_y_(observed_y){}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Map<const Sophus::SE3d> T_ref(parameters[0]);
        Eigen::Map<const Sophus::SE3d> T_cur(parameters[1]);
        Eigen::Map<const Eigen::Vector3d> inv_p(parameters[2]);
        Sophus::SE3d T_cur_ref = T_cur * T_ref.inverse();

        const Eigen::Vector3d p_ref(inv_p[0] / inv_p[2], inv_p[1] / inv_p[2], 1.0 / inv_p[2]);
        const Eigen::Vector3d p_cur = T_cur_ref * p_ref;

        const double predicted_x =  p_cur[0] / p_cur[2];
        const double predicted_y =  p_cur[1] / p_cur[2];
        residuals[0] = predicted_x - observed_x_;
        residuals[1] = predicted_y - observed_y_;

        if(!jacobians) return true;
        double* jacobian0 = jacobians[0];
        double* jacobian1 = jacobians[1];
        double* jacobian2 = jacobians[2];

        //! The point observed is in the normalized plane
        Eigen::Matrix<double, 2, 3, Eigen::RowMajor> Jproj;

        const double z_inv = 1.0 / p_cur[2];
        const double z_inv2 = z_inv*z_inv;
        Jproj << z_inv, 0.0, -p_cur[0]*z_inv2,
            0.0, z_inv, -p_cur[1]*z_inv2;

        if(jacobian0 != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > JRse3(jacobian0);
            JRse3.setZero();
            Eigen::Matrix<double, 2, 3, Eigen::RowMajor> JRP = Jproj*T_cur_ref.rotationMatrix();
            JRse3.block<2,3>(0,0) = -JRP;
            JRse3.block<2,3>(0,3) = JRP*Sophus::SO3d::hat(p_ref);
        }
        if(jacobian1 != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > JCse3(jacobian1);
            JCse3.setZero();
            JCse3.block<2,3>(0,0) = Jproj;
            JCse3.block<2,3>(0,3) = Jproj*Sophus::SO3d::hat(-p_cur);
        }
        if(jacobian2 != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > Jp(jacobian2);
            Eigen::Matrix3d Jpp(T_cur_ref.rotationMatrix());
            Jpp.col(2) = T_cur_ref.rotationMatrix() * (-p_ref);
            Jp.noalias() = Jproj * Jpp * p_ref[2];
        }
        return true;
    }

    static inline ceres::CostFunction *Create(double observed_x, double observed_y) {
        return (new ReprojectionErrorSE3InvPoint(observed_x, observed_y));
    }

private:

    double observed_x_;
    double observed_y_;

};

    //! by jh
    //残差：6 参数块：3,3,3,1,3
    class GVSError: public ceres::SizedCostFunction<6,3,3,3,1,3>
    {
    public:
        GVSError(Matrix3d &Rbi_w_,Matrix3d &Rbj_w_,Vector3d &Pw_ci_,Vector3d &Pw_cj_,Vector3d &delta_a_,Vector3d &delta_v_,Matrix3d &jacobian_P_ba_,Matrix3d &jacobian_V_ba_,double &dt_)
        {
            Rbi_w=Rbi_w_;
            Rbj_w=Rbj_w_;
            Pw_ci=Pw_ci_;
            Pw_cj=Pw_cj_;
            delta_a=delta_a_;
            delta_v=delta_v_;
            jacobian_P_ba=jacobian_P_ba_;
            jacobian_V_ba=jacobian_V_ba_;
            dt=dt_;
        }
        virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians ) const
        {
            Vector3d Vi(parameters[0][0],parameters[0][1],parameters[0][2]);
            Vector3d Vj(parameters[1][0],parameters[1][1],parameters[1][2]);
            Vector3d Gravity(parameters[2][0],parameters[2][1],parameters[2][2]);
//            cout<<"g:"<<Gravity<<endl;
            double scale= parameters[3][0];
//            cout<<"scale:"<<scale<<endl;
            Vector3d ba(parameters[4][0],parameters[4][1],parameters[4][2]);
//            cout<<"ba:"<<ba.transpose()<<endl;
            Eigen::Map<Eigen::Matrix<double, 6, 1 >> residual(residuals);

            residual.block<3,1>(0,0)=Rbi_w*((scale*Pw_cj-scale*Pw_ci-Rbj_w.transpose()*eigen_tc2b+Rbi_w.transpose()*eigen_tc2b)-0.5*Gravity*dt*dt-Rbi_w.transpose()*Vi*dt)
                                     -delta_a-jacobian_P_ba*ba;
            residual.block<3,1>(3,0)=Rbi_w*(Rbj_w.transpose()*Vj-Gravity*dt-Rbi_w.transpose()*Vi)
                                     -delta_v-jacobian_V_ba*ba;
//            cout<<"residual:"<<residual.transpose()<<endl;

            //为什么需要判断？

            if(jacobians)
            {
                if(jacobians[0])
                {
                    Eigen::Map<Eigen::Matrix<double, 6, 3>> jacobian_vi(jacobians[0]);
                    jacobian_vi.block<3,3>(0,0)=-Matrix3d::Identity()*dt;
                    jacobian_vi.block<3,3>(3,0)=-Matrix3d::Identity();
                }
                if(jacobians[1])
                {
                    Eigen::Map<Eigen::Matrix<double, 6, 3>> jacobian_vj(jacobians[1]);
                    jacobian_vj.block<3,3>(0,0)=Matrix3d::Zero();
                    jacobian_vj.block<3,3>(3,0)=Rbi_w*Rbj_w.transpose();

                }
                if(jacobians[2])
                {
                    Eigen::Map<Eigen::Matrix<double, 6, 3>> jacobian_g(jacobians[2]);
                    jacobian_g.block<3,3>(0,0)=-0.5*Rbi_w*dt*dt;
                    jacobian_g.block<3,3>(3,0)=-Rbi_w*dt;
                }
                if(jacobians[3])
                {
                    Eigen::Map<Eigen::Matrix<double, 6, 1>> jacobian_s(jacobians[3]);
                    jacobian_s.block<3,1>(0,0)=Rbi_w*(Pw_cj-Pw_ci);
                    jacobian_s.block<3,1>(3,0)=Matrix<double, 3,1>::Zero();
                }
                if(jacobians[4])
                {
                    Eigen::Map<Eigen::Matrix<double, 6, 3>> jacobian_ba(jacobians[4]);
                    jacobian_ba.block<3,3>(0,0)=-jacobian_P_ba;
                    jacobian_ba.block<3,3>(3,0)=-jacobian_V_ba;
                }
            }

            return true;
        }
    private:
        Matrix3d Rbi_w,Rbj_w;
        Vector3d Pw_ci,Pw_cj;
        Vector3d delta_a,delta_v;
        Matrix3d jacobian_P_ba;
        Matrix3d jacobian_V_ba;
        double dt;

    };


    class IMUError: public ceres::SizedCostFunction<9,9,6,9>
    {
    public:
        IMUError(const Frame::Ptr &last_frame, const Frame::Ptr &cur_frame):last_frame_(last_frame),cur_frame_(cur_frame),preintegration_(cur_frame->preintegration)
        {}
        virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
        {

            //! 参数增量
            Eigen::Map<const Eigen::Vector3d> delta_pi(parameters[0]);
            Eigen::Map<const Eigen::Vector3d> delta_vi(parameters[0]+3);
            Eigen::Map<const Eigen::Vector3d> delta_phii(parameters[0]+6);

            Eigen::Map<const Eigen::Vector3d> delta_bgi(parameters[1]);
            Eigen::Map<const Eigen::Vector3d> delta_bai(parameters[1]+3);

            Eigen::Map<const Eigen::Vector3d> delta_pj(parameters[2]);
            Eigen::Map<const Eigen::Vector3d> delta_vj(parameters[2]+3);
            Eigen::Map<const Eigen::Vector3d> delta_phij(parameters[2]+6);


            //! 根据上一次的优化结果更新帧的PVR

            Matrix3d dR = Sophus::SO3d::exp(delta_phii).matrix();

            Sophus::SE3d pose_;
            pose_ = last_frame_->Twb();
            pose_.rotationMatrix() = pose_.rotationMatrix() * dR;
            pose_.translation() += delta_pi;
            last_frame_->setTwb(pose_);
            last_frame_->v += delta_vi;

            dR = Sophus::SO3d::exp(delta_phij).matrix();
            pose_ = cur_frame_->Twb();
            pose_.rotationMatrix() = pose_.rotationMatrix() * dR;
            pose_.translation() += delta_pj;
            cur_frame_->setTwb(pose_);
            cur_frame_->v += delta_vj;
            cur_frame_->optimal_Tcw_ = cur_frame_->Tcw();

            last_frame_->ba += delta_bai;
            last_frame_->bg += delta_bgi;
            last_frame_->preintegration->ba += delta_bai;
            last_frame_->preintegration->bg += delta_bgi;


            //! 得到两帧的 PVR bias 用于求误差函数
            SE3d T_lastFrame = last_frame_->Twb();
            SE3d T_curFrame = cur_frame_->Twb();

            Eigen::Vector3d Pi = T_lastFrame.translation();
            Eigen::Vector3d Vi = last_frame_->v;
            Eigen::Matrix3d Ri = T_lastFrame.rotationMatrix();
            Eigen::Vector3d Bai = last_frame_->ba;
            Eigen::Vector3d Bgi = last_frame_->bg;


            Eigen::Vector3d Pj = T_curFrame.translation();
            Eigen::Vector3d Vj = cur_frame_->v;
            Eigen::Matrix3d Rj = T_curFrame.rotationMatrix();
            Eigen::Vector3d Baj = cur_frame_->ba;
            Eigen::Vector3d Bgj = cur_frame_->bg;


//            double dt = preintegration_->sum_t;
//            Vector3d delta_pij = preintegration_->dp;
//            Vector3d delta_vij = preintegration_->dv;
//            Matrix3d delta_rij = preintegration_->dR;
//            Vector3d gravity = G ;

            /*
            Eigen::Map<const Eigen::Quaterniond> Qwb_i(parameters[0]);
            Eigen::Map<const Eigen::Vector3d> Pwb_i(parameters[0] + 4);
            Eigen::Map<const Eigen::Quaterniond> Qwb_j(parameters[2]);
            Eigen::Map<const Eigen::Vector3d> Pwb_j(parameters[2] + 4);



            Eigen::Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);
            Eigen::Vector3d Bai(parameters[1][3], parameters[1][4], parameters[1][5]);
            Eigen::Vector3d Bgi(parameters[1][6], parameters[1][7], parameters[1][8]);

            Eigen::Vector3d Vj(parameters[3][0], parameters[3][1], parameters[3][2]);
            Eigen::Vector3d Baj(parameters[3][3], parameters[3][4], parameters[3][5]);
            Eigen::Vector3d Bgj(parameters[3][6], parameters[3][7], parameters[3][8]);
             */

            Eigen::Map<Eigen::Matrix<double, 9, 1>> residual(residuals);


            residual = preintegration_->evaluate(Pi, Ri, Vi, Bai, Bgi, Pj, Rj, Vj, Baj, Bgj);



            double sum_t = preintegration_->sum_t;

            Eigen::Matrix3d dp_dba = preintegration_->jacobian_P_ba;
            Eigen::Matrix3d dp_dbg = preintegration_->jacobian_P_bg;

            Eigen::Matrix3d dr_dbg = preintegration_->jacobian_R_bg;

            Eigen::Matrix3d dv_dba = preintegration_->jacobian_V_ba;
            Eigen::Matrix3d dv_dbg = preintegration_->jacobian_V_bg;

            Eigen::Vector3d rPhiij = residual.segment<3>(6);
            Eigen::Matrix3d ExprPhiijTrans = Sophus::SO3d::exp(rPhiij).inverse().matrix();
            Eigen::Matrix3d JrBiasGCorr = Sophus_new::SO3::JacobianR(dr_dbg * (last_frame_->bg - preintegration_->bg_tmp));// todo 检查一下
            Eigen::Matrix3d JrInv_rPhi = Sophus_new::SO3::JacobianRInv( rPhiij );

//            cout<<"preintegration_->covariance:"<<endl<<preintegration_->covariance<<endl;

            Eigen::Matrix<double, 9, 9> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 9, 9>>(preintegration_->covariance.inverse()).matrixL().transpose();
            residual = sqrt_info * residual;

            if (jacobians)
            {
                if (jacobians[0])
                {
                    Eigen::Map<Eigen::Matrix<double, 9, 9, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);

                    jacobian_pose_i.setZero();
                    jacobian_pose_i.block<3, 3>(0, 0) = -Ri.inverse();// 0 0
                    jacobian_pose_i.block<3, 3>(0, 3) = -Ri.inverse() * sum_t;// 0 0
                    jacobian_pose_i.block<3, 3>(0, 6) = Sophus::SO3d::hat( Ri.inverse() * ( Pj - Pi - Vi * sum_t - 0.5 * G * sum_t * sum_t) );// 0 3
                    jacobian_pose_i.block<3, 3>(3, 3) = -Ri.inverse();
                    jacobian_pose_i.block<3, 3>(3, 6) = Sophus::SO3d::hat( Ri.inverse() * ( Vj - Vi - G * sum_t)  );// 0 3
                    jacobian_pose_i.block<3, 3>(6, 6) = - JrInv_rPhi * Rj.inverse() * Ri;

                    jacobian_pose_i = sqrt_info * jacobian_pose_i;

                    if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8)
                    {
                        ROS_WARN("numerical unstable in preintegration");
                        //std::cout << sqrt_info << std::endl;
                        ROS_BREAK();
                    }
                }
                if (jacobians[1])
                {
                    Eigen::Map<Eigen::Matrix<double, 9, 6, Eigen::RowMajor>> jacobian_speedbias_i(jacobians[1]);

                    jacobian_speedbias_i.setZero();
                    jacobian_speedbias_i.block<3, 3>(0, 0) = -dp_dbg;
                    jacobian_speedbias_i.block<3, 3>(0, 3) = -dp_dba;
                    jacobian_speedbias_i.block<3, 3>(3, 0) = -dv_dbg;
                    jacobian_speedbias_i.block<3, 3>(3, 3) = -dv_dba;
                    jacobian_speedbias_i.block<3, 3>(6, 0) = - JrInv_rPhi * ExprPhiijTrans * JrBiasGCorr * dr_dbg;

                    jacobian_speedbias_i = sqrt_info * jacobian_speedbias_i;

                    ROS_ASSERT(fabs(jacobian_speedbias_i.maxCoeff()) < 1e8);
                    ROS_ASSERT(fabs(jacobian_speedbias_i.minCoeff()) < 1e8);
                }
                if (jacobians[2])
                {
                    Eigen::Map<Eigen::Matrix<double, 9, 9, Eigen::RowMajor>> jacobian_pose_j(jacobians[2]);

                    jacobian_pose_j.setZero();
                    jacobian_pose_j.block<3, 3>(0, 0) = Ri.inverse();
                    jacobian_pose_j.block<3, 3>(3, 3) = Ri.inverse();
                    jacobian_pose_j.block<3, 3>(6, 6) = JrInv_rPhi;

                    jacobian_pose_j = sqrt_info * jacobian_pose_j;

                    ROS_ASSERT(fabs(jacobian_pose_j.maxCoeff()) < 1e8);
                    ROS_ASSERT(fabs(jacobian_pose_j.minCoeff()) < 1e8);
                }

            }
        }

        const Frame::Ptr last_frame_,cur_frame_;
        Preintegration::Ptr preintegration_;
    };


    class BiasError: public ceres::SizedCostFunction<6,3,3,3,3>
    {
    public:
        BiasError(Preintegration* preintegration):preintegration_(preintegration)
        {}
        virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
        {
            Eigen::Vector3d Bai(parameters[0][0], parameters[0][1], parameters[0][2]);
            Eigen::Vector3d Bgi(parameters[1][0], parameters[1][1], parameters[1][2]);
            Eigen::Vector3d Baj(parameters[2][0], parameters[2][1], parameters[2][2]);
            Eigen::Vector3d Bgj(parameters[3][0], parameters[3][1], parameters[3][2]);

            Eigen::Map<Eigen::Matrix<double, 6, 1>> residual(residuals);
            residual.block<3,1>(0,0)=Baj-Bai;
            residual.block<3,1>(3,0)=Bgj-Bgi;

            Eigen::Matrix<double, 6, 6> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 6, 6>>(preintegration_->noise_bais.inverse()).matrixL().transpose();
            residual = sqrt_info * residual;

            if (jacobians)
            {

                if (jacobians[0])
                {
                    Eigen::Map<Eigen::Matrix<double, 3, 3>> jacobian_bai(jacobians[0]);
                    jacobian_bai = -Matrix<double,3,3>::Identity();

                }
                if (jacobians[1])
                {
                    Eigen::Map<Eigen::Matrix<double, 3, 3>> jacobian_bgi(jacobians[1]);
                    jacobian_bgi = -Matrix<double,3,3>::Identity();
                }
                if (jacobians[2])
                {
                    Eigen::Map<Eigen::Matrix<double, 3, 3>> jacobian_baj(jacobians[2]);
                    jacobian_baj = Matrix<double,3,3>::Identity();
                }
                if (jacobians[3])
                {
                    Eigen::Map<Eigen::Matrix<double, 3, 3>> jacobian_bgj(jacobians[3]);
                    jacobian_bgj = Matrix<double,3,3>::Identity();
                }
            }
        }

        Preintegration* preintegration_;
    };


}//! namespace ceres

}//! namespace ssvo

#endif