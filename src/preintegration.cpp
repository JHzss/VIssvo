//
// Created by jh on 18-6-4.
//

#include "preintegration.h"
#include "new_parameters.h"


using namespace cv;

namespace ssvo {

//    double acc_n;
//    double acc_w;
//    double gyr_n;
//    double gyr_w;

    Eigen::Matrix3d skew(Vector3d vector_)
    {
        Eigen::Matrix3d skew_;
        skew_<<0,-1*vector_(2),vector_(1),
            vector_(2),0,-1*vector_(0),
            -1*vector_(1),vector_(0),0;
        return skew_;
    }

    Preintegration::Preintegration(Vector3d &ba_, Vector3d &bg_) : sum_t(0),ba(ba_),bg(bg_),
            covariance(Eigen::Matrix<double,9,9>::Zero()),dp(Eigen::Vector3d::Zero()),dv(Eigen::Vector3d::Zero()),dR(Eigen::Matrix<double,3,3>::Identity())
    {
        noise_ba=(acc_n*acc_n)* Matrix3d::Identity()/0.005;
        noise_bg=(gyr_n*gyr_n)* Matrix3d::Identity()/0.005;
//        noise_ba=Matrix3d::Identity()*2.0e-3*2.0e-3/0.005*100;
//        noise_bg=Matrix3d::Identity()*1.7e-4*1.7e-4/0.005;
        noise_bais.topLeftCorner(3,3)=noise_ba;
        noise_bais.bottomRightCorner(3,3)=noise_bg;
        jacobian_P_ba =Matrix3d::Zero();     // position / gyro
        jacobian_P_bg =Matrix3d::Zero();     // position / acc
        jacobian_V_ba =Matrix3d::Zero();     // velocity / gyro
        jacobian_V_bg =Matrix3d::Zero();     // velocity / acc
        jacobian_R_bg =Matrix3d::Zero();   // rotation / gyro
//        ba=Eigen::Vector3d(-0.02,0.18,0.07);
    }
    //预积分的主要函数,使用欧拉积分的方法
    void Preintegration::run()
    {
        Vector3d acc_tmp,gyr_tmp;
        ba_tmp=ba;
        bg_tmp=bg;
        for(int i=0;i<dt_buf.size();i++)
        {
            if(i==0)
            {
                acc_tmp=acc_buf[0]-ba;
                gyr_tmp=gyr_buf[0]-bg;
                continue;
            }
            //! 由陀螺仪角速度直接算出来的两帧之间的旋转
            Mat deltaR_wt_mat;
            Matrix3d deltaR_wt_eigen;
            Vector3d w;
            Mat w_mat;
            Matrix3d Jr,Jr_inv;
            double dt;
            dt=dt_buf[i];
//            cout<<"i:"<<i<<endl;
//            cout<<"dt:"<<dt<<endl;
//            ROS_ASSERT(dt>0);
            if(dt==0)continue;
            w=gyr_tmp*dt;

//            eigen2cv(w,w_mat);
//            Rodrigues(w_mat,deltaR_wt_mat);
//            cv2eigen(deltaR_wt_mat,deltaR_wt_eigen);

            Sophus::SO3d deltaR_;

            double theta_;
            theta_ = w.norm();
            double half_theta = 0.5*(theta_);

            double imag_factor;
            double real_factor = cos(half_theta);
            if((theta_)<1e-10)
            {
                double theta_sq = (theta_)*(theta_);
                double theta_po4 = theta_sq*theta_sq;
                imag_factor = 0.5-0.0208333*theta_sq+0.000260417*theta_po4;
            }
            else
            {
                double sin_half_theta = sin(half_theta);
                imag_factor = sin_half_theta/(theta_);
            }

            deltaR_= Sophus::SO3d(Quaterniond(real_factor,
                                   imag_factor*w.x(),
                                   imag_factor*w.y(),
                                   imag_factor*w.z()));
            deltaR_wt_eigen=deltaR_.matrix();



            //! 王京简化了一些,计算right jacobian of SO(3)

                /*
                 * 王京中的特殊处理
                 * if(theta<0.00001)
                    {
                         return Jr;// = Matrix3d::Identity();
                    }
                 */
                double theta=w.norm();
                Matrix3d w_skew=skew(w);
                if(theta<0.00001)
                {
                    Jr=Matrix3d::Identity();
                    Jr_inv=Matrix3d::Identity();
                } else
                {
                    Jr=Matrix3d::Identity()
                       -(1-cos(theta))/(theta*theta)*w_skew
                       +(theta-sin(theta))/(theta*theta*theta)*w_skew*w_skew;
                    Jr_inv=Matrix3d::Identity()
                           +0.5*w_skew
                           +(1/(theta*theta)-(1+cos(theta))/(0.5*theta*sin(theta)))*w_skew*w_skew;//SVO的作者公式计算有点问题，查资料看王京写的改正了。
                }

            ///角度太小了？罗德里格斯公式不可以直接用？
//                tmp_dr=cos(theta)*Matrix3d::Identity()+(1-cos(theta))*w*w.transpose()+sin(theta)*w_skew;
            //! calculate Jacobian and coverance
            //todo 这个的理解还是不够，
                //! SVO作者 On-Manifold Preintegration for Real-Time Visual--Inertial Odometry APPENDIX-A
                Matrix<double,9,9> A=Matrix<double,9,9>::Identity();
                A.block(0,3,3,3)=dt*Matrix3d::Identity();
                A.block(0,6,3,3)=-0.5*dR*skew(acc_tmp)*dt*dt;
                A.block(3,6,3,3)=-dR*skew(acc_tmp)*dt;
                A.block(6,6,3,3)=deltaR_wt_eigen.transpose();

                Matrix<double,9,3> Ba=Matrix<double,9,3>::Zero();
                Ba.block(0,0,3,3)=0.5*dR*dt*dt;
                Ba.block(3,0,3,3)=dR*dt;

                Matrix<double,9,3> Bg=Matrix<double,9,3>::Zero();
                Bg.block(6,0,3,3)=Jr*dt;

                covariance = A*covariance*A.transpose()+Ba*noise_ba*Ba.transpose()+Bg*noise_bg*Bg.transpose();

            /// calculate the jacobian to bais to update the dp v R

                jacobian_P_ba += jacobian_V_ba*dt-0.5*dR*dt*dt;
                jacobian_P_bg += jacobian_V_bg*dt-0.5*dR*skew(acc_tmp)*jacobian_R_bg*dt*dt;
                jacobian_V_ba += -dR*dt;
                jacobian_V_bg += -dR*skew(acc_tmp)*jacobian_R_bg*dt;
                //todo 这个公式跟论文中不一样，是为什么？
                jacobian_R_bg = deltaR_wt_eigen.transpose()*jacobian_R_bg-Jr*dt;
//                cout<<"calcul jacobian"<<endl<<jacobian_R_bg<<endl;

            //! update the delta p v r
                dp+=dv*dt+0.5*dR*acc_tmp*dt*dt;
                dv+=dR*acc_tmp*dt;
                //todo VIORB里面有判断这个解的过程，不知道这里需不需要
                dR=dR*deltaR_wt_eigen;


                Quaterniond q_r(dR);
                if(q_r.w()<0)
                {
                    q_r.coeffs()*=-1;
                }
                dR=(q_r.normalized()).toRotationMatrix();

            //! update the measurement
            acc_tmp=acc_buf[i]-ba;
            gyr_tmp=gyr_buf[i]-bg;
        }
        cout<<"covariance:"<<endl<<covariance<<endl;
    }

    //todo 重新传播
    void Preintegration::rerun()
    {
        jacobian_P_ba =Matrix3d::Zero();     // position / gyro
        jacobian_P_bg =Matrix3d::Zero();     // position / acc
        jacobian_V_ba =Matrix3d::Zero();     // velocity / gyro
        jacobian_V_bg =Matrix3d::Zero();     // velocity / acc
        jacobian_R_bg =Matrix3d::Zero();   // rotation / gyro
        covariance=Eigen::Matrix<double,9,9>::Zero();
        dp=Eigen::Vector3d::Zero();
        dv=Eigen::Vector3d::Zero();
        dR=Eigen::Matrix<double,3,3>::Identity();
        run();
    }

    //TODO 微调

    void Preintegration::clearState()
    {
        noise_ba=(acc_n*acc_n)* Matrix3d::Identity();
        noise_bg=(gyr_n*gyr_n)* Matrix3d::Identity();
        jacobian_P_ba =Matrix3d::Zero();     // position / gyro
        jacobian_P_bg =Matrix3d::Zero();     // position / acc
        jacobian_V_ba =Matrix3d::Zero();     // velocity / gyro
        jacobian_V_bg =Matrix3d::Zero();     // velocity / acc
        jacobian_R_bg =Matrix3d::Zero();   // rotation / gyro
        ba.setZero();
        bg.setZero();
        covariance=Eigen::Matrix<double,9,9>::Zero();
        dp=Eigen::Vector3d::Zero();
        dv=Eigen::Vector3d::Zero();
        dR=Eigen::Matrix<double,3,3>::Identity();
    }
}