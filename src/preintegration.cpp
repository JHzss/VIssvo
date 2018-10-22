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
        noise_ba = Matrix3d::Identity()*2.0e-3*2.0e-3/0.005*100; /*(acc_n*acc_n)* Matrix3d::Identity()/0.005*/
        noise_bg = Matrix3d::Identity()*1.7e-4*1.7e-4/0.005; /*(gyr_n*gyr_n)* Matrix3d::Identity()/0.005*/
//        noise_ba=Matrix3d::Identity()*2.0e-3*2.0e-3/0.005*100;
//        noise_bg=Matrix3d::Identity()*1.7e-4*1.7e-4/0.005;
        noise_bias.topLeftCorner(3,3)=noise_ba;
        noise_bias.bottomRightCorner(3,3)=noise_bg;
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
        ba_tmp = ba;
        bg_tmp = bg;
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

            deltaR_wt_eigen = Sophus_new::SO3::exp(w).matrix();
            Jr = Sophus_new::SO3::JacobianR(w);
            Jr_inv = Sophus_new::SO3::JacobianRInv(w);

            ///角度太小了？罗德里格斯公式不可以直接用？
//                tmp_dr=cos(theta)*Matrix3d::Identity()+(1-cos(theta))*w*w.transpose()+sin(theta)*w_skew;
            //! calculate Jacobian and coverance
            //todo 这个的理解还是不够，
                //! SVO作者 On-Manifold Preintegration for Real-Time Visual--Inertial Odometry APPENDIX-A
                Matrix<double,9,9> A = Matrix<double,9,9>::Identity();
                A.block(0,3,3,3) = dt*Matrix3d::Identity();
                A.block(0,6,3,3) = -0.5*dR*skew(acc_tmp)*dt*dt;
                A.block(3,6,3,3) = -dR*skew(acc_tmp)*dt;
                A.block(6,6,3,3) = deltaR_wt_eigen.transpose();

                Matrix<double,9,3> Ba = Matrix<double,9,3>::Zero();
                Ba.block(0,0,3,3) = 0.5*dR*dt*dt;
                Ba.block(3,0,3,3) = dR*dt;

                Matrix<double,9,3> Bg = Matrix<double,9,3>::Zero();
                Bg.block(6,0,3,3) = Jr*dt;

                covariance = A*covariance*A.transpose()+Ba*IMUData::getAccMeasCov()*Ba.transpose()+Bg*IMUData::getGyrMeasCov()*Bg.transpose();

//            std::cout<<"ssvo A:------------------------------------"<<std::endl<<A<<std::endl;
//            std::cout<<"ssvo Bg:------------------------------------"<<std::endl<<Bg<<std::endl;
//            std::cout<<"ssvo Ba:------------------------------------"<<std::endl<<Ba<<std::endl;

            /// calculate the jacobian to bias to update the dp v R

                jacobian_P_ba += jacobian_V_ba*dt - 0.5*dR*dt*dt;
                jacobian_P_bg += jacobian_V_bg*dt - 0.5*dR*skew(acc_tmp)*jacobian_R_bg*dt*dt;
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
//        cout<<"covariance:"<<endl<<covariance<<endl;
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

    void Preintegration::addAndUpdate(Preintegration::Ptr preintegration)
    {
        //! 用关键帧的bias进行计算
        //TODO 融合buf，积分
        dt_buf.insert(dt_buf.end(),preintegration->dt_buf.begin(),preintegration->dt_buf.end());
        acc_buf.insert(acc_buf.end(),preintegration->acc_buf.begin(),preintegration->acc_buf.end());
        gyr_buf.insert(gyr_buf.end(),preintegration->gyr_buf.begin(),preintegration->gyr_buf.end());
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