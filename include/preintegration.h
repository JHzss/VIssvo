//
// Created by jh on 18-6-4.
//

#ifndef SSVO_PREINTEGRATION_H
#define SSVO_PREINTEGRATION_H

#include "global.hpp"
#include "new_parameters.h"
#include <opencv2/core/eigen.hpp>
#include "imudata.h"

namespace ssvo {

    class Preintegration {
    public:
        typedef shared_ptr<Preintegration> Ptr;

        Preintegration(Vector3d &ba, Vector3d &bg);//构造函数的名字！！！一定要与类名一致
        inline static Preintegration::Ptr creat(Vector3d &ba, Vector3d &bg)
        { return Preintegration::Ptr(new Preintegration(ba,bg)); };

        //! calculate the delta p q v
        void run();

        void rerun();

        void clearState();

        Eigen::Matrix<double, 9, 1> evaluate(const Eigen::Vector3d &Pi, const Eigen::Matrix3d &Ri, const Eigen::Vector3d &Vi, const Eigen::Vector3d &Bai, const Eigen::Vector3d &Bgi,
                                              const Eigen::Vector3d &Pj, const Eigen::Matrix3d &Rj, const Eigen::Vector3d &Vj, const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj)
        {
//            Matrix3d Rwb_i=Qwb_i.toRotationMatrix();
//            Matrix3d Rwb_j=Qwb_j.toRotationMatrix();
            Eigen::Matrix<double, 9, 1> residuals;

            Eigen::Vector3d dba = Bai - ba_tmp;//ba_j 的ba_tmp 就是bi的初始值
            Eigen::Vector3d dbg = Bgi - bg_tmp;

            cout<<"dba: "<<endl<<dba<<endl<<"dbg:   "<<endl<<dbg<<endl;

            //! exp(Jr*delta bg)
            Matrix3d deltaR_wt_eigen;
            Vector3d w;
            w = jacobian_R_bg * dbg;

            deltaR_wt_eigen = Sophus_new::SO3::exp(w).matrix();
            cout<<"deltaR_wt_eigen: "<<endl<<deltaR_wt_eigen<<endl;

            Eigen::Matrix3d corrected_delta_R = dR * deltaR_wt_eigen;
            Eigen::Vector3d corrected_delta_v = dv + jacobian_V_ba * dba + jacobian_V_bg * dbg;
            Eigen::Vector3d corrected_delta_p = dp + jacobian_P_ba * dba + jacobian_P_bg * dbg;

            residuals.block<3, 1>(0, 0) = Ri.inverse() * (-0.5 * G * sum_t * sum_t + Pj - Pi - Vi * sum_t) - corrected_delta_p;
            residuals.block<3, 1>(3, 0) = Ri.inverse() * ( Vj - Vi - G * sum_t) - corrected_delta_v;
            //todo
            Sophus_new::SO3 tmp(corrected_delta_R.inverse() * Ri.inverse() * Rj);
            residuals.block<3, 1>(6, 0) = tmp.log();
            return residuals;
        }

    public:

        Eigen::Matrix<double, 9, 9> covariance;
        Vector3d ba_tmp,bg_tmp;// 用于imu残差优化，存储上一次预积分时的加速度计参数
        Eigen::Vector3d ba, bg;

        /**
         * For EuRoc dataset, according to V1_01_easy/imu0/sensor.yaml
         * The params:
         * sigma_g: 1.6968e-4       rad / s / sqrt(Hz)
         * sigma_gw: 1.9393e-5      rad / s^2 / sqrt(Hz)
         * sigma_a: 2.0e-3          m / s^2 / sqrt(Hz)
         * sigma_aw: 3.0e-3         m / s^3 / sqrt(Hz)
         */
//        double hz_inv_t = 0.005;
//        double sigma_g = 1.6968e-4;
//        double sigma_a = 2.0e-3;
//        double sigma_gw = 1.9393e-5;
//        double sigma_aw = 3.0e-3 ;


        Matrix3d noise_ba,noise_bg;
        Matrix<double,6,6> noise_bias;

        double dt, sum_t;
        vector<double> dt_buf;
        vector<Vector3d> acc_buf;
        vector<Vector3d> gyr_buf;

        Vector3d dp;
        Vector3d dv;
        Matrix3d dR;

        //! jacobian for incorporating bias update
        Eigen::Matrix3d jacobian_P_ba;     // position / gyro
        Eigen::Matrix3d jacobian_P_bg;     // position / acc
        Eigen::Matrix3d jacobian_V_ba;     // velocity / gyro
        Eigen::Matrix3d jacobian_V_bg;     // velocity / acc
        Eigen::Matrix3d jacobian_R_bg;   // rotation / gyro

        Eigen::Quaterniond right_Q;

        double img_stamp;//这一组imu得到的预积分对应的图像的stamp
    };

}
#endif //SSVO_PREINTEGRATION_H
