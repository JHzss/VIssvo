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
//            cout<<"Pi: "<<Pi.transpose()<<endl;
//            cout<<"Vi: "<<Vi.transpose()<<endl;
//            cout<<"Bai: "<<Bai.transpose()<<endl;
//            cout<<"Bgi: "<<Bgi.transpose()<<endl;
//            cout<<"Pj: "<<Pj.transpose()<<endl;
//            cout<<"Vj: "<<Vj.transpose()<<endl;
//            cout<<"Baj: "<<Baj.transpose()<<endl;
//            cout<<"Bgj: "<<Bgj.transpose()<<endl;
            
            Eigen::Matrix<double, 9, 1> residuals;

            Eigen::Vector3d dba = Bai - ba_tmp;//ba_j 的ba_tmp 就是bi的初始值
            Eigen::Vector3d dbg = Bgi - bg_tmp;

//            cout<<"dba: "<<endl<<dba<<endl<<"dbg:   "<<endl<<dbg<<endl;

            //! exp(Jr*delta bg)
            Matrix3d deltaR_wt_eigen;
            Vector3d w;
            w = jacobian_R_bg * dbg;

            deltaR_wt_eigen = Sophus_new::SO3::exp(w).matrix();
//            cout<<"deltaR_wt_eigen: "<<endl<<deltaR_wt_eigen<<endl;

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

        void copyPreintegration(Preintegration::Ptr preintegration)
        {
            this->covariance = preintegration->covariance;
            this->ba_tmp = preintegration->ba_tmp;
            this->bg_tmp = preintegration->bg_tmp;
            this->ba = preintegration->ba;
            this->bg = preintegration->bg;

            this->noise_ba = preintegration->noise_ba;
            this->noise_bg = preintegration->noise_bg;
            this->noise_bias = preintegration->noise_bias;
            this->sum_t = preintegration->sum_t;

            this->dp = preintegration->dp;
            this->dv = preintegration->dv;
            this->dR = preintegration->dR;

            this->jacobian_P_ba = preintegration->jacobian_P_ba;
            this->jacobian_P_bg = preintegration->jacobian_P_bg;
            this->jacobian_V_ba = preintegration->jacobian_V_ba;
            this->jacobian_V_bg = preintegration->jacobian_V_bg;
            this->jacobian_R_bg = preintegration->jacobian_R_bg;

            this->img_stamp = preintegration->img_stamp;

            this->dt_buf.assign(preintegration->dt_buf.begin(),preintegration->dt_buf.end());
            this->acc_buf.assign(preintegration->acc_buf.begin(),preintegration->acc_buf.end());
            this->gyr_buf.assign(preintegration->gyr_buf.begin(),preintegration->gyr_buf.end());
        }

        void addAndUpdate(Preintegration::Ptr preintegration);

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
