//
// Created by jh on 18-6-14.
//
#include "imu_vision_align.h"

namespace ssvo
{

///估计陀螺仪bias
Vector3d EstimateGyrBias(deque<Frame::Ptr> &initilization_frame_buffer_)
{
    std::deque<Frame::Ptr>::iterator iter;
    int i=0;
    //todo imu-vision 外参
    Matrix3d eigen_Rb2c =eigen_Rc2b.transpose();
    ///这个用法检查过
    Matrix3d A;
    A.setZero();
    Vector3d b;
    b.setZero();
    for(iter=initilization_frame_buffer_.begin();iter!=initilization_frame_buffer_.end()-1;i=i+3)
    {
        //todo 当前帧的赋值，变量提取 i
        Vector3d temp_b;
        Matrix3d temp_A;
        Matrix3d Rwb_i = (*iter)->Twc().rotationMatrix()*eigen_Rb2c;
        iter++;
        //todo 下一帧的赋值，变量提取 j
        Matrix3d jacobian_R_bg=(*iter)->preintegration->jacobian_R_bg;
        Matrix3d delta_R_ij= (*iter)->preintegration->dR;
        Matrix3d Rwb_j = (*iter)->Twc().rotationMatrix()*eigen_Rb2c;
        Matrix3d Rbw_j = Rwb_j.transpose();
        Matrix3d R_tmp=delta_R_ij.transpose()*Rwb_i.transpose()*Rwb_j;
        Sophus::SO3d so3_tmp(R_tmp);
        temp_A =jacobian_R_bg;
        temp_b =so3_tmp.log();
        A+= temp_A.transpose()*temp_A;
        b+= temp_A.transpose()*temp_b;
    }
    A=A*1000;
    b=b*1000;

    Vector3d delta_bg = A.ldlt().solve(b);
    cout<<"bg~~~~:"<<delta_bg.transpose()<<endl;
    for(auto frame:initilization_frame_buffer_)
    {
        ///这里注意一下bg值都得更新，因为当初没弄好，就注意一下吧
//        frame->bg+=delta_bg;
        frame->preintegration->bg+=delta_bg;
        frame->preintegration->rerun();
    }
    return delta_bg;
}

///估计重力、速度、尺度
bool EstimateGVS(deque<Frame::Ptr> &initilization_frame_buffer_, VectorXd &x)
{
    //!线性方程组求解
    int frame_count=initilization_frame_buffer_.size();
    MatrixXd A{frame_count*3+3+1, frame_count*3+3+1};
    A.setZero();
    VectorXd b{frame_count*3+3+1};
    b.setZero();

    std::deque<Frame::Ptr>::iterator frame_i;
    std::deque<Frame::Ptr>::iterator frame_j;

    int i=0;
    for(frame_i=initilization_frame_buffer_.begin();next(frame_i)!=initilization_frame_buffer_.end();frame_i++,i++)
    {
        MatrixXd temp_A(6,10);
        temp_A.setZero();
        Eigen::VectorXd temp_b(6);
        temp_b.setZero();

        frame_j=next(frame_i);
        double frame_dt=(*frame_j)->preintegration->sum_t;
//        cout<<"sum_t:"<<frame_dt<<endl;

        Matrix3d eigen_Rb2c=eigen_Rc2b.transpose();
        Matrix3d Rwb_i=(*frame_i)->Twc().rotationMatrix()*eigen_Rb2c;
        Matrix3d Rwb_j=(*frame_j)->Twc().rotationMatrix()*eigen_Rb2c;
        Vector3d Pwc_i=(*frame_i)->Twc().translation();
        Vector3d Pwc_j=(*frame_j)->Twc().translation();

        temp_A.block(0,0,3,3)=-Matrix3d::Identity()*frame_dt;
        temp_A.block(0,6,3,3)=-0.5*Rwb_i.transpose()*frame_dt*frame_dt;
        temp_A.block(0,9,3,1)=Rwb_i.transpose()*(Pwc_j-Pwc_i)/100.0;
        temp_A.block(3,0,3,3)=-Matrix3d::Identity();
        temp_A.block(3,3,3,3)=Rwb_i.transpose()*Rwb_j;
        temp_A.block(3,6,3,3)=-Rwb_i.transpose()*frame_dt;

        temp_b.block(0,0,3,1)=(*frame_j)->preintegration->dp-eigen_tc2b+Rwb_i.transpose()*Rwb_j*eigen_tc2b;
        temp_b.block(3,0,3,1)=(*frame_j)->preintegration->dv;

        MatrixXd r_A = temp_A.transpose() * temp_A;
        VectorXd r_b = temp_A.transpose() * temp_b;

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();

        A.block<6, 4>(i * 3, frame_count*3) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(frame_count*3, i * 3) += r_A.bottomLeftCorner<4, 6>();

    }
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);


    Vector3d g=x.segment<3>(frame_count*3);
    cout<<"g:"<<g.norm()<<"---:"<<g.transpose()<<endl;
    double s=(x.tail<1>())(0)/100.0;
    cout<<"scale:"<<s<<endl;
    /*
    for(int i=0;i<initilization_frame_buffer_.size();i++)
    {
        initilization_frame_buffer_[i]->v=x.segment<3>(i*3);
        initilization_frame_buffer_[i]->s=(x.tail<1>())(0);
    }
    //!优化库求解
    ceres::Problem problem;

//    int frame_count=initilization_frame_buffer_.size();
    double velocity_frame[frame_count][3];
    double gravity[3];
    double scale[1];
    double acc_bias[3]={0};
    Vector3d gravity_v;
    double s;
    gravity_v=x.segment<3>(frame_count*3);
    gravity_v=9.8107*gravity_v.normalized();

//    ROS_ASSERT_MSG(gravity_v.norm()<10,"gravity is too large");
    gravity[0] = gravity_v(0);
    gravity[1] = gravity_v(1);
    gravity[2] = gravity_v(2);
    s = x(frame_count*3+3);
    scale[0] = s;

    for(int i;i<frame_count;i++)
    {
        velocity_frame[i][0]=initilization_frame_buffer_[i]->v.x();
        velocity_frame[i][1]=initilization_frame_buffer_[i]->v.y();
        velocity_frame[i][2]=initilization_frame_buffer_[i]->v.z();
        problem.AddParameterBlock(velocity_frame[i],3);
        problem.SetParameterBlockConstant(velocity_frame[i]);
    }
    problem.AddParameterBlock(gravity,3);
    problem.AddParameterBlock(scale,1);
    problem.AddParameterBlock(acc_bias,3);
//        problem.SetParameterBlockConstant(scale);

        for(int i;i<frame_count-1;i++)
        {
            int j=i+1;
            Matrix3d eigen_Rb2c=eigen_Rc2b.transpose();
            Matrix3d Rbi_w= eigen_Rc2b*initilization_frame_buffer_[i]->Tcw().rotationMatrix();
            Matrix3d Rbj_w= eigen_Rc2b*initilization_frame_buffer_[j]->Tcw().rotationMatrix();
            Vector3d Pw_ci= initilization_frame_buffer_[i]->Twc().translation();
            Vector3d Pw_cj= initilization_frame_buffer_[j]->Twc().translation();
            Vector3d delta_a= initilization_frame_buffer_[j]->preintegration->dp;
            Vector3d delta_v= initilization_frame_buffer_[j]->preintegration->dv;
            Matrix3d jacobian_P_ba= initilization_frame_buffer_[j]->preintegration->jacobian_P_ba;
            Matrix3d jacobian_V_ba= initilization_frame_buffer_[j]->preintegration->jacobian_V_ba;
            double dt= initilization_frame_buffer_[j]->preintegration->sum_t;

            ssvo::ceres_slover::GVSError* GVSerror= new ssvo::ceres_slover::GVSError(Rbi_w,Rbj_w,Pw_ci,Pw_cj,delta_a,delta_v,jacobian_P_ba,jacobian_V_ba,dt);
            problem.AddResidualBlock(GVSerror,NULL,velocity_frame[i],velocity_frame[j],gravity,scale,acc_bias);

        }
        ceres::Solver::Options options;
        options.linear_solver_type= ceres::DENSE_SCHUR;
        options.trust_region_strategy_type=ceres::DOGLEG;
//        options.max_num_iterations=8;
        options.max_solver_time_in_seconds=0.2;
        options.minimizer_progress_to_stdout=true;
        ceres::Solver::Summary summary;
        ceres::Solve(options,&problem,&summary);

        cout << summary.BriefReport() << endl;
        ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
        cout<<"gravity:"<<gravity[0]<<" "<<gravity[1]<<" "<<gravity[2]<<endl;
        cout<<"scale:"<<scale[0]<<endl;
        cout<<"ba:"<<acc_bias[0]<<" "<<acc_bias[1]<<" "<<acc_bias[2]<<" "<<endl;

//        ros::shutdown();

//    cout<<x<<endl;
*/
    if(fabs(g.norm() - 9.8) > 0.1 || s < 0)
    {
        cout<<"vio process bad"<<endl;
        return false;
    }
    else
    {
        return true;
    }
}

///重力求精
bool RefineGravity(deque<Frame::Ptr> &initilization_frame_buffer_, VectorXd &x)
{
//    Vector3d G;
//    G.z()=9.8107;
    Vector3d g = x.segment<3>(x.rows() - 4);
    Vector3d g0 = g.normalized() * G.norm();//g0就是原始的解。
    cout<<"g0:"<<g0<<endl;
    Vector3d lx, ly;
    //VectorXd x;

    int frame_count=initilization_frame_buffer_.size();
    int n_state = frame_count * 3 + 2 + 1;
    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    std::deque<Frame::Ptr>::iterator frame_i;
    std::deque<Frame::Ptr>::iterator frame_j;

    for(int k = 0; k < 4; k++)
    {
        MatrixXd lxly(3, 2);
        lxly = TangentBasis(g0);
        int i=0;
        for(frame_i=initilization_frame_buffer_.begin();frame_i!=initilization_frame_buffer_.end()-1;frame_i++,i++)
        {
            MatrixXd temp_A(6,9);
            temp_A.setZero();
            Eigen::VectorXd temp_b(6);
            temp_b.setZero();

            frame_j=next(frame_i);
            double frame_dt=(*frame_j)->preintegration->sum_t;
//        cout<<"sum_t:"<<frame_dt<<endl;

            Matrix3d eigen_Rb2c=eigen_Rc2b.transpose();
            Matrix3d Rwb_i=(*frame_i)->Twc().rotationMatrix()*eigen_Rb2c;
            Matrix3d Rwb_j=(*frame_j)->Twc().rotationMatrix()*eigen_Rb2c;
            Vector3d Pwc_i=(*frame_i)->Twc().translation();
            Vector3d Pwc_j=(*frame_j)->Twc().translation();

//        cout<<"Pwc_i:"<<endl<<Pwc_i<<endl;
//        cout<<"Pwc_j:"<<endl<<Pwc_j<<endl;

            temp_A.block(0,0,3,3)= -Matrix3d::Identity()*frame_dt;
            temp_A.block(0,6,3,2)= -0.5*Rwb_i.transpose()*frame_dt*frame_dt*lxly;
            temp_A.block(0,8,3,1)= Rwb_i.transpose()*(Pwc_j-Pwc_i)/100.0;
            temp_A.block(3,0,3,3)= -Matrix3d::Identity();
            temp_A.block(3,3,3,3)= Rwb_i.transpose()*Rwb_j;
            temp_A.block(3,6,3,2)= -Rwb_i.transpose()*frame_dt*lxly;

            temp_b.block(0,0,3,1)=(*frame_j)->preintegration->dp-eigen_tc2b+Rwb_i.transpose()*Rwb_j*eigen_tc2b+0.5*Rwb_i.transpose()*frame_dt*frame_dt*g0;
            temp_b.block(3,0,3,1)=(*frame_j)->preintegration->dv+Rwb_i.transpose()*frame_dt*g0;

            MatrixXd r_A = temp_A.transpose() * temp_A;
            VectorXd r_b = temp_A.transpose() * temp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            b.tail<3>() += r_b.tail<3>();

            A.block<6, 3>(i * 3, frame_count*3) += r_A.topRightCorner<6, 3>();
            A.block<3, 6>(frame_count*3, i * 3) += r_A.bottomLeftCorner<3, 6>();

        }
        A = A * 1000.0;
        b = b * 1000.0;
        x = A.ldlt().solve(b);
        VectorXd dg=x.segment<2>(frame_count*3);
        g0 = (g0 + lxly * dg).normalized() * G.norm();

        G=g0;
        cout<<"g:"<<g0.norm()<<"---:"<<g0.transpose()<<endl;
        double s=(x.tail<1>())(0)/100.0;
        cout<<"scale:"<<s<<endl;
    }
    for(int i=0;i<initilization_frame_buffer_.size();i++)
    {
        initilization_frame_buffer_[i]->v=x.segment<3>(i*3);
        initilization_frame_buffer_[i]->s=(x.tail<1>())(0)/100;
    }
/*
        ceres::Problem problem;
        double velocity_frame[frame_count][3];
        double gravity[3];
        double scale[1];
        double acc_bias[3]={0};
        Vector3d gravity_v;
        double s;
        gravity_v=g0;

//    ROS_ASSERT_MSG(gravity_v.norm()<10,"gravity is too large");
        gravity[0] = gravity_v(0);
        gravity[1] = gravity_v(1);
        gravity[2] = gravity_v(2);
        s = (x.tail<1>())(0)/100.0;
        scale[0] = s;

        for(int i;i<frame_count;i++)
        {
            velocity_frame[i][0]=initilization_frame_buffer_[i]->v.x();
            velocity_frame[i][1]=initilization_frame_buffer_[i]->v.y();
            velocity_frame[i][2]=initilization_frame_buffer_[i]->v.z();
            problem.AddParameterBlock(velocity_frame[i],3);
//            problem.SetParameterBlockConstant(velocity_frame[i]);
        }
        problem.AddParameterBlock(gravity,3);
        problem.AddParameterBlock(scale,1);
        problem.AddParameterBlock(acc_bias,3);
        problem.SetParameterBlockConstant(scale);
        problem.SetParameterBlockConstant(gravity);

        for(int i;i<frame_count-1;i++)
        {
            int j=i+1;
            Matrix3d eigen_Rb2c=eigen_Rc2b.transpose();
            Matrix3d Rbi_w= eigen_Rc2b*initilization_frame_buffer_[i]->Tcw().rotationMatrix();
            Matrix3d Rbj_w= eigen_Rc2b*initilization_frame_buffer_[j]->Tcw().rotationMatrix();
            Vector3d Pw_ci= initilization_frame_buffer_[i]->Twc().translation();
            Vector3d Pw_cj= initilization_frame_buffer_[j]->Twc().translation();
            Vector3d delta_a= initilization_frame_buffer_[j]->preintegration->dp;
            Vector3d delta_v= initilization_frame_buffer_[j]->preintegration->dv;
            Matrix3d jacobian_P_ba= initilization_frame_buffer_[j]->preintegration->jacobian_P_ba;
            Matrix3d jacobian_V_ba= initilization_frame_buffer_[j]->preintegration->jacobian_V_ba;
            double dt= initilization_frame_buffer_[j]->preintegration->sum_t;

            ssvo::ceres_slover::GVSError* GVSerror= new ssvo::ceres_slover::GVSError(Rbi_w,Rbj_w,Pw_ci,Pw_cj,delta_a,delta_v,jacobian_P_ba,jacobian_V_ba,dt);
            problem.AddResidualBlock(GVSerror,NULL,velocity_frame[i],velocity_frame[j],gravity,scale,acc_bias);

        }
        ceres::Solver::Options options;
        options.linear_solver_type= ceres::DENSE_SCHUR;
        options.trust_region_strategy_type=ceres::DOGLEG;
//        options.max_num_iterations=8;
        options.max_solver_time_in_seconds=0.2;
        options.minimizer_progress_to_stdout=true;
        ceres::Solver::Summary summary;
        ceres::Solve(options,&problem,&summary);

        cout << summary.BriefReport() << endl;
        ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
        cout<<"gravity:"<<gravity[0]<<" "<<gravity[1]<<" "<<gravity[2]<<endl;
        cout<<"scale:"<<scale[0]<<endl;
        cout<<"ba:"<<acc_bias[0]<<" "<<acc_bias[1]<<" "<<acc_bias[2]<<" "<<endl;

*/
//        waitKey(0);
    return true;
}

MatrixXd TangentBasis(Vector3d &g0)
{
        Vector3d b, c;
        Vector3d a = g0.normalized();
        Vector3d tmp(0, 0, 1);
        if(a == tmp)
            tmp << 1, 0, 0;
        b = (tmp - a * (a.transpose() * tmp)).normalized();//改成下面直接叉乘之后效果基本相同，没什么区别
        //    b = a.cross(tmp);
        c = a.cross(b);
        MatrixXd bc(3, 2);
        bc.block<3, 1>(0, 0) = b;
        bc.block<3, 1>(0, 1) = c;
        return bc;
    }

}