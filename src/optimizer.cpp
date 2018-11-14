#include <iomanip>
#include <include/optimizer.hpp>
#include "optimizer.hpp"
#include "config.hpp"
#include "utils.hpp"
#include <map>

namespace ssvo{

 MarginalizationInfo *last_marginalization_info;
 vector<double *> last_marginalization_parameter_blocks;

    void Optimizer::twoViewBundleAdjustment(const KeyFrame::Ptr &kf1, const KeyFrame::Ptr &kf2, bool report, bool verbose)
    {
        cout<<"-------------------------------------twoViewBundleAdjustment--------------------------------------"<<endl;
        kf1->optimal_Tcw_ = kf1->Tcw();
        kf2->optimal_Tcw_ = kf2->Tcw();

        kf1->optimal_Twb_ = kf1->Twb();
        kf2->optimal_Twb_ = kf2->Twb();

        cout<<" test id:"<<endl;
        cout<<kf1->frame_id_<<endl;
        cout<<kf2->frame_id_<<endl;

        ceres::Problem problem;
//    ceres::LocalParameterization* local_parameterization = new ceres_slover::SE3Parameterization();
//    problem.AddParameterBlock(kf1->optimal_Tcw_.data(), SE3d::num_parameters, local_parameterization);
//    problem.AddParameterBlock(kf2->optimal_Tcw_.data(), SE3d::num_parameters, local_parameterization);
//    problem.SetParameterBlockConstant(kf1->optimal_Tcw_.data());

        ceres::LocalParameterization* pvrpose = new PosePVR();
        problem.AddParameterBlock(kf1->PVR, 9, pvrpose);
        problem.AddParameterBlock(kf2->PVR, 9, pvrpose);
        problem.SetParameterBlockConstant(kf1->PVR);

        cout<<"kf1 pose------------------------------------------------------------BEFORE:"<<endl<<kf1->optimal_Tcw_.rotationMatrix()<<endl<<kf1->optimal_Tcw_.translation()<<endl;

        cout<<"kf2 pose------------------------------------------------------------BEFORE:"<<endl<<kf2->optimal_Tcw_.rotationMatrix()<<endl<<kf2->optimal_Tcw_.translation()<<endl;

        std::vector<Feature::Ptr> fts1;
        kf1->getFeatures(fts1);
        MapPoints mpts;

        for(const Feature::Ptr &ft1 : fts1)
        {
            MapPoint::Ptr mpt = ft1->mpt_;
            if(mpt == nullptr)//! should not happen
                continue;

            Feature::Ptr ft2 = mpt->findObservation(kf2);

            if(ft2 == nullptr || ft2->mpt_ == nullptr)
                continue;

            mpt->optimal_pose_ = mpt->pose();
            mpts.push_back(mpt);

            ceres::CostFunction* cost_function1 = ceres_slover::ReprojectionErrorSE3::Create(ft1->fn_[0]/ft1->fn_[2], ft1->fn_[1]/ft1->fn_[2]);//, 1.0/(1<<ft1->level_));
//        problem.AddResidualBlock(cost_function1, NULL, kf1->optimal_Tcw_.data(), mpt->optimal_pose_.data());
            problem.AddResidualBlock(cost_function1, NULL, kf1->PVR, mpt->optimal_pose_.data());

            ceres::CostFunction* cost_function2 = ceres_slover::ReprojectionErrorSE3::Create(ft2->fn_[0]/ft2->fn_[2], ft2->fn_[1]/ft2->fn_[2]);//, 1.0/(1<<ft2->level_));
//        problem.AddResidualBlock(cost_function2, NULL, kf2->optimal_Tcw_.data(), mpt->optimal_pose_.data());
            problem.AddResidualBlock(cost_function2, NULL, kf2->PVR, mpt->optimal_pose_.data());

        }


        ceres::Solver::Options options;
        ceres::Solver::Summary summary;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.minimizer_progress_to_stdout = report & verbose;
//    options_.gradient_tolerance = 1e-4;
//    options_.function_tolerance = 1e-4;
        //options_.max_solver_time_in_seconds = 0.2;

        ceres::Solve(options, &problem, &summary);

        //! update pose

        Vector3d t = Vector3d(kf2->PVR[0],kf2->PVR[1],kf2->PVR[2]);
        Vector3d w = Vector3d(kf2->PVR[6],kf2->PVR[7],kf2->PVR[8]);

        Matrix3d ttttt = Sophus_new::SO3::exp(w).matrix();

        SE3d tmp(ttttt,t);

        kf2->setTwb(tmp);

//    kf2->v = Vector3d(kf2->PVR[3],kf2->PVR[4],kf2->PVR[5]);

        //todo
//    kf2->updatePoseAndBias();

        kf1->optimal_Tcw_ = kf1->Tcw();
        kf2->optimal_Tcw_ = kf2->Tcw();

        cout<<"kf1 pose------------------------------------------------------------after:"<<endl<<kf1->optimal_Tcw_.rotationMatrix()<<endl<<kf1->optimal_Tcw_.translation()<<endl;
        cout<<"kf2 pose------------------------------------------------------------after:"<<endl<<kf2->optimal_Tcw_.rotationMatrix()<<endl<<kf2->optimal_Tcw_.translation()<<endl;
//    kf2->setTcw(kf2->optimal_Tcw_);

        std::for_each(mpts.begin(), mpts.end(), [](MapPoint::Ptr mpt){mpt->setPose(mpt->optimal_pose_);});

        //! Report
        reportInfo(problem, summary, report, verbose);

//    waitKey(0);
    }

    void Optimizer::localBundleAdjustment(const KeyFrame::Ptr &keyframe, std::list<MapPoint::Ptr> &bad_mpts, int size, int min_shared_fts, bool report, bool verbose)
    {
        //! 在vio初始化之前使用localba
        if(keyframe->id_ <= WINDOW_SIZE)
        {
            cout<<"-------------------------------------localBundleAdjustment--------------------------------------"<<endl;
            double t0 = (double)cv::getTickCount();
            size = size > 0 ? size-1 : 0;
            std::set<KeyFrame::Ptr> actived_keyframes = keyframe->getConnectedKeyFrames(size, min_shared_fts);
            actived_keyframes.insert(keyframe);
            std::unordered_set<MapPoint::Ptr> local_mappoints;
            std::set<KeyFrame::Ptr> fixed_keyframe;

            for(const KeyFrame::Ptr &kf : actived_keyframes)
            {
                MapPoints mpts;
                kf->getMapPoints(mpts);
                for(const MapPoint::Ptr &mpt : mpts)
                {
                    local_mappoints.insert(mpt);
                }
            }

            for(const MapPoint::Ptr &mpt : local_mappoints)
            {
                const std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();
                for(const auto &item : obs)
                {
                    if(actived_keyframes.count(item.first))
                        continue;

                    fixed_keyframe.insert(item.first);
                }
            }

            ceres::Problem problem;
//    ceres::LocalParameterization* local_parameterization = new ceres_slover::SE3Parameterization();

//    for(const KeyFrame::Ptr &kf : fixed_keyframe)
//    {
//        kf->optimal_Tcw_ = kf->Tcw();
//        problem.AddParameterBlock(kf->optimal_Tcw_.data(), SE3d::num_parameters, local_parameterization);
//        problem.SetParameterBlockConstant(kf->optimal_Tcw_.data());
//    }
//
//    for(const KeyFrame::Ptr &kf : actived_keyframes)
//    {
//        kf->optimal_Tcw_ = kf->Tcw();
//        problem.AddParameterBlock(kf->optimal_Tcw_.data(), SE3d::num_parameters, local_parameterization);
//        if(kf->id_ <= 1)
//            problem.SetParameterBlockConstant(kf->optimal_Tcw_.data());
//    }


            ceres::LocalParameterization* pvrpose = new PosePVR();


            for(const KeyFrame::Ptr &kf : fixed_keyframe)
            {
                problem.AddParameterBlock(kf->PVR, 9, pvrpose);
                problem.SetParameterBlockConstant(kf->PVR);
            }

            for(const KeyFrame::Ptr &kf : actived_keyframes)
            {
//        kf->optimal_Tcw_ = kf->Tcw();
                problem.AddParameterBlock(kf->PVR, 9, pvrpose);
                if(kf->id_ <= 1)
                    problem.SetParameterBlockConstant(kf->PVR);
                problem.SetParameterBlockConstant(kf->PVR);
            }

            double scale = Config::imagePixelUnSigma() * 2;
            ceres::LossFunction* lossfunction = new ceres::HuberLoss(scale);
            for(const MapPoint::Ptr &mpt : local_mappoints)
            {
                mpt->optimal_pose_ = mpt->pose();
                const std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();

                for(const auto &item : obs)
                {
                    const KeyFrame::Ptr &kf = item.first;
                    const Feature::Ptr &ft = item.second;
                    ceres::CostFunction* cost_function1 = ceres_slover::ReprojectionErrorSE3::Create(ft->fn_[0]/ft->fn_[2], ft->fn_[1]/ft->fn_[2]);//, 1.0/(1<<ft->level_));
//            problem.AddResidualBlock(cost_function1, lossfunction, kf->optimal_Tcw_.data(), mpt->optimal_pose_.data());
                    problem.AddResidualBlock(cost_function1, lossfunction, kf->PVR, mpt->optimal_pose_.data());


                }
            }

            ceres::Solver::Options options;
            ceres::Solver::Summary summary;
            options.linear_solver_type = ceres::DENSE_SCHUR;
            options.minimizer_progress_to_stdout = report & verbose;

            ceres::Solve(options, &problem, &summary);

            //! update pose
//    for(const KeyFrame::Ptr &kf : actived_keyframes)
//    {
//        kf->setTcw(kf->optimal_Tcw_);
//    }

            for(const KeyFrame::Ptr &kf : actived_keyframes)
            {
                kf->updatePose();
            }

            //! update mpts & remove mappoint with large error
            std::set<KeyFrame::Ptr> changed_keyframes;
            static const double max_residual = Config::imagePixelUnSigma2() * std::sqrt(3.81);
            for(const MapPoint::Ptr &mpt : local_mappoints)
            {
                const std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();
                for(const auto &item : obs)
                {
                    double residual = utils::reprojectError(item.second->fn_.head<2>(), item.first->Tcw(), mpt->optimal_pose_);
                    if(residual < max_residual)
                        continue;

                    mpt->removeObservation(item.first);
                    changed_keyframes.insert(item.first);
//            std::cout << " rm outlier: " << mpt->id_ << " " << item.first->id_ << " " << obs.size() << std::endl;

                    if(mpt->type() == MapPoint::BAD)
                    {
                        bad_mpts.push_back(mpt);
                    }
                }

                mpt->setPose(mpt->optimal_pose_);
            }

            for(const KeyFrame::Ptr &kf : changed_keyframes)
            {
                kf->updateConnections();
            }

            //! Report
            double t1 = (double)cv::getTickCount();
            LOG_IF(INFO, report) << "[Optimizer] Finish local BA for KF: " << keyframe->id_ << "(" << keyframe->frame_id_ << ")"
                                 << ", KFs: " << actived_keyframes.size() << "(+" << fixed_keyframe.size() << ")"
                                 << ", Mpts: " << local_mappoints.size()
                                 << ", remove " << bad_mpts.size() << " bad mpts."
                                 << " (" << (t1-t0)/cv::getTickFrequency() << "ms)";

            reportInfo(problem, summary, report, verbose);
        }


    }
    /*
//void Optimizer::localBundleAdjustmentWithInvDepth(const KeyFrame::Ptr &keyframe, std::list<MapPoint::Ptr> &bad_mpts, int size, bool report, bool verbose)
//{
//    double t0 = (double)cv::getTickCount();
//    std::set<KeyFrame::Ptr> actived_keyframes = keyframe->getConnectedKeyFrames(size);
//    actived_keyframes.insert(keyframe);
//    std::unordered_set<MapPoint::Ptr> local_mappoints;
//    std::list<KeyFrame::Ptr> fixed_keyframe;
//
//    for(const KeyFrame::Ptr &kf : actived_keyframes)
//    {
//        MapPoints mpts;
//        kf->getMapPoints(mpts);
//        for(const MapPoint::Ptr &mpt : mpts)
//        {
//            local_mappoints.insert(mpt);
//        }
//    }
//
//    for(const MapPoint::Ptr &mpt : local_mappoints)
//    {
//        const std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();
//        for(const auto &item : obs)
//        {
//            if(actived_keyframes.count(item.first))
//                continue;
//
//            fixed_keyframe.push_back(item.first);
//        }
//    }
//
//    ceres::Problem problem;
//    ceres::LocalParameterization* local_parameterization = new ceres_slover::SE3Parameterization();
//
//    std::unordered_set<KeyFrame::Ptr> local_keyframes;
//    for(const KeyFrame::Ptr &kf : fixed_keyframe)
//    {
//        local_keyframes.insert(kf);
//        kf->optimal_Tcw_ = kf->Tcw();
//        problem.AddParameterBlock(kf->optimal_Tcw_.data(), SE3d::num_parameters, local_parameterization);
//        problem.SetParameterBlockConstant(kf->optimal_Tcw_.data());
//    }
//
//    for(const KeyFrame::Ptr &kf : actived_keyframes)
//    {
//        local_keyframes.insert(kf);
//        kf->optimal_Tcw_ = kf->Tcw();
//        problem.AddParameterBlock(kf->optimal_Tcw_.data(), SE3d::num_parameters, local_parameterization);
//        if(kf->id_ <= 1)
//            problem.SetParameterBlockConstant(kf->optimal_Tcw_.data());
//    }
//
//    double scale = Config::pixelUnSigma() * 2;
//    ceres::LossFunction* lossfunction = new ceres::HuberLoss(scale);
//    std::unordered_map<MapPoint::Ptr, KeyFrame::Ptr> optimal_invdepth_mappoints;
//    for(const MapPoint::Ptr &mpt : local_mappoints)
//    {
//        KeyFrame::Ptr ref_kf = mpt->getReferenceKeyFrame();
//        if(local_keyframes.count(ref_kf))
//        {
//            optimal_invdepth_mappoints.emplace(mpt, ref_kf);
//            Vector3d pose = ref_kf->Tcw() * mpt->pose();
//            mpt->optimal_pose_ << pose[0]/pose[2], pose[1]/pose[2], 1.0/pose[2];
//            const std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();
//            const Feature::Ptr ref_ft = obs.find(ref_kf)->second;
//            for(const auto &item : obs)
//            {
//                const KeyFrame::Ptr &kf = item.first;
//                if(kf == ref_kf)
//                    continue;
//
//                const Feature::Ptr &ft = item.second;
//                ceres::CostFunction* cost_function1 = ceres_slover::ReprojectionErrorSE3InvDepth::Create(
//                    ref_ft->fn_[0]/ref_ft->fn_[2], ref_ft->fn_[1]/ref_ft->fn_[2],
//                    ft->fn_[0]/ft->fn_[2], ft->fn_[1]/ft->fn_[2]);
//                problem.AddResidualBlock(cost_function1, lossfunction, ref_kf->optimal_Tcw_.data(), kf->optimal_Tcw_.data(), mpt->optimal_pose_.data());
//            }
//        }
//        else
//        {
//            mpt->optimal_pose_ = mpt->pose();
//            const std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();
//            for(const auto &item : obs)
//            {
//                const KeyFrame::Ptr &kf = item.first;
//                const Feature::Ptr &ft = item.second;
//                ceres::CostFunction* cost_function1 = ceres_slover::ReprojectionErrorSE3::Create(ft->fn_[0]/ft->fn_[2], ft->fn_[1]/ft->fn_[2]);
//                problem.AddResidualBlock(cost_function1, lossfunction, kf->optimal_Tcw_.data(), mpt->optimal_pose_.data());
//            }
//        }
//    }
//
//    ceres::Solver::Options options;
//    ceres::Solver::Summary summary;
//    options.linear_solver_type = ceres::DENSE_SCHUR;
//    options.minimizer_progress_to_stdout = true;//report & verbose;
//
//    ceres::Solve(options, &problem, &summary);
//
//    //! update pose
//    for(const KeyFrame::Ptr &kf : actived_keyframes)
//    {
//        kf->setTcw(kf->optimal_Tcw_);
//    }
//
//    //! update mpts & remove mappoint with large error
//    std::set<KeyFrame::Ptr> changed_keyframes;
//    double max_residual = Config::pixelUnSigma2() * 2;
//    for(const MapPoint::Ptr &mpt : local_mappoints)
//    {
//        if(optimal_invdepth_mappoints.count(mpt))
//        {
//            KeyFrame::Ptr ref_kf = optimal_invdepth_mappoints[mpt];
//            Feature::Ptr ref_ft = ref_kf->getFeatureByMapPoint(mpt);
//            mpt->optimal_pose_ << mpt->optimal_pose_[0]/mpt->optimal_pose_[2], mpt->optimal_pose_[1]/mpt->optimal_pose_[2], 1.0/mpt->optimal_pose_[2];
//            mpt->optimal_pose_ = ref_kf->Twc() * mpt->optimal_pose_;
//        }
//
//        const std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();
//        for(const auto &item : obs)
//        {
//            double residual = utils::reprojectError(item.second->fn_.head<2>(), item.first->Tcw(), mpt->optimal_pose_);
//            if(residual < max_residual)
//                continue;
//
//            mpt->removeObservation(item.first);
//            changed_keyframes.insert(item.first);
////            std::cout << " rm outlier: " << mpt->id_ << " " << item.first->id_ << " " << obs.size() << std::endl;
//
//            if(mpt->type() == MapPoint::BAD)
//            {
//                bad_mpts.push_back(mpt);
//            }
//        }
//
//        mpt->setPose(mpt->optimal_pose_);
//    }
//
//    for(const KeyFrame::Ptr &kf : changed_keyframes)
//    {
//        kf->updateConnections();
//    }
//
//    //! Report
//    double t1 = (double)cv::getTickCount();
//    LOG_IF(INFO, report) << "[Optimizer] Finish local BA for KF: " << keyframe->id_ << "(" << keyframe->frame_id_ << ")"
//                         << ", KFs: " << actived_keyframes.size()
//                         << ", Mpts: " << local_mappoints.size() << "(" << optimal_invdepth_mappoints.size() << ")"
//                         << ", remove " << bad_mpts.size() << " bad mpts."
//                         << " (" << (t1-t0)/cv::getTickFrequency() << "ms)";
//
//    reportInfo(problem, summary, report, verbose);
//}
    */

    void Optimizer::motionOnlyBundleAdjustment(const Frame::Ptr &last_frame,const Frame::Ptr &frame, bool use_seeds, bool vio, bool reject, bool report, bool verbose)
    {

        cout<<"-------------------------------------motionOnlyBundleAdjustment--------------------------------------"<<endl;
        static const size_t OPTIMAL_MPTS = 150;

//    frame->optimal_Tcw_ = frame->Tcw();

        ceres::Problem problem;
        ceres::LocalParameterization* pvrpose = new PosePVR();

        ceres::LocalParameterization* biaspose = new PoseBias();
        ceres::LocalParameterization* biaspose1 = new PoseBias();
        problem.AddParameterBlock(frame->PVR, 9, pvrpose);

        //!by jh
        last_frame->bgba[0] = last_frame->preintegration->bg.x();
        last_frame->bgba[1] = last_frame->preintegration->bg.y();
        last_frame->bgba[2] = last_frame->preintegration->bg.z();
        last_frame->bgba[3] = last_frame->preintegration->ba.x();
        last_frame->bgba[4] = last_frame->preintegration->ba.y();
        last_frame->bgba[5] = last_frame->preintegration->ba.z();

        frame->bgba[0] = frame->preintegration->bg.x();
        frame->bgba[1] = frame->preintegration->bg.y();
        frame->bgba[2] = frame->preintegration->bg.z();
        frame->bgba[3] = frame->preintegration->ba.x();
        frame->bgba[4] = frame->preintegration->ba.y();
        frame->bgba[5] = frame->preintegration->ba.z();

        problem.AddParameterBlock(last_frame->bgba,6,biaspose);
        problem.AddParameterBlock(frame->bgba,6,biaspose1);

        static const double scale = Config::imagePixelUnSigma() * std::sqrt(3.81);
        ceres::LossFunction* lossfunction = new ceres::HuberLoss(scale);

        std::vector<Feature::Ptr> fts;
        frame->getFeatures(fts);
        const size_t N = fts.size();
        std::vector<ceres::ResidualBlockId> res_ids(N);

        for(size_t i = 0; i < N; ++i)
        {
            Feature::Ptr ft = fts[i];
            MapPoint::Ptr mpt = ft->mpt_;
            if(mpt == nullptr)
                continue;

            mpt->optimal_pose_ = mpt->pose();
            ceres::CostFunction* cost_function = ceres_slover::ReprojectionErrorSE3::Create(ft->fn_[0]/ft->fn_[2], ft->fn_[1]/ft->fn_[2]);//, 1.0/(1<<ft->level_));
            res_ids[i] = problem.AddResidualBlock(cost_function, lossfunction, frame->PVR, mpt->optimal_pose_.data());
            problem.SetParameterBlockConstant(mpt->optimal_pose_.data());

        }

//    if(N < OPTIMAL_MPTS)
        //! false 不用种子点，用了效果不好
        if(use_seeds)
        {
            std::vector<Feature::Ptr> ft_seeds;
            frame->getSeeds(ft_seeds);
            const size_t needed = OPTIMAL_MPTS - N;
            if(ft_seeds.size() > needed)
            {
                std::nth_element(ft_seeds.begin(), ft_seeds.begin()+needed, ft_seeds.end(),
                                 [](const Feature::Ptr &a, const Feature::Ptr &b)
                                 {
                                     return a->seed_->getInfoWeight() > b->seed_->getInfoWeight();
                                 });

                ft_seeds.resize(needed);
            }

            const size_t M = ft_seeds.size();
            res_ids.resize(N+M);
            for(int i = 0; i < M; ++i)
            {
                Feature::Ptr ft = ft_seeds[i];
                Seed::Ptr seed = ft->seed_;
                if(seed == nullptr)
                    continue;

                seed->optimal_pose_.noalias() = seed->kf->Twc() * (seed->fn_ref / seed->getInvDepth());

                ceres::CostFunction* cost_function = ceres_slover::ReprojectionErrorSE3::Create(seed->fn_ref[0]/seed->fn_ref[2], seed->fn_ref[1]/seed->fn_ref[2], seed->getInfoWeight());
//            res_ids[i] = problem.AddResidualBlock(cost_function, lossfunction, frame->optimal_Tcw_.data(), seed->optimal_pose_.data());
                res_ids[i] = problem.AddResidualBlock(cost_function, lossfunction, frame->PVR, seed->optimal_pose_.data());
                problem.SetParameterBlockConstant(seed->optimal_pose_.data());

            }
        }

        /*
        if(vio)
        {
            //! imu误差
            ceres::LocalParameterization* pvrpose1 = new PosePVR();
            problem.AddParameterBlock(last_frame->PVR,9,pvrpose1);
            problem.SetParameterBlockConstant(last_frame->PVR);
            ceres::CostFunction* imu_factor = new ceres_slover::IMUError(last_frame,frame);
            problem.AddResidualBlock(imu_factor,NULL,last_frame->PVR,last_frame->bgba,frame->PVR,frame->bgba);
            //! bias误差
            ceres::CostFunction* bias_factor = new ceres_slover::BiasError(frame->preintegration);
            problem.AddResidualBlock(bias_factor,NULL,last_frame->bgba,frame->bgba);
        }
         */

        ceres::Solver::Options options;
        ceres::Solver::Summary summary;
        options.linear_solver_type = ceres::DENSE_SCHUR;
//    options.trust_region_strategy_type = ceres::DOGLEG;
        options.minimizer_progress_to_stdout = report & verbose;
        options.max_linear_solver_iterations = 20;

        ceres::Solve(options, &problem, &summary);

//        cout << summary.FullReport() << endl;
//
//        waitKey(0);

        /*
//        double sum=0;
//        int l=0;
//        std::vector<ceres::ResidualBlockId> ids;
//        problem.GetResidualBlocks(&ids);
//        for (auto & id: ids)
//        {
//            cout<<l<<":   ";
//            sum += reprojectionError(problem, id).transpose() * reprojectionError(problem, id);
//            cout<<reprojectionError(problem, id).transpose() * reprojectionError(problem, id)<<endl;
//            l++;
//        }
//
//    cout<<"sum: "<<sum<<endl;
//    cout<<"视觉误差量个数："<<N<<endl;
//    cout << summary.FullReport() << endl;
//    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
//        waitKey(0);
         */
        //todo 改了特征点的权重之后先不要加这个东西！！！
        if(reject)
        {
            int remove_count = 0;

            static const double TH_REPJ = 3.81 * Config::imagePixelUnSigma2();
            for(size_t i = 0; i < N; ++i)
            {
                Feature::Ptr ft = fts[i];
                if(reprojectionError(problem, res_ids[i]).squaredNorm() > TH_REPJ * (1 << ft->level_))
                {
                    remove_count++;
                    problem.RemoveResidualBlock(res_ids[i]);
                    frame->removeFeature(ft);
                }
            }

            ceres::Solve(options, &problem, &summary);

            LOG_IF(WARNING, report) << "[Optimizer] Motion-only BA removes " << remove_count << " points";
        }

        //! update pose
        frame->updatePose();
        //! Report
//    reportInfo(problem, summary, report, verbose);
/*
        double sum=0;
        int l=0;
        std::vector<ceres::ResidualBlockId> ids;
        problem.GetResidualBlocks(&ids);
        std::vector<ceres::ResidualBlockId>::iterator iter;
        for (iter=ids.begin()+1;iter!=ids.end();iter++)
        {
            problem.RemoveResidualBlock(*iter);
        }
        std::vector<ceres::ResidualBlockId> ids2;
        problem.GetResidualBlocks(&ids2);
        for (auto & id: ids2)
        {
            cout<<l<<":   ";
            sum += reprojectionError(problem, id).transpose() * reprojectionError(problem, id);
            cout<<reprojectionError(problem, id).transpose() * reprojectionError(problem, id)<<endl;
            l++;
        }
        cout<<"init sum: "<<sum<<endl;
        ceres::Solve(options, &problem, &summary);
        sum = 0; l = 0;
        std::vector<ceres::ResidualBlockId> ids1;
        problem.GetResidualBlocks(&ids1);
        for (auto & id: ids1)
        {
            cout<<l<<":   ";
            sum += reprojectionError(problem, id).transpose() * reprojectionError(problem, id);
            cout<<reprojectionError(problem, id).transpose() * reprojectionError(problem, id)<<endl;
            l++;
        }
        cout<<"sum: "<<sum<<endl;
//        cout<<"视觉误差量个数："<<N<<endl;
        cout << summary.FullReport() << endl;
        */

    }

    void Optimizer::refineMapPoint(const MapPoint::Ptr &mpt, int max_iter, bool report, bool verbose)
    {

#if 0
        ceres::Problem problem;
    double scale = Config::pixelUnSigma() * 2;
    ceres::LossFunction* lossfunction = new ceres::HuberLoss(scale);

    //! add obvers kf
    const std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();
    const KeyFrame::Ptr kf_ref = mpt->getReferenceKeyFrame();

    mpt->optimal_pose_ = mpt->pose();

    for(const auto &obs_item : obs)
    {
        const KeyFrame::Ptr &kf = obs_item.first;
        const Feature::Ptr &ft = obs_item.second;
        kf->optimal_Tcw_ = kf->Tcw();

        ceres::CostFunction* cost_function = ceres_slover::ReprojectionErrorSE3::Create(ft->fn_[0]/ft->fn_[2], ft->fn_[1]/ft->fn_[2]);//, 1.0/(1<<ft->level_));
        problem.AddResidualBlock(cost_function, lossfunction, kf->optimal_Tcw_.data(), mpt->optimal_pose_.data());
        problem.SetParameterBlockConstant(kf->optimal_Tcw_.data());
    }

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = report & verbose;
    options.max_linear_solver_iterations = max_iter;

    ceres::Solve(options, &problem, &summary);

    mpt->setPose(mpt->optimal_pose_);

    reportInfo(problem, summary, report, verbose);
#else

        double t0 = (double)cv::getTickCount();
        mpt->optimal_pose_ = mpt->pose();
        Vector3d pose_last = mpt->optimal_pose_;
        const std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();
        const size_t n_obs = obs.size();

        Matrix3d A;
        Vector3d b;
        double init_chi2 = std::numeric_limits<double>::max();
        double last_chi2 = std::numeric_limits<double>::max();
        const double EPS = 1E-10;

        const bool progress_out = report&verbose;
        bool convergence = false;
        int i = 0;
        for(; i < max_iter; ++i)
        {
            A.setZero();
            b.setZero();
            double new_chi2 = 0.0;

            //! compute res
            for(const auto &obs_item : obs)
            {
                const SE3d Tcw = obs_item.first->Tcw();
                const Vector2d fn = obs_item.second->fn_.head<2>();

                const Vector3d point(Tcw * mpt->optimal_pose_);
                const Vector2d resduial(point.head<2>()/point[2] - fn);

                new_chi2 += resduial.squaredNorm();

                Eigen::Matrix<double, 2, 3> Jacobain;

                const double z_inv = 1.0 / point[2];
                const double z_inv2 = z_inv*z_inv;
                Jacobain << z_inv, 0.0, -point[0]*z_inv2, 0.0, z_inv, -point[1]*z_inv2;

                Jacobain = Jacobain * Tcw.rotationMatrix();

                A.noalias() += Jacobain.transpose() * Jacobain;
                b.noalias() -= Jacobain.transpose() * resduial;
            }

            if(i == 0)  {init_chi2 = new_chi2;}

            if(last_chi2 < new_chi2)
            {
                LOG_IF(INFO, progress_out) << "iter " << std::setw(2) << i << ": failure, chi2: " << std::scientific << std::setprecision(6) << new_chi2/n_obs;
                mpt->setPose(pose_last);
                return;
            }

            last_chi2 = new_chi2;

            const Vector3d dp(A.ldlt().solve(b));

            pose_last = mpt->optimal_pose_;
            mpt->optimal_pose_.noalias() += dp;

            LOG_IF(INFO, progress_out) << "iter " << std::setw(2) << i << ": success, chi2: " << std::scientific << std::setprecision(6) << new_chi2/n_obs << ", step: " << dp.transpose();

            if(dp.norm() <= EPS)
            {
                convergence = true;
                break;
            }
        }

        mpt->setPose(mpt->optimal_pose_);
        double t1 = (double)cv::getTickCount();
        LOG_IF(INFO, report) << std::scientific  << "[Optimizer] MapPoint " << mpt->id_
                             << " Error(MSE) changed from " << std::scientific << init_chi2/n_obs << " to " << last_chi2/n_obs
                             << "(" << obs.size() << "), time: " << std::fixed << (t1-t0)*1000/cv::getTickFrequency() << "ms, "
                             << (convergence? "Convergence" : "Unconvergence");

#endif
    }

    Vector2d Optimizer::reprojectionError(const ceres::Problem& problem, ceres::ResidualBlockId id)
    {
        auto cost = problem.GetCostFunctionForResidualBlock(id);
        std::vector<double*> parameterBlocks;
        problem.GetParameterBlocksForResidualBlock(id, &parameterBlocks);
        Vector2d residual;
        cost->Evaluate(parameterBlocks.data(), residual.data(), nullptr);
        return residual;
    }

    void Optimizer::reportInfo(const ceres::Problem &problem, const ceres::Solver::Summary summary, bool report, bool verbose)
    {
        if(!report) return;

        if(!verbose)
        {
            LOG(INFO) << summary.BriefReport();
        }
        else
        {
            LOG(INFO) << summary.FullReport();
            std::vector<ceres::ResidualBlockId> ids;
            problem.GetResidualBlocks(&ids);
            for (size_t i = 0; i < ids.size(); ++i)
            {
                LOG(INFO) << "BlockId: " << std::setw(5) << i <<" residual(RMSE): " << reprojectionError(problem, ids[i]).norm();
            }
        }
    }

    void Optimizer::slideWindowJointOptimization(vector<Frame::Ptr> &all_frame_buffer, uint64_t *frame_id_window,System::System_Status system_status)
    {

        //! 相机位姿、特征点pose设置没有问题

        cout<<"-------------------------------------slideWindowJointOptimization--------------------------------------"<<endl;

        //! 检查滑动窗口中的帧是不是正确的
        LOG_ASSERT(all_frame_buffer.back()->id_ == frame_id_window[WINDOW_SIZE]);
        cout<<"frame in Window: ";
        for(int i = 0; i < WINDOW_SIZE+1; i++)
        {
            int id = frame_id_window[i];
            cout<<" "<<id;
        }
        cout<<endl;
        cout<<"Keyframe in Window: ";
        for(int i = 0; i < WINDOW_SIZE+1; i++)
        {
            int id = all_frame_buffer[frame_id_window[i]]->getRefKeyFrame()->id_;
            cout<<" "<<id;
        }
        cout<<endl;
        cout<<"mappoint in the lastest frame: "<<all_frame_buffer[frame_id_window[WINDOW_SIZE]]->featureNumber()<<endl;

        //! 将前9帧的普通帧的位姿更新成其自身关键帧的位姿
        for(int i = 0; i < WINDOW_SIZE; i++)
        {
            int id = frame_id_window[i];
            all_frame_buffer[id]->setPose(all_frame_buffer[id]->getRefKeyFrame()->pose());
            LOG_ASSERT(all_frame_buffer[id]->getRefKeyFrame()->frame_id_==id);
        }

        //! 将滑窗中所有帧的mapPoint的观测统一添加到Infos中，同时记录mappont的ID和观测次数
        vector<uint64_t > mapPointIdInWindow;
        map<uint64_t , int> mpt_obsTimes;
        vector<Info> Infos;
        static const double max_residual = Config::imagePixelUnSigma2() * std::sqrt(3.81);

        for(int i = 0; i < WINDOW_SIZE+1; i++)
        {
            Frame::Ptr frame = all_frame_buffer[frame_id_window[i]];
            MapPoints mpts;
            Features fts;
            frame->getMapPointsAndFeatures(mpts,fts);

            MapPoints::iterator iterator1;
            Features::iterator iterator2;
            LOG_ASSERT(mpts.size()==fts.size());
            for(iterator1 = mpts.begin(),iterator2=fts.begin();iterator1!=mpts.end();iterator1++,iterator2++)
            {
                MapPoint::Ptr mpt = *iterator1;
                Feature::Ptr ft = *iterator2;
                //! 如果mappoint质量不好就不要
//                if(mpt->isBad())
//                    continue;
                Info info;
                info.isbad = false;
                info.mpt = mpt;
                info.frame = frame;
                info.feature = ft;
                Infos.push_back(info);
                //todo
                //如果还没有添加过该mpt，就添加，否则观测次数增加
                if(find(mapPointIdInWindow.begin(),mapPointIdInWindow.end(),mpt->id_)==mapPointIdInWindow.end())
                {
                    mapPointIdInWindow.push_back(mpt->id_);
                    mpt_obsTimes[mpt->id_] = 1;
                }
                else
                {
                    mpt_obsTimes[mpt->id_]++;
                }
                //todo记录更多的信息
            }
        }

        cout<<"Infos size-----------------------------------------------------------------: "<<Infos.size()<<endl;
        //! 构造problem
        ceres::Problem problem;
        double scale = Config::imagePixelUnSigma() * 2;
        //todo 理解不同的鲁棒核函数的区别
        ceres::LossFunction* lossfunction = new ceres::HuberLoss(scale);

//        if (last_marginalization_info)
//        {
//            // construct new marginlization_factor
//            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
//            problem.AddResidualBlock(marginalization_factor, NULL, last_marginalization_parameter_blocks);
//        }

        for(int i = 0; i < WINDOW_SIZE+1; i++)
        {
            ceres::LocalParameterization* pvrpose = new PosePVR();
            int id = frame_id_window[i];
            problem.AddParameterBlock(all_frame_buffer[id]->PVR,9,pvrpose);
            //todo test
//            if(i< WINDOW_SIZE)
//                problem.SetParameterBlockConstant(all_frame_buffer[id]->PVR);
            if(all_frame_buffer[id]->getRefKeyFrame()->id_<=1)
            {
                problem.SetParameterBlockConstant(all_frame_buffer[id]->PVR);
            }
        }
        int residual_num = 0;
        for(auto &info:Infos)
        {
            MapPoint::Ptr mpt = info.mpt;
            Frame::Ptr frame = info.frame;
            Feature::Ptr feature = info.feature;
            LOG_ASSERT(mpt->id_ == feature->mpt_->id_);
            mpt->optimal_pose_ = mpt->pose();
            mpt->pose_double[0] = mpt->pose().x();mpt->pose_double[1] = mpt->pose().y();mpt->pose_double[2] = mpt->pose().z();

            info.ResID = nullptr;
//            if(mpt->isSoBad())continue;
            if(mpt_obsTimes[info.mpt->id_]<2 || info.mpt->getReferenceKeyFrame()->frame_id_>=frame_id_window[WINDOW_SIZE-2])
            {
                continue;
            }

            ceres::CostFunction* cost_function = ceres_slover::ReprojectionErrorSE3::Create(feature->fn_[0]/feature->fn_[2], feature->fn_[1]/feature->fn_[2]);//, 1.0/(1<<ft->level_));
            info.ResID = problem.AddResidualBlock(cost_function, lossfunction, frame->PVR, mpt->pose_double);
//            info.ResID = problem.AddResidualBlock(cost_function, lossfunction, frame->PVR, mpt->optimal_pose_.data());
//            problem.SetParameterBlockConstant(mpt->pose_double);
            residual_num ++;
        }

//        waitKey(0);
        std::set<KeyFrame::Ptr> changed_keyframes;
        double TH_REPJ = 3.81 * Config::imagePixelUnSigma2();
        for(auto &info:Infos)
        {
            if(info.ResID == nullptr)continue;
            MapPoint::Ptr mpt = info.mpt;
            Frame::Ptr frame = info.frame;
            Feature::Ptr feature = info.feature;

//            if(frame->id_ == frame_id_window[WINDOW_SIZE]) TH_REPJ *= 2;

            if(reprojectionError(problem, info.ResID).squaredNorm() > TH_REPJ * (1 << feature->level_))
            {
                info.isbad = true;
                std::cout << " rm outlier: mpt " << mpt->id_ << " in " << frame->id_ << " error " << reprojectionError(problem, info.ResID).squaredNorm() * 460 * 460 << std::endl;

                problem.RemoveResidualBlock(info.ResID);
                if(mpt->type() == MapPoint::BAD)
                {
                    //todo 需要重新考虑一下
//                    mpt->setRemove();
                    system_status.BadPoints.push_back(mpt);
                }
            }

        }



        //! add imu
        /*
        for(int i = 0; i < WINDOW_SIZE; i++)
        {
            int id_i = frame_id_window[i];
            int id_j = frame_id_window[i+1];

            cout<<"imu residual id_i: "<<id_i<<endl;
            cout<<"imu residual id_j: "<<id_j<<endl;
            Frame::Ptr last_frame = all_frame_buffer[id_i];
            Frame::Ptr frame = all_frame_buffer[id_j];
            //! imu误差
            ceres::CostFunction* imu_factor = new ceres_slover::IMUError(last_frame,frame);
            problem.AddResidualBlock(imu_factor,NULL,last_frame->PVR,last_frame->bgba,frame->PVR,frame->bgba);

            //! bias误差
            ceres::CostFunction* bias_factor = new ceres_slover::BiasError(frame->preintegration);
            problem.AddResidualBlock(bias_factor,NULL,last_frame->bgba,frame->bgba);
        }
         */

        double sum=0;
        int l=0;
        std::vector<ceres::ResidualBlockId> ids;
        problem.GetResidualBlocks(&ids);
        for (auto & id: ids)
        {
            cout<<l<<":   ";
            sum += (reprojectionError(problem, id).transpose() * reprojectionError(problem, id));
            cout<< (reprojectionError(problem, id).transpose() * reprojectionError(problem, id))* 460 * 460<<endl;
            l++;
        }
//        cout<<"residual num: "<<residual_num<<endl;
        cout<<"sum "<<l<<" "<<sum* 460 * 460<<endl;
//        cout<<"视觉误差量个数："<<N<<endl;

        //todo 先验误差、闭环误差

        ceres::Solver::Options options;
        ceres::Solver::Summary summary;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.max_linear_solver_iterations = 20;

        ceres::Solve(options, &problem, &summary);
        cout << summary.FullReport() << endl;
//        waitKey(0);

        //todo 这个是优化后的误差。。
        /*
        ids.clear();
        problem.GetResidualBlocks(&ids);

        static const double TH_REPJ = 3.81 * Config::imagePixelUnSigma2();
        for (auto & id: ids)
        {
            if(reprojectionError(problem, id).squaredNorm() > TH_REPJ)
            {
                problem.RemoveResidualBlock(id);
            }
        }

        ceres::Solve(options, &problem, &summary);
        cout << summary.FullReport() << endl;
        waitKey(0);
         */

//        all_frame_buffer[frame_id_window[10]]->updatePose();
        for(int i=0;i<WINDOW_SIZE+1;i++)
        {
            int id = frame_id_window[i];
            all_frame_buffer[id]->updatePose();

            //!注意更新关键帧的位姿
            if(i < WINDOW_SIZE)
            {
                LOG_ASSERT(all_frame_buffer[id]->getRefKeyFrame()->frame_id_==id);
                all_frame_buffer[id]->getRefKeyFrame()->setPose(all_frame_buffer[id]->pose());
            }

        }

        for(auto &info:Infos)
        {
            if(info.ResID== nullptr)continue;
            if(info.isbad== true)continue;
            MapPoint::Ptr mpt = info.mpt;
            Frame::Ptr frame = info.frame;
            Feature::Ptr feature = info.feature;

//            double residual = utils::reprojectError(feature->fn_.head<2>(), frame->Tcw(), mpt->optimal_pose_);
            static const double TH_REPJ = std::sqrt(3.81) * Config::imagePixelUnSigma2();

            if(reprojectionError(problem, info.ResID).squaredNorm() > TH_REPJ * (1 << feature->level_))
            {
                info.isbad = true;
                problem.RemoveResidualBlock(info.ResID);
//            std::cout << " rm outlier: " << mpt->id_ << " " << item.first->id_ << " " << obs.size() << std::endl;
//                mpt->removeObservation(frame->getRefKeyFrame());
//                changed_keyframes.insert(frame->getRefKeyFrame());
//                frame->removeMapPoint(mpt);

                if(mpt->type() == MapPoint::BAD)
                {
                    //todo 需要重新考虑一下
//                    mpt->setRemove();
                    system_status.BadPoints.push_back(mpt);
                }
            }

//            mpt->setPose(mpt->optimal_pose_);
            mpt->setPose(mpt->pose_double);
        }

        ceres::Solve(options, &problem, &summary);
        cout << summary.FullReport() << endl;

        for(int i=0;i<WINDOW_SIZE+1;i++)
        {
            int id = frame_id_window[i];
            all_frame_buffer[id]->updatePose();

            //!注意更新关键帧的位姿
            if(i<WINDOW_SIZE)
            {
                LOG_ASSERT(all_frame_buffer[id]->getRefKeyFrame()->frame_id_==id);
                all_frame_buffer[id]->getRefKeyFrame()->setPose(all_frame_buffer[id]->pose());
            }

        }
        for(auto &info:Infos)
        {
            if(info.ResID == nullptr)continue;
            MapPoint::Ptr mpt = info.mpt;
            Frame::Ptr frame = info.frame;
            if(info.isbad)
            {
                mpt->removeObservation(frame->getRefKeyFrame());
                changed_keyframes.insert(frame->getRefKeyFrame());
                frame->removeMapPoint(mpt);
                continue; //! 本次mappoint的观测失败，不用来更新mappoint位姿
            }
            mpt->setPose(mpt->pose_double);
        }

        for(const KeyFrame::Ptr &kf : changed_keyframes)
        {
            kf->updateConnections();
        }

        waitKey(0);

        //! 更新位姿


        //todo pvr bgba -> pose,ba,bg
//        for(int i=0;i<WINDOW_SIZE+1;i++)
//        {
//            int id = frame_id_window[i];
//            all_frame_buffer[id]->updatePoseAndBias();
//        }

        //! update mpts & remove mappoint with large error
/*
        int badnum = 0;
        for(auto &info : Infos)
        {
            if(info.ResID == nullptr)continue;
//            cout<<"----------------------------------set bad point ---------------------------------------"<<endl;
            MapPoint::Ptr mpt = info.mpt;
            Frame::Ptr frame = info.frame;
            Feature::Ptr feature = info.feature;

            double residual = utils::reprojectError(feature->fn_.head<2>(), frame->Tcw(), mpt->optimal_pose_);
            if(residual < max_residual)
            {
//                info.mpt->setPose(info.mpt->pose_double);
//                info.mpt->setPose(info.mpt->optimal_pose_);
                continue;
            }
            else
            {
                problem.RemoveResidualBlock(info.ResID);
                badnum ++;
                cout<<"----------------------------------set bad point --------------------------"<<mpt->id_<<" : "<<residual*460*460<<endl;
                mpt->removeObservation(frame->getRefKeyFrame());

                frame->removeMapPoint(mpt);
//                mpt->setBad();
//                system_status.BadPoints.push_back(mpt);

            }

        }
        cout<<"badnum: "<<badnum<<endl;*/
//        waitKey(0);

//        ceres::Solve(options, &problem, &summary);
//        cout << summary.FullReport() << endl;

/*
        for(int i=0;i<WINDOW_SIZE+1;i++)
        {
            int id = frame_id_window[i];
            all_frame_buffer[id]->updatePose();

            //!注意更新关键帧的位姿
            if(i<WINDOW_SIZE)
            {
                LOG_ASSERT(all_frame_buffer[id]->getRefKeyFrame()->frame_id_==id);
                all_frame_buffer[id]->getRefKeyFrame()->setPose(all_frame_buffer[id]->pose());
            }

        }

        for(auto &info : Infos)
        {
            if(info.ResID == nullptr) continue;
            MapPoint::Ptr mpt = info.mpt;
//            info.mpt->setPose(info.mpt->pose_double);
            info.mpt->setPose(info.mpt->optimal_pose_);
        }
        */



//        waitKey(0);

//        for(const KeyFrame::Ptr &kf : changed_keyframes)
//        {
//            kf->updateConnections();
//        }
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        //todo 封装先验误差
/*
        if(system_status.slideWindowFlag == System::Slide_old)
        {
            MarginalizationInfo *marginalization_info = new MarginalizationInfo();//新建一个要边缘化的信息,就是封装先验存储的地方

            if (last_marginalization_info)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    //参数块里没有特征点的深度，所以不需要加进去
                    if (last_marginalization_parameter_blocks[i] == all_frame_buffer[frame_id_window[0]]->PVR)
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);
                marginalization_info->addResidualBlockInfo(residual_block_info);//todo 先验信息的残差，暂时先放下
            }


            vector<Info> infos_marg;
            for(auto &info:Infos)
            {
                if(info.mpt->getReferenceKeyFrame()->frame_id_ != frame_id_window[0] || mpt_obsTimes[info.mpt->id_]<2) continue;

                MapPoint::Ptr mpt = info.mpt;
                Frame::Ptr frame = info.frame;
                Feature::Ptr feature = info.feature;
                vector<int> drop_set;

                if(frame->id_ == frame_id_window[0]) drop_set = {0,1};
                    else drop_set = {1};


                ceres::CostFunction* cost_function = ceres_slover::ReprojectionErrorSE3::Create(feature->fn_[0]/feature->fn_[2], feature->fn_[1]/feature->fn_[2]);//, 1.0/(1<<ft->level_));
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(cost_function,lossfunction,
                                                                               vector<double *>{frame->PVR, mpt->pose_double},
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);//todo 先验信息的残差，暂时先放下
            }

            //@brief 计算每一块残差块的雅克比矩阵和残差，并将参数块的信数值添加进去，用于之后的残差的更新
            marginalization_info->preMarginalize();

            marginalization_info->marginalize();

            vector<double *> parameter_blocks;
            marginalization_info->saveKeep();

            for (int i = 1; i <= WINDOW_SIZE; i++)
            {
                double *addr = all_frame_buffer[frame_id_window[i]]->PVR;
                parameter_blocks.push_back(addr);
            }

            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;//这个参数块是指保留下来的参数块的指针，因为在slidewindow中，进行了swap，所以地址和内容是对应的

        }
        else
        {
            if (last_marginalization_info &&
                std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), all_frame_buffer[frame_id_window[WINDOW_SIZE - 1]]->PVR))
            {
                MarginalizationInfo *marginalization_info = new MarginalizationInfo();
                if (last_marginalization_info)
                {
                    vector<int> drop_set;
                    for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                    {
                        if (last_marginalization_parameter_blocks[i] == all_frame_buffer[frame_id_window[WINDOW_SIZE - 1]]->PVR)
                            drop_set.push_back(i);
                    }
                    // construct new marginlization_factor
                    MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                                   last_marginalization_parameter_blocks,
                                                                                   drop_set);

                    marginalization_info->addResidualBlockInfo(residual_block_info);
                }


                marginalization_info->preMarginalize();

                marginalization_info->marginalize();

//                vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
                vector<double *> parameter_blocks;
                marginalization_info->saveKeep();
                for (int i = 0; i <= WINDOW_SIZE; i++)
                {
                    double *addr = all_frame_buffer[frame_id_window[i]]->PVR;
                    parameter_blocks.push_back(addr);
                }
                if (last_marginalization_info)
                    delete last_marginalization_info;
                last_marginalization_info = marginalization_info;
                last_marginalization_parameter_blocks = parameter_blocks;

            }
        }
*/


    }


}