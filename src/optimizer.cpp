#include <iomanip>
#include <include/optimizer.hpp>
#include "optimizer.hpp"
#include "config.hpp"
#include "utils.hpp"
#include <map>

namespace ssvo{


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

        Matrix3d ttttt = Sophus::SO3d::exp(w).matrix();

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
/*
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

*/
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

    void Optimizer::slideWindowJointOptimization(vector<Frame::Ptr> &all_frame_buffer, uint64_t *frame_id_window)
    {

        cout<<"-------------------------------------slideWindowJointOptimization--------------------------------------"<<endl;
        std::unordered_set<MapPoint::Ptr> local_mappoints;
        vector<uint64_t > mappoints_ids;
        vector<pair<uint64_t ,vector<uint64_t >>> mappointID_frameIDS;
        vector<pair<uint64_t ,vector<Vector3d >>> mappointfn_frameIDS;
        ceres::Problem problem;
        std::set<KeyFrame::Ptr> actived_keyframes;
        std::set<KeyFrame::Ptr> fixed_keyframe;

        for(int i = 0; i < WINDOW_SIZE; i++)
        {
            int id = frame_id_window[i];
            actived_keyframes.insert(all_frame_buffer[id]->getRefKeyFrame());
        }

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
                if(fixed_keyframe.count(item.first))
                    continue;
                fixed_keyframe.insert(item.first);
            }
        }


        ceres::LocalParameterization* pvrpose = new PosePVR();
        for(const KeyFrame::Ptr &kf : fixed_keyframe)
        {
            problem.AddParameterBlock(kf->PVR, 9, pvrpose);
            problem.SetParameterBlockConstant(kf->PVR);
        }

        for(const KeyFrame::Ptr &kf : actived_keyframes)
        {
            problem.AddParameterBlock(kf->PVR, 9, pvrpose);
            if(kf->id_ <= 1)
                problem.SetParameterBlockConstant(kf->PVR);
        }


        //todo 理解不同的鲁棒核函数的区别
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
                problem.AddResidualBlock(cost_function1, lossfunction, kf->PVR, mpt->optimal_pose_.data());
            }
        }

        std::vector<Feature::Ptr> fts;
        Frame::Ptr frame = all_frame_buffer[WINDOW_SIZE];
        frame->getFeatures(fts);
        const size_t N = fts.size();

        for(size_t i = 0; i < N; ++i)
        {
            Feature::Ptr ft = fts[i];
            MapPoint::Ptr mpt = ft->mpt_;
            if(mpt == nullptr)
                continue;
            mpt->optimal_pose_ = mpt->pose();
            ceres::CostFunction* cost_function = ceres_slover::ReprojectionErrorSE3::Create(ft->fn_[0]/ft->fn_[2], ft->fn_[1]/ft->fn_[2]);//, 1.0/(1<<ft->level_));
            problem.AddResidualBlock(cost_function, lossfunction, frame->PVR, mpt->optimal_pose_.data());
        }


        //! add pose parameter
//        for(int i = 0; i < WINDOW_SIZE+1; i++)
//        {
//            int id = (int)frame_id_window[i];
//            ceres::LocalParameterization *pvrpose = new PosePVR();
//            ceres::LocalParameterization *biaspose = new PoseBias();
//            problem.AddParameterBlock(all_frame_buffer[id]->PVR,9,pvrpose);
//            problem.AddParameterBlock(all_frame_buffer[i]->bgba,6,biaspose);
//        }

        //! add imu

//        for(int i = 0; i < WINDOW_SIZE; i++)
//        {
//            int id_i = frame_id_window[i];
//            int id_j = frame_id_window[i+1];
//
//            cout<<"imu residual id_i: "<<id_i<<endl;
//            cout<<"imu residual id_j: "<<id_j<<endl;
//            Frame::Ptr last_frame = all_frame_buffer[id_i];
//            Frame::Ptr frame = all_frame_buffer[id_j];
//            //! imu误差
//            ceres::CostFunction* imu_factor = new ceres_slover::IMUError(last_frame,frame);
//            problem.AddResidualBlock(imu_factor,NULL,last_frame->PVR,last_frame->bgba,frame->PVR,frame->bgba);
//
//            //! bias误差
//            ceres::CostFunction* bias_factor = new ceres_slover::BiasError(frame->preintegration);
//            problem.AddResidualBlock(bias_factor,NULL,last_frame->bgba,frame->bgba);
//        }

        //! add feature
        //TODO 需要加一个特征点在滑窗内的匹配,需要用特征点的描述子进行匹配

//        map<uint64_t ,int > mptID_times;
//
//        for(int i = 0; i < WINDOW_SIZE+1 ; i++)
//        {
//            int id = frame_id_window[i];
//            std::vector<Feature::Ptr> fts;
//            Frame::Ptr frame = all_frame_buffer[id];
//            frame->getFeatures(fts);
//            cout<<"frame id:------------------------------"<<frame->id_<<endl;
//            const size_t N = fts.size();
//            for(int j = 0; j < N ; j++)
//            {
//                Feature::Ptr &ft = fts[j];
//                MapPoint::Ptr &mpt = ft->mpt_;
//                ceres::CostFunction* cost_function = ceres_slover::ReprojectionErrorSE3::Create(ft->fn_[0]/ft->fn_[2], ft->fn_[1]/ft->fn_[2]);//, 1.0/(1<<ft->level_));
//                problem.AddResidualBlock(cost_function, lossfunction, frame->optimal_Tcw_.data(), mpt->optimal_pose_.data());
//
//            }
//
//        }

        /*
        map<uint64_t ,int > mptID_times;
        for(int i = 0; i< WINDOW_SIZE+1; i++ )
        {
            int id = frame_id_window[i];
            std::vector<Feature::Ptr> fts;
            Frame::Ptr frame = all_frame_buffer[id];
            frame->getFeatures(fts);
            const size_t N = fts.size();
//        std::vector<ceres::ResidualBlockId> res_ids(N);
            for(size_t j = 0; j < N; ++j)
            {
                Feature::Ptr ft = fts[j];
                MapPoint::Ptr mpt = ft->mpt_;

                if(mpt == nullptr)
                    continue;

                vector<uint64_t >::iterator it = find(mappoints_ids.begin(), mappoints_ids.end(), mpt->id_);

                if(it==mappoints_ids.end())
                {
                    mptID_times[mpt->id_]=0;
                    vector<uint64_t> frame_ids;
                    frame_ids.push_back(frame->id_);
                    mappointID_frameIDS.push_back(make_pair(mpt->id_,frame_ids));

                    vector<Vector3d> frame_fns;
                    frame_fns.push_back(ft->fn_);
                    mappointfn_frameIDS.push_back(make_pair(mpt->id_,frame_fns));


                    local_mappoints.push_back(mpt);
                    mappoints_ids.push_back(mpt->id_);

                    mptID_times[mpt->id_]++;
                }
                else
                {
                    for(int i=0;i<mappointID_frameIDS.size();i++)
                    {
                        if(mappointID_frameIDS[i].first==mpt->id_)
                        {
                            mptID_times[mpt->id_]++;
                            mappointID_frameIDS[i].second.push_back(frame->id_);
                            mappointfn_frameIDS[i].second.push_back(ft->fn_);
                            break;
                        }
                    }
                }
            }

        }
        cout<<" mappoint size: "<<local_mappoints.size()<<endl;



        for(int i=0;i<local_mappoints.size();i++)
        {

            MapPoint::Ptr mpt = local_mappoints[i];
            if(mptID_times[mpt->id_]<2||mpt->getReferenceKeyFrame()->frame_id_>frame_id_window[WINDOW_SIZE-2])continue;

//            cout<<"被观测次数： "<<mptID_times[mpt->id_]<<endl;

            mpt->optimal_pose_ = mpt->pose();
//            cout<<"mappoint id: "<<mpt->id_<<endl;
            for(int j = 0;j<mappointID_frameIDS[i].second.size();j++)
            {
                Vector3d fn = mappointfn_frameIDS[i].second[j];
                ceres::CostFunction* cost_function1 = ceres_slover::ReprojectionErrorSE3::Create(fn[0]/fn[2], fn[1]/fn[2]);//, 1.0/(1<<ft->level_));
                problem.AddResidualBlock(cost_function1, lossfunction, all_frame_buffer[mappointID_frameIDS[i].second[j]]->PVR, mpt->optimal_pose_.data());
            }
        }


        */
        //todo 先验误差、闭环误差

        ceres::Solver::Options options;
        ceres::Solver::Summary summary;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.max_linear_solver_iterations = 20;

        ceres::Solve(options, &problem, &summary);


        double sum=0;
        int l=0;
        std::vector<ceres::ResidualBlockId> ids;
        problem.GetResidualBlocks(&ids);
        for (auto & id: ids)
        {
//            cout<<l<<":   ";
            sum += (reprojectionError(problem, id).transpose() * reprojectionError(problem, id));
//            cout<< reprojectionError(problem, id).transpose() * reprojectionError(problem, id)<<endl;
            l++;
        }

        cout<<"sum "<<l<<" "<<sum<<endl;
//        cout<<"视觉误差量个数："<<N<<endl;
        cout << summary.FullReport() << endl;
//        waitKey(0);


//        waitKey(0);

//        for(int i=0;i<WINDOW_SIZE+1;i++)
//        {
//            int id = frame_id_window[i];
//            all_frame_buffer[id]->updatePose();
//        }
        for(const KeyFrame::Ptr &kf : actived_keyframes)
        {
            kf->updatePose();
        }


        //todo pvr bgba -> pose,ba,bg
//        for(int i=0;i<WINDOW_SIZE+1;i++)
//        {
//            int id = frame_id_window[i];
//            all_frame_buffer[id]->updatePoseAndBias();
//        }

        //! update mpts & remove mappoint with large error
//        std::set<KeyFrame::Ptr> changed_keyframes;
//        static const double max_residual = Config::imagePixelUnSigma2() * std::sqrt(3.81);
        for(const MapPoint::Ptr &mpt : local_mappoints)
        {
//            const std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();
//            for(const auto &item : obs)
//            {
//                double residual = utils::reprojectError(item.second->fn_.head<2>(), item.first->Tcw(), mpt->optimal_pose_);
//                if(residual < max_residual)
//                    continue;
//
//                mpt->removeObservation(item.first);
//                changed_keyframes.insert(item.first);
////            std::cout << " rm outlier: " << mpt->id_ << " " << item.first->id_ << " " << obs.size() << std::endl;
//
////            if(mpt->type() == MapPoint::BAD)
////            {
////                bad_mpts.push_back(mpt);
////            }
//            }

            mpt->setPose(mpt->optimal_pose_);
        }

//        for(const KeyFrame::Ptr &kf : changed_keyframes)
//        {
//            kf->updateConnections();
//        }

        //todo 封装先验误差


    }


}