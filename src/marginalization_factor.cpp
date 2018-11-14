#include "marginalization_factor.h"

/**
 * @brief 计算每一块信息的残差和雅可比
 */

namespace ssvo{

void ResidualBlockInfo::Evaluate()
{
    residuals.resize(cost_function->num_residuals());//确定残差的维度

    std::vector<int> block_sizes = cost_function->parameter_block_sizes();//参数块的大小
    raw_jacobians = new double *[block_sizes.size()];//这个是double类型的数据，雅可比的个数，有几个参数块就有几个雅可比矩阵，因为是残差对参数块求偏导数的原因
    jacobians.resize(block_sizes.size());//vector型变量

    for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
    {
        jacobians[i].resize(cost_function->num_residuals(), block_sizes[i]);//对vector型变量中每一个雅克比矩阵的维度进行确定
        raw_jacobians[i] = jacobians[i].data();//矩阵的指针，这个技巧需要研究一下，从matrix类型转换到double类型,一般之前用到的都是double类型的转到matrix类型的
        //dim += block_sizes[i] == 7 ? 6 : block_sizes[i];
    }
    //残差和雅可比的计算方法之前已经写过了，就是那个虚函数，这样就把残差和雅可比计算出来了
    cost_function->Evaluate(parameter_blocks.data(), residuals.data(), raw_jacobians);//计算这个参数块的残差，雅可比,之后会用到
/*
    //std::vector<int> tmp_idx(block_sizes.size());
    //Eigen::MatrixXd tmp(dim, dim);
    //for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
    //{
    //    int size_i = localSize(block_sizes[i]);
    //    Eigen::MatrixXd jacobian_i = jacobians[i].leftCols(size_i);
    //    for (int j = 0, sub_idx = 0; j < static_cast<int>(parameter_blocks.size()); sub_idx += block_sizes[j] == 7 ? 6 : block_sizes[j], j++)
    //    {
    //        int size_j = localSize(block_sizes[j]);
    //        Eigen::MatrixXd jacobian_j = jacobians[j].leftCols(size_j);
    //        tmp_idx[j] = sub_idx;
    //        tmp.block(tmp_idx[i], tmp_idx[j], size_i, size_j) = jacobian_i.transpose() * jacobian_j;
    //    }
    //}
    //Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(tmp);
    //std::cout << saes.eigenvalues() << std::endl;
    //ROS_ASSERT(saes.eigenvalues().minCoeff() >= -1e-6);
*/
    //TODO 这部分还没有研究
    if (loss_function)
    {
        double residual_scaling_, alpha_sq_norm_;

        double sq_norm, rho[3];

        sq_norm = residuals.squaredNorm();//平方范数
        loss_function->Evaluate(sq_norm, rho);//定义loss function的时候应该就是为了调用这里的evaluate
        //printf("sq_norm: %f, rho[0]: %f, rho[1]: %f, rho[2]: %f\n", sq_norm, rho[0], rho[1], rho[2]);

        double sqrt_rho1_ = sqrt(rho[1]);

        if ((sq_norm == 0.0) || (rho[2] <= 0.0))
        {
            residual_scaling_ = sqrt_rho1_;
            alpha_sq_norm_ = 0.0;
        }
        else
        {
            const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
            const double alpha = 1.0 - sqrt(D);
            residual_scaling_ = sqrt_rho1_ / (1 - alpha);
            alpha_sq_norm_ = alpha / sq_norm;
        }

        for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
        {
            jacobians[i] = sqrt_rho1_ * (jacobians[i] - alpha_sq_norm_ * residuals * (residuals.transpose() * jacobians[i]));
        }

        residuals *= residual_scaling_;
    }
}

MarginalizationInfo::~MarginalizationInfo()
{
    //ROS_WARN("release marginlizationinfo");
    
    for (auto it = parameter_block_data.begin(); it != parameter_block_data.end(); ++it)
        delete it->second;

    for (int i = 0; i < (int)factors.size(); i++)
    {

        delete[] factors[i]->raw_jacobians;
        
        delete factors[i]->cost_function;

        delete factors[i];
    }
}

/**
 * @brief 添加ResidualBlockInfo
 * @param residual_block_info
 * 包含了预积分的信息，残差块信息以及残差的类型
 */
void MarginalizationInfo::addResidualBlockInfo(ResidualBlockInfo *residual_block_info)
{
    //所有的残差块都先放到这里
    factors.emplace_back(residual_block_info);

    // 这个残差信息中所有的参数块的内容
    std::vector<double *> &parameter_blocks = residual_block_info->parameter_blocks;                            // 参数块数据,定义了一个引用
    // 每一个残差块的大小，跟parameter_blocks.size()大小相同
    //TODO 这个的大小是怎么来的
    std::vector<int> parameter_block_sizes = residual_block_info->cost_function->parameter_block_sizes();       // 记载参数块长度的数组,应该是{9,3}
    //定义每一个参数块的指针及其大小的map
    for (int i = 0; i < static_cast<int>(residual_block_info->parameter_blocks.size()); i++)
    {
        double *addr = parameter_blocks[i];//每一个参数块的指针
        int size = parameter_block_sizes[i];//每一个参数块的大小
        // reinterpret_cast:转换运算符，转换一个指针类型到一个足够大的整数类型
        parameter_block_size[reinterpret_cast<long>(addr)] = size;//<参数块指针，大小>
    }
    //定义需要移除的参数块的指针及其大小的map，其中大小被初始化成0，之后还会有赋值
    for (int i = 0; i < static_cast<int>(residual_block_info->drop_set.size()); i++)
    {
        double *addr = parameter_blocks[residual_block_info->drop_set[i]];
        parameter_block_idx[reinterpret_cast<long>(addr)] = 0;
    }
}
/**
 * @brief 将IMU、camera的参数块都添加到统一的内存（parameter_block_data）中
 * 包括imu残差和所有首次观测到特征点的残差
 */
void MarginalizationInfo::preMarginalize()
{
    for (auto it : factors)//遍历每一个残差信息
    {
        /**
         * @brief 计算每一块信息的残差和雅可比
         */
        it->Evaluate();//计算残差和雅可比

        std::vector<int> block_sizes = it->cost_function->parameter_block_sizes();
        for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
        {
            long addr = reinterpret_cast<long>(it->parameter_blocks[i]);
            int size = block_sizes[i];
            //这个技巧是防止重复添加，没有添加进去的参数块才会被添加进去
            if (parameter_block_data.find(addr) == parameter_block_data.end())
            {
                double *data = new double[size];

                // memcpy：内存拷贝函数，data：目的地;it->parameter_blocks[i]:数据来源；残差的信息统一放到这里

                //todo 这里貌似不可以那么写。。。
                memcpy(data, it->parameter_blocks[i], sizeof(double) * size);
                parameter_block_data[addr] = data;//把所有的参数块数据同意添加到parameter_block_data中
            }
        }
    }
}

int MarginalizationInfo::localSize(int size) const
{
    return size/* == 7 ? 6 : size*/;
}

int MarginalizationInfo::globalSize(int size) const
{
    return size/* == 6 ? 7 : size*/;
}

/**
 * @brief 构造稀疏矩阵H
 * @param threadsstruct
 * @return
 */

void* ThreadsConstructA(void* threadsstruct)
{
    //遍历这个线程中所有的残差因子
    ThreadsStruct* p = ((ThreadsStruct*)threadsstruct);
    for (auto it : p->sub_factors)
    {
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
        {
            int idx_i = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
            int size_i = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])];
            if (size_i == 7)
                size_i = 6;
            Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);

            // H= J^T J 分块计算，
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
            {
                int idx_j = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                int size_j = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])];
                if (size_j == 7)
                    size_j = 6;
                Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);

                // 对角线
                if (i == j)
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;//size_i行，size_j列
                else
                {
                    // H矩阵的右上角和左下角都是对称阵
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    p->A.block(idx_j, idx_i, size_j, size_i) = p->A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            p->b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
        }
    }
    return threadsstruct;
}

void MarginalizationInfo::marginalize()
{
    int pos = 0;//A b矩阵的维度


    // 将所有的参数块在在矩阵中的位置确定下来，其中前m个是要去除的参数块，后n个与之有约束关联的参数块，pose=m+n
    //设置这个变量的作用就是在添加的时候能分清楚先添加那些需要移除的变量...以便构造A b矩阵的时候将参数块分开
    for (auto &it : parameter_block_idx)
    {
        it.second = pos;//之前这个被设置成0，现在对其进行赋值
        // localSize：size == 7 ? 6 : size
        pos += localSize(parameter_block_size[it.first]);//todo size是7的话就+6，否则就是本身，为什么？？？
    }

    m = pos;

    for (const auto &it : parameter_block_size)
    {
        if (parameter_block_idx.find(it.first) == parameter_block_idx.end())//这个参数不在要优化的参数块里，再添加这个
        {
            parameter_block_idx[it.first] = pos;
            pos += localSize(it.second);
        }
    }

    n = pos - m;

    //todo 也就是说，位姿参数块 +6 ，速度参数块+9 特征参数快+1 为何？

    //ROS_DEBUG("marginalization, pos: %d, m: %d, n: %d, size: %d", pos, m, n, (int)parameter_block_idx.size());

//    TicToc t_summing;
    Eigen::MatrixXd A(pos, pos);
    Eigen::VectorXd b(pos);
    A.setZero();
    b.setZero();
    //作者注释掉的这一块应该就是直接构造的，但是消耗的时间太长了，就弄了四个线程来构造这个矩阵
    /*
    for (auto it : factors)
    {
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
        {
            int idx_i = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
            int size_i = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])]);
            Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
            {
                int idx_j = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                int size_j = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])]);
                Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
                if (i == j)
                    A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                else
                {
                    A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    A.block(idx_j, idx_i, size_j, size_i) = A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
        }
    }
    ROS_INFO("summing up costs %f ms", t_summing.toc());
    */
    //multi thread

    // NUM_THREADS 4
    // 利用多线程构造稀疏矩阵H
//    TicToc t_thread_summing;
    pthread_t tids[NUM_THREADS];
    ThreadsStruct threadsstruct[NUM_THREADS];
    int i = 0;

    //将factors中的残差分成四份分别添加到四个线程中，分别进行构造
    for (auto it : factors)
    {
        threadsstruct[i].sub_factors.push_back(it);
        i++;
        i = i % NUM_THREADS;
    }
    // 创建4个线程计算，把残差分成四个线程，然后再累加构造A b,测试发现A矩阵的维度一般都是200多。。。
    for (int i = 0; i < NUM_THREADS; i++)
    {
//        TicToc zero_matrix;
        threadsstruct[i].A = Eigen::MatrixXd::Zero(pos,pos);
        threadsstruct[i].b = Eigen::VectorXd::Zero(pos);
        threadsstruct[i].parameter_block_size = parameter_block_size;
        threadsstruct[i].parameter_block_idx = parameter_block_idx;

        // 构造多线程
        /*
         * 参数说明：
            tidp：新创建的线程ID会被设置成tidp指向的内存单元。
            attr：用于定制各种不能的线程属性，默认为NULL
            start_rtn：新创建的线程从start_rtn函数的地址开始运行，该函数只有一个void类型的指针参数即arg，
            如果start_rtn需要多个参数，可以将参数放入一个结构中，然后将结构的地址作为arg传入。
         */
        int ret = pthread_create( &tids[i], NULL, ThreadsConstructA ,(void*)&(threadsstruct[i]));
        if (ret != 0)
        {
            ROS_WARN("pthread_create error");
            ROS_BREAK();
        }
    }
    for( int i = NUM_THREADS - 1; i >= 0; i--)  
    {
        // pthread_join：等待一个线程的结束
        pthread_join( tids[i], NULL ); 
        A += threadsstruct[i].A;
        b += threadsstruct[i].b;
    }
//    std::cout<<"A:----------- "<<A.rows()<<std::endl;
    //ROS_DEBUG("thread summing up costs %f ms", t_thread_summing.toc());
    //ROS_INFO("A diff %f , b diff %f ", (A - tmp_A).sum(), (b - tmp_b).sum());


    //TODO
    Eigen::MatrixXd Amm = 0.5 * (A.block(0, 0, m, m) + A.block(0, 0, m, m).transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);

    //ROS_ASSERT_MSG(saes.eigenvalues().minCoeff() >= -1e-4, "min eigenvalue %f", saes.eigenvalues().minCoeff());

    Eigen::MatrixXd Amm_inv = saes.eigenvectors() * Eigen::VectorXd((saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() * saes.eigenvectors().transpose();
    //printf("error1: %f\n", (Amm * Amm_inv - Eigen::MatrixXd::Identity(m, m)).sum());


    // https://blog.csdn.net/heyijia0327/article/details/53707261 公式（2）（3）之间
    Eigen::VectorXd bmm = b.segment(0, m);
    Eigen::MatrixXd Amr = A.block(0, m, m, n);
    Eigen::MatrixXd Arm = A.block(m, 0, n, m);
    Eigen::MatrixXd Arr = A.block(m, m, n, n);
    Eigen::VectorXd brr = b.segment(m, n);


    //todo MatrixXd的小可可以再运行的时候改变？
    A = Arr - Arm * Amm_inv * Amr;//维度在60-70
    b = brr - Arm * Amm_inv * bmm;

    //特征分解
    //链接：https://baike.baidu.com/item/%E7%89%B9%E5%BE%81%E5%88%86%E8%A7%A3/12522621?fr=aladdin 分解方法一栏
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);
    // TODO 这里作者威慑呢么要写成这个形式不理解。。。
    // Eigen::VectorXd：类型转换；
    // 如果saes2.eigenvalues().array() > eps这个条件满足，则当前这个矩阵元素的值为saes2.eigenvalues().array()相对对应的值，否则为0；
    // 换言之，如果求解出来的特征值向量中有值小于无限小，则把这个元素置零，否则仍为这个数
    Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
    Eigen::VectorXd S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));

    // 每一个矩阵元素开平方
    Eigen::VectorXd S_sqrt = S.cwiseSqrt();
    Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

    // 雅克比矩阵
    // asDiagonal将S_sqrt这个矢量作为矩阵对角线，其他值为0
    linearized_jacobians = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();

    // 残差
    linearized_residuals = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;
    //std::cout << A << std::endl
    //          << std::endl;
    //std::cout << linearized_jacobians << std::endl;
    //printf("error2: %f %f\n", (linearized_jacobians.transpose() * linearized_jacobians - A).sum(),
    //      (linearized_jacobians.transpose() * linearized_residuals - b).sum());
}

std::vector<double *> MarginalizationInfo::getParameterBlocks(std::unordered_map<long, double *> &addr_shift)
{
    std::vector<double *> keep_block_addr;//存放要保留的参数块数组的指针的容器
    keep_block_size.clear();
    keep_block_idx.clear();
    keep_block_data.clear();

    for (const auto &it : parameter_block_idx)
    {
        if (it.second >= m)
        {
            keep_block_size.push_back(parameter_block_size[it.first]);
            keep_block_idx.push_back(parameter_block_idx[it.first]);
            keep_block_data.push_back(parameter_block_data[it.first]);//这个是保存上一次参数块指针中的值，用于参数块计算dx，然后计算残差
            keep_block_addr.push_back(addr_shift[it.first]);
        }
    }
    sum_block_size = std::accumulate(std::begin(keep_block_size), std::end(keep_block_size), 0);

    return keep_block_addr;
}

    void MarginalizationInfo::saveKeep()
    {
        keep_block_size.clear();
        keep_block_idx.clear();
        keep_block_data.clear();

        for (const auto &it : parameter_block_idx)
        {
            if (it.second >= m)
            {
                keep_block_size.push_back(parameter_block_size[it.first]);
                keep_block_idx.push_back(parameter_block_idx[it.first]);
                keep_block_data.push_back(parameter_block_data[it.first]);//这个是保存上一次参数块指针中的值，用于参数块计算dx，然后计算残差
//                keep_block_addr.push_back(it.first);
            }
        }
        sum_block_size = std::accumulate(std::begin(keep_block_size), std::end(keep_block_size), 0); //accumulate带有三个形参：头两个形参指定要累加的元素范围，第三个形参则是累加的初值。
    }


/**
 * @brief 配置ceres mutable_parameter_block_sizes和set_num_residuals
 * @param _marginalization_info
 */
MarginalizationFactor::MarginalizationFactor(MarginalizationInfo* _marginalization_info):marginalization_info(_marginalization_info)
{
    int cnt = 0;
    for (auto it : marginalization_info->keep_block_size)
    {
        mutable_parameter_block_sizes()->push_back(it);
        cnt += it;
    }
    //printf("residual size: %d, %d\n", cnt, n);
    set_num_residuals(marginalization_info->n);
};


/**
 * @brief 计算残差向量和雅克比矩阵
 * @param parameters 参数块队列，与CostFunction::parameter_block_sizes_参数个数是一样的
 * @param residuals 残差队列，与num_residuals_队列的长度是一样的
 * @param jacobians
 * @return
 */
bool MarginalizationFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    //printf("internal addr,%d, %d\n", (int)parameter_block_sizes().size(), num_residuals());
    //for (int i = 0; i < static_cast<int>(keep_block_size.size()); i++)
    //{
    //    //printf("unsigned %x\n", reinterpret_cast<unsigned long>(parameters[i]));
    //    //printf("signed %x\n", reinterpret_cast<long>(parameters[i]));
    //printf("jacobian %x\n", reinterpret_cast<long>(jacobians));
    //printf("residual %x\n", reinterpret_cast<long>(residuals));
    //}
    int n = marginalization_info->n;
    int m = marginalization_info->m;
    Eigen::VectorXd dx(n);

    for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
    {
        int size = marginalization_info->keep_block_size[i];
        int idx = marginalization_info->keep_block_idx[i] - m;
        Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size);
        Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(marginalization_info->keep_block_data[i], size);

        dx.segment(idx, size) = x - x0;
    }
    Eigen::Map<Eigen::VectorXd>(residuals, n) = marginalization_info->linearized_residuals + marginalization_info->linearized_jacobians * dx;//高斯牛顿迭代求解时会用到。
    if (jacobians)
    {

        for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
        {
            if (jacobians[i])
            {
                int size = marginalization_info->keep_block_size[i], local_size = marginalization_info->localSize(size);
                int idx = marginalization_info->keep_block_idx[i] - m;
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(jacobians[i], n, size);
                jacobian.setZero();
                jacobian.leftCols(local_size) = marginalization_info->linearized_jacobians.middleCols(idx, local_size);//FEJ?雅可比不变？起始位置，列数
            }
        }
    }
    return true;
}

}