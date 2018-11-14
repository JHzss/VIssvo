#pragma once
#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>
#include "global.hpp"


namespace ssvo{

const int NUM_THREADS = 4;

struct ResidualBlockInfo
{
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    void Evaluate();

    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    std::vector<double *> parameter_blocks;//这个残差中参数块数组，例如特征残差中，4 4 4 1，分别是残差块数组的大小
    std::vector<int> drop_set;//需要去掉的参数块的id，0 1，就是第0 1 个参数块，0 3 就是第0 3 个参数块

    double **raw_jacobians;//
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    Eigen::VectorXd residuals;

    int localSize(int size)
    {
        return size/* == 7 ? 6 : size*/;
    }
};

struct ThreadsStruct
{
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; //global size
    std::unordered_map<long, int> parameter_block_idx; //local size
};

//搞清楚这些是什么意思
class MarginalizationInfo
{
  public:
    ~MarginalizationInfo();
    int localSize(int size) const;
    int globalSize(int size) const;
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);
    void preMarginalize();
    void marginalize();
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);


    void saveKeep();

    std::vector<ResidualBlockInfo *> factors;//残差块信息
    int m, n;//m比n大，感觉不太科学啊
    std::unordered_map<long, int> parameter_block_size; //global size  <所有参数块地址，参数块大小>，所有的参数
    int sum_block_size;
    std::unordered_map<long, int> parameter_block_idx; //local size  <所有参数块的地址，在矩阵中起始的位置> 设置这个变量的作用就是在添加的时候能分清楚先添加那些变量...以便构造A b矩阵
    std::unordered_map<long, double *> parameter_block_data;

    //! 保存量
    std::vector<int> keep_block_size; //global size
    std::vector<int> keep_block_idx;  //local size
    std::vector<double *> keep_block_data;

    Eigen::MatrixXd linearized_jacobians;  // 最终的雅克比矩阵
    Eigen::VectorXd linearized_residuals;  // 最终的残差量
    const double eps = 1e-8;

};

class MarginalizationFactor : public ceres::CostFunction
{
  public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo* marginalization_info;
};

}