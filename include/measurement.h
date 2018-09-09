//
// Created by jh on 18-6-4.
//

#ifndef SSVO_MEASUREMENT_H
#define SSVO_MEASUREMENT_H

#include "global.hpp"
using namespace std;

int image=0;
sensor_msgs::ImageConstPtr first_image;
sensor_msgs::ImuPtr imu_begin,imu_end;
sensor_msgs::ImuPtr imu0,imu1,imu2,imu3,imu4,imu5,imu6,imu7;
ros::Time image_first_time,image_second_time;

double Hermite(uint64_t stamp_,uint64_t stamp10, double a0,uint64_t stamp11, double a1,uint64_t stamp12, double a2,uint64_t stamp13, double a3)
{
    if(stamp_==stamp11)
        return a1;
    if(stamp_==stamp12)
        return a2;
    double stamp,stamp0,stamp1,stamp2,stamp3;
    double k01,k12,k23,k1,k2;
    double result,result1,result2,result3,result4;

    stamp=((double)stamp_/1000000000.0);
    stamp0=((double)stamp10/1000000000.0);
    stamp1=((double)stamp11/1000000000.0);
    stamp2=((double)stamp12/1000000000.0);
    stamp3=((double)stamp13/1000000000.0);
    k01=(a1-a0)/(stamp1-stamp0);
    k12=(a2-a1)/(stamp2-stamp1);
    k23=(a3-a2)/(stamp3-stamp2);
    k1=(k01+k12)/2;
    k2=(k12+k23)/2;
    result1=(1-2*((stamp-stamp1)/(stamp1-stamp2)))*((stamp-stamp2)/(stamp1-stamp2))*((stamp-stamp2)/(stamp1-stamp2))*a1;
    result2=(1-2*((stamp-stamp2)/(stamp2-stamp1)))*((stamp-stamp1)/(stamp2-stamp1))*((stamp-stamp1)/(stamp2-stamp1))*a2;
    result3=(stamp-stamp1)*((stamp-stamp2)/(stamp1-stamp2))*((stamp-stamp2)/(stamp1-stamp2))*k1;
    result4=(stamp-stamp2)*((stamp-stamp1)/(stamp2-stamp1))*((stamp-stamp1)/(stamp2-stamp1))*k2;
    result=result1+result2+result3+result4;
    cout<<"hermite result:"<<result<<endl;
    return result;
}
sensor_msgs::ImuPtr HermiteImu(sensor_msgs::ImuPtr imu0,sensor_msgs::ImuPtr imu1,sensor_msgs::ImuPtr imu2,sensor_msgs::ImuPtr imu3,ros::Time image_time)
{
    sensor_msgs::ImuPtr imu(new sensor_msgs::Imu(*imu0));
    uint64_t stamp;

    double ax,ay,az;
    double gx,gy,gz;
    stamp=image_time.toNSec();
    gx=Hermite(stamp,imu0->header.stamp.toNSec(),imu0->angular_velocity.x,imu1->header.stamp.toNSec(),imu1->angular_velocity.x,imu2->header.stamp.toNSec(),imu2->angular_velocity.x,imu3->header.stamp.toNSec(),imu3->angular_velocity.x);
    gy=Hermite(stamp,imu0->header.stamp.toNSec(),imu0->angular_velocity.y,imu1->header.stamp.toNSec(),imu1->angular_velocity.y,imu2->header.stamp.toNSec(),imu2->angular_velocity.y,imu3->header.stamp.toNSec(),imu3->angular_velocity.y);
    gz=Hermite(stamp,imu0->header.stamp.toNSec(),imu0->angular_velocity.z,imu1->header.stamp.toNSec(),imu1->angular_velocity.z,imu2->header.stamp.toNSec(),imu2->angular_velocity.z,imu3->header.stamp.toNSec(),imu3->angular_velocity.z);

    ax=Hermite(stamp,imu0->header.stamp.toNSec(),imu0->linear_acceleration.x,imu1->header.stamp.toNSec(),imu1->linear_acceleration.x,imu2->header.stamp.toNSec(),imu2->linear_acceleration.x,imu3->header.stamp.toNSec(),imu3->linear_acceleration.x);
    ay=Hermite(stamp,imu0->header.stamp.toNSec(),imu0->linear_acceleration.y,imu1->header.stamp.toNSec(),imu1->linear_acceleration.y,imu2->header.stamp.toNSec(),imu2->linear_acceleration.y,imu3->header.stamp.toNSec(),imu3->linear_acceleration.y);
    az=Hermite(stamp,imu0->header.stamp.toNSec(),imu0->linear_acceleration.z,imu1->header.stamp.toNSec(),imu1->linear_acceleration.z,imu2->header.stamp.toNSec(),imu2->linear_acceleration.z,imu3->header.stamp.toNSec(),imu3->linear_acceleration.z);

    imu->header.stamp=image_time;
    imu->angular_velocity.x=gx;
    imu->angular_velocity.y=gy;
    imu->angular_velocity.z=gz;
    imu->linear_acceleration.x=ax;
    imu->linear_acceleration.y=ay;
    imu->linear_acceleration.z=az;
//    cout<<"imu "<<"stamp: "<<imu->header.stamp<<endl<<"acc:"<<endl<<imu->linear_acceleration<<endl<<"gyr:"<<endl<<imu->angular_velocity<<endl;
    return imu;
}

vector<pair<vector<sensor_msgs::ImuPtr>,sensor_msgs::ImageConstPtr>> getMeasurement(std::queue<sensor_msgs::ImageConstPtr> &image_buf,std::queue<sensor_msgs::ImuPtr> &imu_buf)
{
    cout<<image_buf.size()<<"     "<<imu_buf.size()<<endl;
    //注意这里为什么把measurements定义在while循环的外面，因为getMeasurement()一直循环进行，而且要得到measurements，定义在里面会导致清空
    vector<pair<vector<sensor_msgs::ImuPtr>,sensor_msgs::ImageConstPtr>> measurements;
    while (true)
    {
        pair<vector<sensor_msgs::ImuPtr>,sensor_msgs::ImageConstPtr> measurement;
        vector<sensor_msgs::ImuPtr> imus;
        sensor_msgs::ImageConstPtr img;
        //image信息或IMU信息太少,设置成50保证image对应足够的imu
        imus.clear();
        if((image_buf.size()<3)||imu_buf.size()<20)
        {
            return measurements;
        }
        //刚开始的时候
        if(first_image==nullptr)
        {
            //IMU信息比Image信息来的晚
            if(imu_buf.front()->header.stamp>image_buf.front()->header.stamp)
            {
                ROS_INFO("IMU is later than image,wait for imu and delete image");
                image_buf.pop();
                continue;
            }
            if(imu_buf.front()->header.stamp<image_buf.front()->header.stamp)
            {
                img=image_buf.front();
                image_first_time=img->header.stamp;
                first_image=image_buf.front();
                image_buf.pop();
                image++;//判断这是第几帧（系统的首帧作为第0帧）
            }
        }
        //表示进入连续跟踪模式
        if(image>0)
        {
            //只保留第一帧图像最近两个的imu进行差值
            if(image==1)
            {
                while (imu_buf.front()->header.stamp<=image_first_time)
                {
                    if(imu1!= nullptr)
                        imu0=imu1;//todo 需要判定图像帧之前是否存在两帧以上的imu信息
                    imu1=imu_buf.front();
                    imu_buf.pop();
                }
                if(imu0== nullptr)
                {
                    imu0=imu1;//todo 需要判定图像帧之前是否存在两帧以上的imu信息
                }
                imu_begin=imu1;
                imus.emplace_back(imu_begin);
            }
            ///添加 imus
            if(image!=1)
            {
                image_first_time=image_second_time;//在连续运行的时候保证first time是上一次的second time
                imu_begin=imu_end;//todo 赋值有问题
                imus.emplace_back(imu_begin);
                imus.emplace_back(imu6);
            }
            image_second_time=image_buf.front()->header.stamp;
            while(imu_buf.front()->header.stamp<=image_second_time)
            {
                //只有在处理第一帧的时候才需要0 1 2 3
                if(image==1)
                {
                    if(imus.size()==1)
                        imu3=imu_buf.front();
                    if(imus.size()==2)
                    {
                        imu2=imu3;
                        imu3=imu_buf.front();
                    }
                }
                imus.emplace_back(imu_buf.front());
                if(imu5!= nullptr)
                    imu4=imu5;
                imu5=imu_buf.front();
                imu_buf.pop();
            }


            if(imu4== nullptr) //保证imu4不为空
            {
                ROS_INFO("set imu4 , maybe wrong");
                imu4=imu5;
            }
            imu_end=imu5;  //对imu_end赋初值
            //todo imu和图像之间额顺序混的话需要给出错误提示
            //添加imus结束
            imu6=imu_buf.front();
            imu_buf.pop();
            imu7=imu_buf.front();
            if(image==1)
            {
                imu_begin=HermiteImu(imu0,imu1,imu2,imu3,image_first_time);
                imus.front()=imu_begin;
                imu_end=HermiteImu(imu4,imu5,imu6,imu7,image_second_time);
                imus.emplace_back(imu_end);
            }
            else
            {
                imu_end=HermiteImu(imu4,imu5,imu6,imu7,image_second_time);
                imus.emplace_back(imu_end);
            }
            measurement=(make_pair(imus,image_buf.front()));
            measurements.push_back(measurement);
            image_buf.pop();//这个必须要有
            /*至此，得到的信息应该是这样的：（其中，imu_buf在imu6之前的（包括6）都被pop了，imu_buf的front指针指向7）
             * 图像序列也被pop了，现在的front是接下来要处理的image
            *           0123                4567
            * ,IMU序列  ||||||||||||||||||||||||
            *           -----------------------
            *   图像序列  |                   |
            *     imu_begin,2,3,……,4,5,imu_end;
            */
            image++;
        }
    }
}

#endif //SSVO_MEASUREMENT_H
