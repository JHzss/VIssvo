#include "system.hpp"
#include "measurement.h"
#include "new_parameters.h"
#include <queue>
#include <mutex>

using namespace ssvo;
using namespace std;

std::condition_variable con;
std::queue<sensor_msgs::ImageConstPtr> ros_image_buf;
std::queue<sensor_msgs::ImuPtr> ros_imu_buf;
std::mutex ros_image_buf_lock;
std::mutex ros_imu_buf_lock;
vector<pair<vector<sensor_msgs::ImuPtr>,sensor_msgs::ImageConstPtr>> getMeasurement(std::queue<sensor_msgs::ImageConstPtr> &image_buf,std::queue<sensor_msgs::ImuPtr> &imu_buf);


void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    ros_image_buf_lock.lock();
    ros_image_buf.push(img_msg);
//    std::cout<<ros_image_buf.size()<<std::endl;
    ros_image_buf_lock.unlock();
}
void imu_callback(const sensor_msgs::ImuPtr &imu_msg)
{
    ros_imu_buf_lock.lock();
    ros_imu_buf.push(imu_msg);
    ros_imu_buf_lock.unlock();
    con.notify_one();
}
void process(System &vo_)
{
    while(true)
    {
        vector<pair<vector<sensor_msgs::ImuPtr>,sensor_msgs::ImageConstPtr>> measurements;
        pair<vector<sensor_msgs::ImuPtr>,sensor_msgs::ImageConstPtr> measurement;

        unique_lock<mutex> lk(ros_imu_buf_lock);
        con.wait(lk,[&]
        {
            return (measurements=getMeasurement(ros_image_buf,ros_imu_buf)).size()!=0;// 等价于 while(!measurement=getMeasurement()).first.size()!=0) { cv.wait(lk); }
        });
        lk.unlock();
        cout<<"measurement: "<<measurements.size()<<endl;
        //measure:<imus,image>
        fstream imustxt("/home/jh/imus.txt", ios::out);
        for(auto &measure:measurements)
        {
            for(auto imu_:measure.first)
            {
                imustxt<<imu_->header.stamp<<" "<<imu_->linear_acceleration.x<<" "<<imu_->linear_acceleration.y<<" "<<imu_->linear_acceleration.z<<endl;
            }
            imustxt<<endl;
            vo_.process(measure);
        }

//        if(vo_.get_vio_init())
//        {
//            vo_.saveTrajectoryTUM("/home/jh/trajectory_vio.txt");
//        }

    }



}

int main(int argc, char *argv[])
{
    ros::init(argc,argv,"ssvo");
    ros::NodeHandle n("~");

    std::string config_file;
    n.getParam("config_file",config_file);//!只是将launch文件中的config_file value传导过来

    std::cout<<"config_file: "<<config_file<<std::endl;

    google::InitGoogleLogging(argv[0]);

    System vo(config_file);
    LoadParameters(n);

    std::cout<<"img_topic:"<<vo.IMAGE_TOPIC_s<<std::endl;
    std::cout<<"imu_topic:"<<vo.IMU_TOPIC_s<<std::endl;

    ros::Subscriber sub_image = n.subscribe(vo.IMAGE_TOPIC_s,1000,img_callback);
    ros::Subscriber sub_imu = n.subscribe(vo.IMU_TOPIC_s,1000,imu_callback);

    std::thread main_thread(process,std::ref(vo));//用std::ref来包装引用类型的参数

    ros::spin();

    vo.saveTrajectoryTUM("trajectory.txt");
    cv::waitKey(0);

    return 0;
}