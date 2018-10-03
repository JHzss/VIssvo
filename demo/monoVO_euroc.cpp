#include "system.hpp"
#include "dataset.hpp"
#include "time_tracing.hpp"
#include "new_parameters.h"

using namespace ssvo;

vector<pair<vector<ssvo::IMUData>,pair<Mat,double>>> getMeasurement_noros(vector<string> vstrImageFilenames, vector<double> vTimestamps , vector<ssvo::IMUData> vImus);
void LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps);
void LoadImus(const string &strImuPath, vector<ssvo::IMUData> &vImus);
void Loadconfig(const string &strconfigPath);

//int image=0;
//sensor_msgs::ImageConstPtr first_image;
//sensor_msgs::ImuPtr imu_begin,imu_end;
//sensor_msgs::ImuPtr imu0,imu1,imu2,imu3,imu4,imu5,imu6,imu7;
//ros::Time image_first_time,image_second_time;
#include <stdlib.h>
#include <signal.h>
volatile sig_atomic_t sigflag = 0;
void sigint_function(int sig)
{
    LOG(INFO)<<"SIGINT catch"<<endl;
    sigflag = 1;
}

int main(int argc, char *argv[])
{
    google::InitGoogleLogging(argv[0]);

    if (argc != 6) {
        cerr << endl << "Usage: ./monoVO_dataset  vocabulary  settings  image_folder  image_times_file  imu_path"
             << endl;
        cerr << endl << "For example: " << endl <<
             "./monoVO_dataset /home/jh/catkin_ws/src/LearnVIORB/Vocabulary /home/jh/catkin_ws/src/ssvo/config/euroc.yaml  /home/jh/MH_01_easy/cam0/data /home/jh/MH_01_easy/cam0/data.csv /home/jh/MH_01_easy/imu0/data.csv"
             << endl;
        return 1;
    }

    signal(SIGINT, sigint_function);

//    EuRocDataReader dataset(argv[2]);

    /**
     * @brief 读取IMU信息和图像信息
     * vImus               所有的imu数据
     * vstrImageFilenames  所有的图像的文件名
     * vTimestamps         所有图像的时间戳
     * nImus               imu数据大小
     * nImages             图像数据大小
     */
    vector<ssvo::IMUData> vImus;
    LoadImus(argv[5], vImus);
    int nImus = vImus.size();
    cout << "Imus in data: " << nImus << endl;
    if (nImus <= 0) {
        cerr << "ERROR: Failed to load imus" << endl;
        return 1;
    }

    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    LoadImages(string(argv[3]), string(argv[4]), vstrImageFilenames, vTimestamps);

    int nImages = vstrImageFilenames.size();
    cout << "Images in data: " << nImages << endl;

    if (nImages <= 0) {
        cerr << "ERROR: Failed to load images" << endl;
        return 1;
    }

/** @brief 构造pair用于追踪
 *
 */
    cout<<"Begin make vMeasurements..."<<endl;
    vector<pair<vector<ssvo::IMUData>,pair<Mat,double>>> vMeasurements = getMeasurement_noros(vstrImageFilenames,vTimestamps,vImus);
    cout<<"Make vMeasurements finished!"<<endl;

    System vo(argv[2]);
    Loadconfig(argv[2]);


    ssvo::Timer<std::micro> timer;

    int N = vMeasurements.size();

    for(int i = 0 ; i < vMeasurements.size() ; i++)
    {
        pair<vector<ssvo::IMUData>,pair<Mat,double>> measurement = vMeasurements[i];
        timer.start();
        vo.process(measurement);
        timer.stop();

        double time_process = timer.duration();

        double time_wait = 0;

        if(i < N -1)
            time_wait = (vMeasurements[i+1].second.second - measurement.second.second)*1e6;
        else
            time_wait = (measurement.second.second - vMeasurements[i-1].second.second)*1e6;

        if(time_process < time_wait)
        {
            std::this_thread::sleep_for(std::chrono::microseconds((int)(time_wait - time_process)));

            cout<<"wait ------------------------------>"<<(int)(time_wait - time_process)<<endl;
        }

        cout<<"vMeasurements.size(): "<<vMeasurements.size()<<" i: "<<i<<endl;
    }

    cout<<"begin save tra!~"<<endl;

    vo.saveTrajectoryTUM("/home/jh/trajectory_euroc.txt");
    getchar();

        return 0;
    }

    void LoadImus(const string &strImuPath, vector<ssvo::IMUData> &vImus)
    {
        ifstream fImus;
        fImus.open(strImuPath.c_str());
        vImus.reserve(300000);
        //int testcnt = 10;
        while (!fImus.eof()) {
            string s;
            getline(fImus, s);
            if (!s.empty()) {
                char c = s.at(0);
                if (c < '0' || c > '9')
                    continue;

                stringstream ss;
                ss << s;
                double tmpd;
                int cnt = 0;
                double data[10];    // timestamp, wx,wy,wz, ax,ay,az
                while (ss >> tmpd) {
                    data[cnt] = tmpd;
                    cnt++;
                    if (cnt == 7)
                        break;
                    if (ss.peek() == ',' || ss.peek() == ' ')
                        ss.ignore();
                }
                data[0] *= 1e-9;
                ssvo::IMUData imudata(data[1], data[2], data[3],
                                      data[4], data[5], data[6], data[0]);
                vImus.push_back(imudata);

            }
        }
    }

    void LoadImages(const string &strImagePath, const string &strPathTimes,
                    vector<string> &vstrImages, vector<double> &vTimeStamps)
    {
        ifstream fTimes;
        fTimes.open(strPathTimes.c_str());
        vTimeStamps.reserve(50000);
        vstrImages.reserve(50000);
        //int testcnt=10;
        while (!fTimes.eof()) {
            string s;
            getline(fTimes, s);
            if (!s.empty()) {
                char c = s.at(0);
                if (c < '0' || c > '9')
                    continue;

                stringstream ss;
                ss << s;
                long tmpi;
                int cnt = 0;
                while (ss >> tmpi) {
                    cnt++;
                    if (cnt == 1) {
                        //cout<<tmpi<<endl;
                        break;
                    }
                    if (ss.peek() == ',' || ss.peek() == ' ' || ss.peek() == '.')
                        ss.ignore();
                }
                //string tmpstr(strImagePath + "/" + ss.str() + ".png");
                string tmpstr(strImagePath + "/" + to_string(tmpi) + ".png");
                //cout<<tmpstr<<endl;
                vstrImages.push_back(tmpstr);
                //double t;
                //ss >> t;
                vTimeStamps.push_back(tmpi * 1e-9);

            }
        }
    }

    vector<pair<vector<ssvo::IMUData>,pair<Mat,double>>> getMeasurement_noros(vector<string> vstrImageFilenames, vector<double> vTimestamps , vector<ssvo::IMUData> vImus)
    {
        //注意这里为什么把measurements定义在while循环的外面，因为getMeasurement()一直循环进行，而且要得到measurements，定义在里面会导致清空

        vector<pair<vector<ssvo::IMUData>,pair<Mat,double>>> measurements;
        measurements.reserve(5000);
        pair<vector<ssvo::IMUData>,pair<Mat,double>> measurement;

        int nImages = vstrImageFilenames.size();
        int imuindex = 0;

        int imagestart = 0;
        const double startimutime = vImus[0]._t;
//        printf("startimutime: %.6f\n",startimutime);
//        printf("imagestarttime: %.6f\n",vTimestamps[0]);

        //! 解决图像信息太早的问题
        while(1)
        {
            if(startimutime <= vTimestamps[imagestart])
                break;
            imagestart++;

            if(sigflag) // ctrl-c exit
                return measurements;
        }
        cout<<"imagestart:"<<imagestart<<endl;

        //! 解决IMU信息太早的问题
        while(vImus[imuindex]._t<= vTimestamps[imagestart])
        {
            imuindex++;
        }
        imuindex--;

        //! 存储，配对
        Mat im;
        for(int ni=imagestart; ni<nImages; ni++)
        {

            if(sigflag) // ctrl-c exit
                break;

            // Read image from file
            im = cv::imread(vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
            double tframe = vTimestamps[ni];

            vector<ssvo::IMUData> vimu;
            while(1)
            {
                const ssvo::IMUData& imudata = vImus[imuindex];
                if(imudata._t > tframe)
                {
                    imuindex--;
                    break;
                }
                if(imuindex == (vImus.size()-1))
                    break;

                vimu.push_back(imudata);
                imuindex++;
            }

            if(imuindex == (vImus.size()-1))
                break;

            if(im.empty())
            {
                cerr << endl << "Failed to load image at: "
                     <<  vstrImageFilenames[ni] << endl;
            }

            measurement = make_pair(vimu,make_pair(im,tframe));
            measurements.push_back(measurement);
        }

    }

    void Loadconfig(const string &strconfigPath)
    {
        string config_file = strconfigPath;
        cv::FileStorage filename(config_file,cv::FileStorage::READ);
        if(!filename.isOpened())
        {
            std::cerr << "ERROR: Wrong path to settings" << std::endl;
        }
        else
        {
            cout<<"load config file successfully"<<endl;
        }

        filename["extrinsicRotation"]>> Rc2b;
        filename["extrinsicTranslation"]>> tc2b;

        cv::cv2eigen(Rc2b,eigen_Rc2b);
        cv::cv2eigen(tc2b,eigen_tc2b);

        Tc2b = SE3d(eigen_Rc2b,eigen_tc2b);

        acc_n=filename["acc_n"];
        acc_w=filename["acc_w"];
        gyr_n=filename["gyr_n"];
        gyr_w=filename["gyr_w"];

        cout<<"load finished"<<endl;
    }

