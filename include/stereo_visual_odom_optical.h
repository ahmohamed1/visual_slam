#include "opencv2/opencv.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2D.hpp"
#include <fstream>

using namespace std;
using namespace cv;




class Stereo_Visual_Odom_optical{
    public:
        Stereo_Visual_Odom_optical(std::string file_path, std::string ground_truth="")
        {
            load_calibration(file_path);
            print_calibrations();
            detector = ORB::create(4000);

            // initialize stereo matchers
            sgbm = cv::StereoSGBM::create(0, 96, 9, 8 * 9*9, 32*9*9, 1, 63, 10, 100, 32);

            trajectory = cv::Mat(700, 700, CV_8UC3, cv::Scalar(255,255,255));
            cv::rectangle(trajectory,cv::Point2f(10,10), cv::Point2f(130,70), cv::Scalar(0,0,0),2);
            cv::putText(trajectory, "gound truth", cv::Point2f(15,30),1 ,1.1, cv::Scalar(255,0,0));
            cv::putText(trajectory, "Computed", cv::Point2f(15,55),1, 1.1, cv::Scalar(0,0,255));
            if(ground_truth != "")
            {
                load_ground_truth(ground_truth);
            }
        }

        void print_calibrations()
        {
            std::cout<<"Camera Matrix: \n"<< K <<std::endl;
            std::cout<<"Focal Length: \n"<< focal_length <<std::endl;
            std::cout<<"Principle Point: \n"<< principal_point <<std::endl;
        }

        int loop_through_image(std::string file_dir)
        {
            cv::VideoCapture images_r;
            cv::VideoCapture images_l;
            if( images_r.open(file_dir) == false)
            {
                std::cout<<"[ERROR] cannot open right the folder..."<<std::endl;
                return -1;
            }
            if( images_l.open(file_dir) == false)
            {
                std::cout<<"[ERROR] cannot open left the folder..."<<std::endl;
                return -1;
            }

            cv::Mat frame_left_current;
            cv::Mat frame_right_current;
            cv::Mat frame_left_old;
            cv::Mat frame_right_old;
            cv::Mat disparity_map;
            std::vector<cv::KeyPoint> keypoints_left_old;
            
            int frame_indx = 2;
            images_r.read(frame_right_old);
            images_l.read(frame_left_old);

            // compute feature in single frame
            detector->detect(frame_left_old, keypoints_left_old);
            std::vector<cv::Point2f> keypoints_left_pos_old(keypoints_left_old.size()), keypoints_left_pos_new;
            cv::KeyPoint::convert(keypoints_left_old, keypoints_left_pos_old, vector<int>());
            
            //compute disparity map
            sgbm->compute(frame_left_old, frame_right_old, disparity_map);
            
            // compute 3d position of the feature detected
            std::vector<cv::Point3f> keyPoint_3d_pose;
            compute_3D_pose_points(keypoints_left_pos_old, disparity_map, keyPoint_3d_pose);

            images_r.read(frame_right_current);
            track_features(frame_right_old, frame_right_current, keypoints_left_pos_old, keypoints_left_pos_new);
            compute_position(keypoints_left_pos_old, keypoints_left_pos_new, frame_indx);

            
            while(images_l.read(frame_left_current))
            {
                if(keypoints_left_pos_old.size() < 2000)
                {
                    detector->detect(frame_left_old, keypoints_left_old);
                    cv::KeyPoint::convert(keypoints_left_old, keypoints_left_pos_old, vector<int>());
                }

                
                frame_indx ++;
                // plot the point trajectory 
                bool break_loop = plot_path(frame_left_current, frame_indx);
                if (break_loop)
                    break;
            }   
            return 0;
        }

        void compute_3D_pose_points(std::vector<cv::Point2f> keypoints, cv::Mat disparity_map, std::vector<cv::Point3f> &keyPoint_3d_pose)
            {
                for (int i = 0; i < keypoints.size(); i++)
                {
                    int v = keypoints[i].x;
                    int u = keypoints[i].y;

                    if(disparity_map.at<float>(v,u) <=10.0 || disparity_map.at<float>(v,u) >=96.0)
                        continue;
                    cv::Point3f point_3d;
                    point_3d.x = (u - principal_point.x) / focal_length;
                    point_3d.y = (v - principal_point.y) / focal_length;
                    point_3d.z = focal_length * baseline / (disparity_map.at<float>(v,u));
                    keyPoint_3d_pose.push_back(point_3d);
                }
                
            }
    private:
        Ptr<cv::FeatureDetector> detector;
        cv::Ptr<cv::StereoSGBM> sgbm;
        cv::Mat trajectory;

        // Camera Intrinsics
        cv::Mat K;
        double focal_length;
        cv::Point2d principal_point;

        cv::Mat camera_pose = cv::Mat::zeros(3,1, CV_64F);
        cv::Mat camera_rotation = cv::Mat::eye(3,3, CV_64F);

        std::vector<cv::Mat> translation_groundtruth;
        cv::Mat rotation_groundtruth;
        cv::Size window_size=Size(21,21);																								
        cv::TermCriteria termcrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);
        vector<float> _error;

        bool DEGUB = false;

        void track_features(cv::Mat frame_old, cv::Mat frame_current,
                            std::vector<cv::Point2f>& keypoints_pos_old, 
                            std::vector<cv::Point2f>& keypoints_pos_new)
        {
            // use optical flow to detect the feature in the next frame
            std::vector<uchar> status;
            cv::calcOpticalFlowPyrLK(frame_old, frame_current, 
                                    keypoints_pos_old, keypoints_pos_new,
                                    status, _error, window_size, 3, termcrit,0, 0.001);
        
            // remove the outlier points
            int index = 0;
            for (int i = 0; i < status.size(); i++)
            {
                cv::Point2f point = keypoints_pos_new.at(i-index);
                if((status.at(i) == 0) || (point.x<0) || (point.y <0))
                {
                    if((point.x<0) || (point.y <0))
                        status.at(i) = 0;
                    keypoints_pos_old.erase(keypoints_pos_old.begin() + (i - index));
                    keypoints_pos_new.erase(keypoints_pos_new.begin() + (i - index));
                    index++;
                }
            }
        }

        void compute_position(std::vector<cv::Point2f> keypoints_pos_old, 
                              std::vector<cv::Point2f> keypoints_pos_new,
                              int frame_index)
        {
            cv::Mat R, t, mask;
            cv::Mat essential_matrix = cv::findEssentialMat(keypoints_pos_new, keypoints_pos_old,
                                                    focal_length, principal_point, RANSAC, 0.999, 1.0, mask);
            cv::recoverPose(essential_matrix, keypoints_pos_new, keypoints_pos_old, R, t, 
                            focal_length, principal_point, mask);
                            // Add to the final rotations
            double scale = getAbsoluteScale(frame_index, 0, t.at<double>(2));
            if ((scale>0.1)&&(t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1)))
            {
                camera_pose = camera_pose + scale*(camera_rotation*t);
                camera_rotation = R*camera_rotation;
            }
            // This is the first iteration
            if(frame_index == 2)
            {
                camera_pose = t.clone();
                camera_rotation = R.clone();
            }
        }
        void load_calibration(std::string file_path)
        {
            std::string line;
            std::ifstream f(file_path);
            std::getline(f, line);
            std::istringstream ss(line);
            cv::Mat projection_matrix = cv::Mat1d(3, 4);
            for (int r = 0; r < 3; r++)
            {
                for (int c = 0; c < 4; c++)
                {
                    double data =0.0f;
                    ss >> data;
                    projection_matrix.at<double>(r,c) = data;
                }
            }
            K = projection_matrix(cv::Range(0, 3), cv::Range(0, 3));
            focal_length = projection_matrix.at<double>(0,0);
            double cx = projection_matrix.at<double>(0,2);
            double cy = projection_matrix.at<double>(1,2);
            principal_point = cv::Point2d(cx,cy);
        }

        void load_ground_truth(std::string file_path)
        {
            std::cout<<"[INFO] Load ground truth data.. \n";
            std::string line;
            std::ifstream f(file_path);
            
            while(std::getline(f, line))
            {
                std::istringstream ss(line);
                cv::Mat tranformation_matrix = cv::Mat1d(3, 4);
                for (int r = 0; r < 3; r++)
                {
                    for (int c = 0; c < 4; c++)
                    {
                        double data =0.0f;
                        ss >> data;
                        tranformation_matrix.at<double>(r,c) = data;
                    }
                }
                rotation_groundtruth = tranformation_matrix(cv::Range(0, 3), cv::Range(0, 3));
                translation_groundtruth.push_back(tranformation_matrix(cv::Range(0, 3), cv::Range(3,4)));
            }
            // std::cout<<"tranformation_matrix: \n"<< tranformation_matrix<<std::endl;
            // std::cout<<"Rotation: "<< rotation_groundtruth<<std::endl;
            // std::cout<<"Translation: " << translation_groundtruth<<std::endl;
        }

        /**
         * @brief Get the Absolute Scale object
         * this function took from https://github.com/avisingh599/mono-vo/blob/master/src/visodo.cpp
         * @param frame_id 
         * @param sequence_id 
         * @param z_cal 
         * @return double 
         */

        double getAbsoluteScale(int frame_id, int sequence_id, double z_cal)	
        {
            string line;
            int i = 0;
            ifstream myfile ("../../datasets/00.txt");
            double x =0, y=0, z = 0;
            double x_prev, y_prev, z_prev;
            if (myfile.is_open())
            {
                while (( getline (myfile,line) ) && (i<=frame_id))
                {
                    z_prev = z;
                    x_prev = x;
                    y_prev = y;
                    std::istringstream in(line);
                    //cout << line << '\n';
                    for (int j=0; j<12; j++)  
                    {
                        in >> z ;
                        if (j==7) y=z;
                        if (j==3)  x=z;
                    }
                    i++;
                }
                myfile.close();
            }
            else 
            {
                cout << "Unable to open file";
                return 0;
            }
            return sqrt((x-x_prev)*(x-x_prev) + (y-y_prev)*(y-y_prev) + (z-z_prev)*(z-z_prev)) ;
        }

        bool plot_path(cv::Mat image,int frame_indx)
        {
            double x = camera_pose.at<double>(0)+350;
            double y = camera_pose.at<double>(2)+100;
            cv::circle(trajectory, cv::Point2d(x, y), 1, cv::Scalar(0,0,255),-1);

            // plote the ground truth
            double y_gt = translation_groundtruth[frame_indx].at<double>(2) + 100;
            double x_gt = translation_groundtruth[frame_indx].at<double>(0) + 350;
            cv::circle(trajectory, cv::Point2d(x_gt, y_gt), 1, cv::Scalar(250,0,0),-1);

            cv::imshow("trajectory", trajectory);
            cv::imshow("view", image);
            char ikey = waitKey(1);
            if (ikey =='q')
                return true;
            else
                return false;
        }
};
