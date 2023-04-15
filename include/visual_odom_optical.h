#include "opencv2/opencv.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2D.hpp"
#include <fstream>

using namespace std;
using namespace cv;

/**
 * @brief Get the Absolute Scale object
 * this function took from https://github.com/avisingh599/mono-vo/blob/master/src/visodo.cpp
 * @param frame_id 
 * @param sequence_id 
 * @param z_cal 
 * @return double 
 */

double getAbsoluteScale(int frame_id, int sequence_id, double z_cal)	{
  
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
      for (int j=0; j<12; j++)  {
        in >> z ;
        if (j==7) y=z;
        if (j==3)  x=z;
      }
      
      i++;
    }
    myfile.close();
  }

  else {
    cout << "Unable to open file";
    return 0;
  }

  return sqrt((x-x_prev)*(x-x_prev) + (y-y_prev)*(y-y_prev) + (z-z_prev)*(z-z_prev)) ;

}


class Visual_Odom_optical{
    public:
        Visual_Odom_optical(std::string file_path)
        {
            load_calibration(file_path);
            print_calibrations();
            detector = ORB::create(3000);
            trajectory = cv::Mat(700, 700, CV_8UC3, cv::Scalar(255,255,255));
        }

        void print_calibrations()
        {
            std::cout<<"Camera Matrix: "<< K <<std::endl;
            std::cout<<"Focal Length: "<< focal_length <<std::endl;
            std::cout<<"Principle Point: "<< principal_point <<std::endl;
        }

        int loop_through_image(std::string file_dir)
        {
            cv::VideoCapture images;
            if( images.open(file_dir) == false)
            {
                std::cout<<"[ERROR] cannot open the folder..."<<std::endl;
                return -1;
            }
            cv::Mat frame_current;
            cv::Mat frame_old;
            std::vector<cv::KeyPoint> keypoints_old;
            
            int frame_indx = 0;
            image.read(frame_old);
            detector->detect(frame_old, keypoints_old);
            std::vector<cv::Point2f> keypoints_pos_old(keypoints_old.size()), keypoints_pos_new;
            cv::KeyPoint::convert(keypoints_old, keypoints_pos_old, vector<int>());
            vector<float> error;					
            cv::Size window_size=Size(21,21);																								
            cv::TermCriteria termcrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);
            while(images.read(frame_current))
            {
                if(keypoints_pos_old.size() < 2000)
                {
                    detector->detect(frame_old, keypoints_old);
                    cv::KeyPoint::convert(keypoints_old, keypoints_pos_old, vector<int>());
                }
                // use optical flow to detect the feature in the next frame
                std::vector<uchar> status;
                cv::calcOpticalFlowPyrLK(frame_old, frame_current, 
                                         keypoints_pos_old, keypoints_pos_new,
                                         status, error, window_size, 3, termcrit,0, 0.001);

                // remove the outlier points
                int index = 0;
                for (int i = 0; i < status.size(); i++)
                {
                    cv::Point2f point = keypoints_pos_new.at(i-index);
                    if(status.at(i) == 0 || (point.x<0 || point.y <0))
                    {
                        if(point.x<0 || point.y <0)
                            status.at(i) = 0;
                    }
                    keypoints_pos_old.eras(keypoints_pos_old.begin() + (i - index));
                    keypoints_pos_new.eras(keypoints_pos_new.begin() + (i - index));
                    index++;
                }
                
                cv::Mat R, t, essential_matrix, mask;
                essential_matrix = cv::findEssentialMat(keypoints_pos_new, keypoints_pos_old,
                                                        focal_length, principal_point, RANSAC, 0.999, 1.0, mask);

                cv::recoverPose(essential_matrix, keypoints_pos_new, keypoints_pos_old,
                                R, t, principal_point, mask);
                
                // assigne current to old
                frame_old = frame_current.clone();
                keypoints_pos_old = keypoints_pos_new;
                frame_indx ++;
                // Add to the final rotations
                double scale = getAbsoluteScale(frame_indx, 0, t.at<double>(2));
                if ((scale>0.1)&&(t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1)))
                {
                    camera_pose = camera_pose + scale*(camera_rotation*t);
                    camera_rotation = R*camera_rotation;
                }
                
                // plot the point trajectory 
                double x = camera_pose.at<double>(0)+350;
                double y = camera_pose.at<double>(2)+100;
                cv::circle(trajectory, cv::Point2d(x, y), 1, cv::Scalar(0,0,255),-1);
                cv::imshow("trajectory", trajectory);
                cv::imshow("view", frame_current);
                char ikey = waitKey(1);
                if (ikey =='q')
                    break;
                // assinge the current frmae to old frame
                
            }
            return 0;
        }

    private:
        Ptr<cv::FeatureDetector> detector;
        cv::Mat trajectory;

        // Camera Intrinsics
        cv::Mat K;
        double focal_length;
        cv::Point2d principal_point;

        cv::Mat translation_groundtruth;
        cv::Mat rotation_groundtruth;

        bool DEGUB = false;


        void load_calibration(std::string file_path)
        {
            std::string line;
            std::ifstream f(file_path);
            std::getline(f, line);
            std::istringstream ss(line);
            cv::Mat projection_matrix = cv::Mat1d(3, 4);
            for (int r = 0; r < 4; r++)
            {
                for (int c = 0; c < 3; c++)
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
            std::string line;
            std::ifstream f(file_path);
            std::getline(f, line);
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
            translation_groundtruth = tranformation_matrix(cv::Range(0, 3), cv::Range(3,4));

            // std::cout<<"tranformation_matrix: \n"<< tranformation_matrix<<std::endl;
            // std::cout<<"Rotation: "<< rotation_groundtruth<<std::endl;
            // std::cout<<"Translation: " << translation_groundtruth<<std::endl;
        }
};
