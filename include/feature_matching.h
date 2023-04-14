#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2D.hpp"
#include <fstream>

using namespace std;
using namespace cv;

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


class Feature_matching{
    public:
        Feature_matching(std::string file_path)
        {
            load_calibration(file_path);
            print_calibrations();
            detector = ORB::create();
            descriptor = ORB::create();
            matcher = DescriptorMatcher::create("BruteForce-Hamming");
            trajectory = cv::Mat(700, 700, CV_8UC3, cv::Scalar(255,255,255));
        }

        void print_calibrations()
        {
            std::cout<<"Camera Matrix: "<< K <<std::endl;
            std::cout<<"Focal Length: "<< focal_length <<std::endl;
            std::cout<<"Principle Point: "<< principal_point <<std::endl;
        }

        void detect_feature(cv::Mat img_1, cv::Mat img_2)
        {
            detector->detect(img_1, this->keypoints_1);
            detector->detect(img_2, this->keypoints_2);
            
            descriptor->compute(img_1, this->keypoints_1, this->descriptors_1);
            descriptor->compute(img_2, this->keypoints_2, this->descriptors_2);
        }

        std::vector<cv::DMatch> match_features()
        {
            //−− use Hamming distance to match the features
            std::vector<cv::DMatch> matches;
            matcher->match(this->descriptors_1, this->descriptors_2, matches);
            //−− sort and remove the outliers
            // min and max distance
            auto min_max = minmax_element(matches.begin(), matches.end(),
            [](const cv::DMatch &m1, const cv::DMatch &m2) { return m1.distance < m2.distance; });
            double min_dist = min_max.first->distance;
            double max_dist = min_max.second->distance;

            if(DEGUB)
            {
                printf("-- Max dist : %f \n", max_dist);
                printf("-- Min dist : %f \n", min_dist);
            }
            // Remove the bad matching
            std::vector<cv::DMatch> good_matches;
            for (int i = 0; i < this->descriptors_1.rows; i++)
            {
                if(matches[i].distance <= max(2 * min_dist, 30.0))
                    good_matches.push_back(matches[i]);
            }
            return good_matches;
        }

        void show_matching(cv::Mat img_1, cv::Mat img_2, std::vector<cv::DMatch> good_matches)
        {

            cv::Mat img_goodmatch;
            drawMatches(img_1, this->keypoints_1, img_2, this->keypoints_2, good_matches, img_goodmatch);
            imshow("good matches", img_goodmatch);
            char ikey = waitKey(1);
        }

    
        void pose_estimation_2d2d(std::vector<cv::KeyPoint> key_points_1,
                                  std::vector<cv::KeyPoint> key_points_2,
                                  std::vector<cv::DMatch> matches,
                                  cv::Mat &R, cv::Mat &t) 
        {
            //−− Convert the matching point to the form of vector<Point2f>
            std::vector<cv::Point2f> points1;
            std::vector<cv::Point2f> points2;

            for (int i = 0; i < (int) matches.size(); i++) {
                points1.push_back(key_points_1[matches[i].queryIdx].pt);
                points2.push_back(key_points_2[matches[i].trainIdx].pt);
            }

            //−− Calculate fundamental matrix
            cv::Mat fundamental_matrix;
            fundamental_matrix = cv::findFundamentalMat(points1, points2, CV_FM_8POINT);
            if(DEGUB)
                cout << "fundamental_matrix is " << endl << fundamental_matrix << endl;

            //−− Calculate essential matrix
            
            
            cv::Mat essential_matrix, mask;
            essential_matrix = cv::findEssentialMat(points1, points2, focal_length, principal_point, RANSAC, 0.999, 1.0, mask);
            if(DEGUB)
                cout << "essential_matrix is " << endl << essential_matrix << endl;

            //−− Calculate homography matrix
            //−− But the scene is not planar, and calculating the homography matrix here is of little significance
            cv::Mat homography_matrix;
            homography_matrix = cv::findHomography(points1, points2, RANSAC, 3);
            if(DEGUB)
                cout << "homography_matrix is " << endl << homography_matrix << endl;

            //−− Recover rotation and translation from the essential matrix.
            recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point, mask);
            if(DEGUB)
            {
                cout << "R is " << endl << R << endl;
                cout << "t is " << endl << t << endl;
            }
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
            int frame_indx = 0;
            cv::Mat camera_pose = cv::Mat::zeros(3,1, CV_64F);
            cv::Mat camera_rotation = cv::Mat::eye(3,3, CV_64F);
            while(images.read(frame_current))
            {
                if(frame_indx == 0)
                {
                    frame_old = frame_current.clone();
                    frame_indx ++;
                    continue;
                }
                cv::Mat R;
                cv::Mat t;
                detect_feature(frame_current, frame_old);
                std::vector<cv::DMatch> matches =  match_features();
                show_matching(frame_current, frame_old, matches);
                pose_estimation_2d2d(keypoints_1,
                                     keypoints_2,
                                     matches,
                                     R, t);
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
                char ikey = waitKey(1);
                if (ikey =='q')
                    break;
                // assinge the current frmae to old frame
                frame_old = frame_current.clone();
                frame_indx ++;
            }
            return 0;
        }

    private:
        std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
        Mat descriptors_1, descriptors_2;
        // Ptr<cv::FeatureDetector> detector = ORB::create();
        // cv::Ptr<cv::DescriptorExtractor> descriptor = ORB::create();
        // Ptr<cv::DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce−Hamming");
        Ptr<cv::FeatureDetector> detector;
        cv::Ptr<cv::DescriptorExtractor> descriptor;
        Ptr<cv::DescriptorMatcher> matcher;

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
