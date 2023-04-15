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
        Feature_matching(std::string file_path, std::string ground_truth="")
        {
            load_calibration(file_path);
            print_calibrations();
            detector = ORB::create();
            descriptor = ORB::create();
            matcher = DescriptorMatcher::create("BruteForce-Hamming");
            trajectory = cv::Mat(700, 700, CV_8UC3, cv::Scalar(255,255,255));

            if(ground_truth != "")
            {
                load_ground_truth(ground_truth);
            }
        }

        void print_calibrations()
        {
            std::cout<<"Camera Matrix: "<< K <<std::endl;
            std::cout<<"Focal Length: "<< focal_length <<std::endl;
            std::cout<<"Principle Point: "<< principal_point <<std::endl;
        }

        void detect_feature(cv::Mat img_1, cv::Mat img_2)
        {
            detector->detect(img_1, this->keypoints_prevouse);
            detector->detect(img_2, this->keypoints_current);
            
            descriptor->compute(img_1, this->keypoints_prevouse, this->descriptors_1);
            descriptor->compute(img_2, this->keypoints_current, this->descriptors_2);
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
            drawMatches(img_1, this->keypoints_prevouse, img_2, this->keypoints_current, good_matches, img_goodmatch);
            imshow("good matches", img_goodmatch);
            char ikey = waitKey(1);
        }

    
        void pose_estimation_2d2d(std::vector<cv::KeyPoint> key_points_prvouse,
                                  std::vector<cv::KeyPoint> key_points_current,
                                  std::vector<cv::DMatch> matches,
                                  cv::Mat &R, cv::Mat &t) 
        {
            //−− Convert the matching point to the form of vector<Point2f>
            std::vector<cv::Point2f> points_prvouse;
            std::vector<cv::Point2f> points_current;

            for (int i = 0; i < (int) matches.size(); i++) {
                points_prvouse.push_back(key_points_prvouse[matches[i].queryIdx].pt);
                points_current.push_back(key_points_current[matches[i].trainIdx].pt);
            }

            //−− Calculate essential matrix
            cv::Mat essential_matrix, mask, R_1,R_2,T_;
            essential_matrix = cv::findEssentialMat(points_current, points_prvouse, focal_length, principal_point, RANSAC, 0.999, 1.0);
            if(DEGUB)
                cout << "essential_matrix is " << endl << essential_matrix << endl;
                
            //−− Recover rotation and translation from the essential matrix.
            cv::recoverPose(essential_matrix, points_current, points_prvouse, R, t, focal_length, principal_point);
            if(cv::determinant(R) < 0)
            {
                R = -R;
                t = -t;
            }

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
                detect_feature(frame_old,frame_current);
                std::vector<cv::DMatch> matches =  match_features();
                show_matching(frame_old, frame_current, matches);
                pose_estimation_2d2d(keypoints_prevouse,
                                     keypoints_current,
                                     matches,
                                     R, t);
                // Add to the final rotations
                double scale = getAbsoluteScale(frame_indx-1, 0, t.at<double>(2));
                if ((scale>0.1)&&(t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1)))
                {
                    camera_pose = camera_pose + scale*(camera_rotation*t);
                    camera_rotation = R*camera_rotation;
                }
                
                // plot the point trajectory 
                double x = camera_pose.at<double>(0)+350;
                double y = camera_pose.at<double>(2)+100;
                // std::cout<<x <<" , "<< y <<std::endl;
                cv::circle(trajectory, cv::Point2d(x, y), 1, cv::Scalar(0,0,255),-1);

                // plote the ground truth
                double y_gt = translation_groundtruth[frame_indx].at<double>(2) + 100;
                double x_gt = translation_groundtruth[frame_indx].at<double>(0) + 350;
                cv::circle(trajectory, cv::Point2d(x_gt, y_gt), 1, cv::Scalar(250,0,0),-1);
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
        std::vector<cv::KeyPoint> keypoints_prevouse, keypoints_current;
        Mat descriptors_1, descriptors_2;
        Ptr<cv::FeatureDetector> detector;
        cv::Ptr<cv::DescriptorExtractor> descriptor;
        Ptr<cv::DescriptorMatcher> matcher;

        cv::Mat trajectory;

        // Camera Intrinsics
        cv::Mat K;
        double focal_length;
        cv::Point2d principal_point;

        std::vector<cv::Mat> translation_groundtruth;
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


        cv::Mat form_transf(cv::Mat R, cv::Mat t) {
            cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
            cv::Mat R_part = T(cv::Rect(0, 0, 3, 3));
            R.copyTo(R_part);
            cv::Mat t_part = T(cv::Rect(3, 0, 1, 3));
            cv::Mat(t).reshape(1, 3).copyTo(t_part);
            return T;
        }

        void decomposeEssentialMat(const Mat& E, Mat& R, Mat& t, const Mat& K, const std::vector<Point2f>& pts1, const std::vector<Point2f>& pts2)
        {
            Matx33d W(0,-1,0,1,0,0,0,0,1); // skew-symmetric matrix for right-handed coordinates
            SVD svd(E, SVD::FULL_UV); // perform singular value decomposition

            Matx33d U = svd.u; // get left singular vectors
            Matx33d Vt = svd.vt; // get right singular vectors

            // ensure that det(U) and det(V) are positive
            if (determinant(U) < 0)
                U *= -1;
            if (determinant(Vt) < 0)
                Vt *= -1;

            // compute possible rotation matrices
            Matx33d R1x = U * Matx33d(W) * U.t();
            Matx33d R2x = U * Matx33d(-W) * U.t();

            // ensure that R1 and R2 are valid rotation matrices (i.e. det(R) = 1)
            if (determinant(Mat(R1x)) < 0)
                R1x *= -1;
            if (determinant(Mat(R2x)) < 0)
                R2x *= -1;

            // compute possible translation vectors
            Matx31d t1 = U.col(2);
            Matx31d t2 = -U.col(2);

            // triangulate 3D points to select the most correct solution
            Matx34d P1 = Matx34d::eye();
            Matx34d P2_1(R1x(0,0), R1x(0,1), R1x(0,2), t1(0),
                         R1x(1,0), R1x(1,1), R1x(1,2), t1(1),
                         R1x(2,0), R1x(2,1), R1x(2,2), t1(2));
            Matx34d P2_2(R1x(0,0), R1x(0,1), R1x(0,2), t2(0),
                         R1x(1,0), R1x(1,1), R1x(1,2), t2(1),
                         R1x(2,0), R1x(2,1), R1x(2,2), t2(2));
            Matx34d P2_3(R2x(0,0), R2x(0,1), R2x(0,2), t1(0),
                         R2x(1,0), R2x(1,1), R2x(1,2), t1(1),
                         R2x(2,0), R2x(2,1), R2x(2,2), t1(2));
            Matx34d P2_4(R2x(0,0), R2x(0,1), R2x(0,2), t2(0),
                         R2x(1,0), R2x(1,1), R2x(1,2), t2(1),
                         R2x(2,0), R2x(2,1), R2x(2,2), t2(2));

            Matx44d A;
            A.row(0) = pts1[0].x * P1.row(2) - P1.row(0);
            A.row(1) = pts1[0].y * P1.row(2) - P1.row(1);
            A.row(2) = pts2[0].x * P2_1.row(2) - P2_1.row(0);
            A.row(3) = pts2[0].y * P2_1.row(2) - P2_1.row(1);

            Mat_<double> u, vt, w;
            SVD::compute(A, w, u, vt);

            Matx41d Xh = vt.row(3);
            Matx31d X(Xh(0)/Xh(3), Xh(1)/Xh(3), Xh(2)/Xh(3));

            int num_in_front_1 = 0, num_in_front_2 = 0, num_in_front_3 = 0, num_in_front_4 = 0;
            for (int i = 0; i < pts1.size(); i++)
            {
                Matx31d Xh1 = P2_1 * Matx41d(X(0), X(1), X(2), 1);
                Matx31d X1(Xh1(0)/Xh1(2), Xh1(1)/Xh1(2), Xh1(2)/Xh1(2));
                if (X1(2) > 0)
                    num_in_front_1++;

                Matx31d Xh2 = P2_2 * Matx41d(X(0), X(1), X(2), 1);
                Matx31d X2(Xh2(0)/Xh2(2), Xh2(1)/Xh2(2), Xh2(2)/Xh2(2));
                if (X2(2) > 0)
                    num_in_front_2++;

                Matx31d Xh3 = P2_3 * Matx41d(X(0), X(1), X(2), 1);
                Matx31d X3(Xh3(0)/Xh3(2), Xh3(1)/Xh3(2), Xh3(2)/Xh3(2));
                if (X3(2) > 0)
                    num_in_front_3++;

                Matx31d Xh4 = P2_4 * Matx41d(X(0), X(1), X(2), 1);
                Matx31d X4(Xh4(0)/Xh4(2), Xh4(1)/Xh4(2), Xh4(2)/Xh4(2));
                if (X4(2) > 0)
                    num_in_front_4++;
            }

            int max_num_in_front = std::max(std::max(num_in_front_1, num_in_front_2), std::max(num_in_front_3, num_in_front_4));
            if (max_num_in_front == num_in_front_1)
            {
                R = Mat(R1x);
                t = Mat(t1);
            }
            else if (max_num_in_front == num_in_front_2)
            {
                R = Mat(R1x);
                t = Mat(t2);
            }
            else if (max_num_in_front == num_in_front_3)
            {
                R = Mat(R2x);
                t = Mat(t1);
            }
            else
            {
                R = Mat(R2x);
                t = Mat(t2);
            }
        }

};
