#include<iostream>
#include "opencv2/opencv.hpp"
#include "include/feature_matching.h"
#include "include/visual_odom_optical.h"
#include "include/stereo_visual_odom_optical.h"

int main()
{

    std::string directory = "../../datasets/00/image_0/%06d.png";
    std::string directory_r = "../../datasets/00/image_1/%06d.png";
    std::string calibration_dir = "../../datasets/00/calib.txt";
    std::string ground_truth_dir ="../../datasets/00.txt";
    // Feature_matching fm(calibration_dir, ground_truth_dir);
    Visual_Odom_optical fm(calibration_dir, ground_truth_dir);
    fm.loop_through_image(directory);

    // Stereo_Visual_Odom_optical svod(calibration_dir, ground_truth_dir);
    // svod.loop_through_image(directory, directory_r);
    return 0;
}