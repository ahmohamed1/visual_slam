#include<iostream>
#include "opencv2/opencv.hpp"
#include "include/feature_matching.h"


int main()
{

    std::string directory = "../../datasets/00/image_0/%06d.png";
    std::string calibration_dir = "../../datasets/00/calib.txt";
    std::string ground_truth_dir ="../../datasets/00.txt";
    Feature_matching fm(calibration_dir);
    fm.loop_through_image(directory);
    return 0;
}