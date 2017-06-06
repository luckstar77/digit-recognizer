#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "ALDigitRecognize.hpp"

using namespace cv;
using namespace std;

int main(int argc,char** argv)
{
	Mat image = imread(argv[1],CV_LOAD_IMAGE_COLOR);
    unsigned char *result;
	if(!image.data)
	{
		return -1;
	}
    
    result = ALDigitRecognize(*argv[3], image.data, argv[2]);
    
    printf("callback : %d, %d, %d, %d, %d, %d", result[0], result[1], result[2], result[3], result[4], result[5]);
    waitKey(0);
}
