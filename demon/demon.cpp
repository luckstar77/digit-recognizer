#include <opencv2/highgui/highgui.hpp>
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
    
    printf("callback : %d, %c, %c, %c, %c, %c", result[0], result[1], result[2], result[3], result[4], result[5]);
    
	system("pause");
	waitKey(0);
}
