#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

Mat src, src_gray, dst;
Mat erosion_dst, dilation_dst;


int thresh = 150;
int max_thresh = 255;
RNG rag(12345);

int morph_elem = 2;
int morph_size = 15;
int morph_operator = 3;
int const max_operator = 4;

const char* window_name = "Morphology Transformations Demo";

void thresh_callback(int, void*);

int main(int argc, char** argv)
{
    src = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    if(!src.data)
    {
        cout << "not"<< endl;
        return -1;
    }
    
    cvtColor(src,src_gray,COLOR_BGR2GRAY);
    namedWindow(window_name, WINDOW_AUTOSIZE);
    
    thresh_callback(0,0);
    
    waitKey(0);
    return 0;
}


void thresh_callback(int, void*)
{
    Mat canny_output;
    vector<vector<Point>>contours;
    
    vector<Vec4i> hierarchy;
    Mat roi2 = src_gray(Rect(15, 50, 70, 19));
    int w = 17;
    //Mat roiA = src_gray(Rect(15,50,w,19));
    //Mat roiB = src_gray(Rect(15+w,50,w,19));
    //Mat roiC = src_gray(Rect(15+w+w,50,w,19));
    //Mat roiD = src_gray(Rect(15+w+w+w,50,w,19));
    //imshow("roiA",roiA);
    //imshow("roiB",roiB);
    //imshow("roiC",roiC);
    //imshow("roiD",roiD);§ó§ï£¾
    
    /*threshold(roiA,roiA,165,255,THRESH_BINARY);
    threshold(roiB,roiB,165,255,THRESH_BINARY);
    threshold(roiC,roiC,165,255,THRESH_BINARY);
    threshold(roiD,roiD,165,255,THRESH_BINARY);*/
    
//    imwrite("D:\sampleA.jpg",roiA);
//    imwrite("D:\sampleB.jpg",roiB);
//    imwrite("D:\sampleC.jpg",roiC);
//    imwrite("D:\sampleD.jpg",roiD);
    
    Mat element = getStructuringElement(morph_elem,
                                        Size(2*morph_size +1,2*morph_size+1),
                                        Point(morph_size,morph_size));
    //morphologyEx(roi2,dst,6,element);
    threshold(roi2,dst,168,255,THRESH_BINARY_INV);
    imshow("threshold",dst);
    
    Canny(dst,canny_output,235, 255 *2,3);
    imshow("canny",canny_output);
    
    findContours(canny_output,contours,hierarchy,
                 RETR_LIST,CHAIN_APPROX_NONE,Point(0,0));
    imshow("findContour",canny_output);
    
    vector<Rect>boundRect(contours.size());
    Mat drawing = Mat::zeros(canny_output.size(),CV_8UC3);
    for(size_t i =0;i<contours.size();i++)
    {
        Scalar color = Scalar(rag.uniform(0,255),
                              rag.uniform(0,255),rag.uniform(0,255));
        
        boundRect[i]=boundingRect(Mat(contours[i]));
        
        rectangle(drawing,boundRect[i].tl(),
                  boundRect[i].br(),color,1,8,0);
        /*drawContours(drawing, contours,(int)i,
         color,2,4,hierarchy,0,Point());*/
    }
    
    namedWindow("Contours",WINDOW_AUTOSIZE);
    imshow("Contours",drawing);
}
