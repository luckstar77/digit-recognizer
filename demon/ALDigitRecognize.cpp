//
//  ALDigitRecognize.cpp
//  shintao-recognize
//
//  Created by develop on 2017/6/6.
//  Copyright © 2017年 develop. All rights reserved.
//
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#ifdef WINDOWS
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif
#include <time.h>
#include "ALRect.hpp"
#include "ALDigitRecognize.hpp"

using namespace cv;
using namespace std;

static unsigned char result[7] = {0};
const int WIDTH = 320, HEIGHT = 140;
map<int, ALRect> component;

void IcvprCcaByTwoPass(const Mat& _binImg, Mat& _lableImg);
Scalar IcvprGetRandomColor();
void IcvprLabelColor(const Mat& _labelImg, Mat& _colorLabelImg);
bool SortLtx(const ALRect lhs,const ALRect rhs);
void drawHisImg(const Mat &src,Mat &dst);
short SetNumericMax(int type);
void ShowWindow(const char *title, Mat src, int x, int y);
void drawHistImg(const Mat &src, Mat &dst);
bool FindROI(const Mat& _srcImg,Mat& _roiImg);
int ROILTX = 0, ROILTY = 0, ROIRDX = WIDTH, ROIRDY = HEIGHT;

unsigned char *ALDigitRecognize(int type, unsigned char *imageBuf, char *svmFilePath) {
#ifdef SHOWWINDOW
    srand(time(NULL));
#endif
    CvSVM svm;
    svm.load(svmFilePath);
    
    vector<ALRect> numeric;
    component.clear();
    
    memset( result, 0, 7 * sizeof(unsigned char) );
    result[0] = 1;
    
    if(svm.get_var_count() == 0) {
        result[0] = 2;
        return result;
    }
    
    Mat src_gray,dst,thres,src_down;
    Mat src = Mat(HEIGHT, WIDTH, CV_8UC3, imageBuf);
    short numericMax = SetNumericMax(type);
    ShowWindow((const char *)"src", src, WIDTH * 1.5, HEIGHT * 3);
    
    cvtColor(src,src_gray,COLOR_BGR2GRAY);
    cvtColor(src,src_down,COLOR_BGR2GRAY);
    
    Mat numbricROI = src_gray;
    int makeup = 0;
    bool bFindROI = FindROI(src, numbricROI);
    
    switch(type) {
        case 1:
        case 211:
        case 4:
        case 321:
        case 5:
        case 322:
        case 6:
        case 331:
        case 9:
        case 431:
        case 11:
        case 511:
        case 12:
        case 512:
        case 15:
        case 611:
		case 18:
		case 451:
		case 19:
		case 631:
		case 20:
		case 632:
            break;
        default:
            dilate(src_gray,src_gray,Mat(),Point(-1,-1),1);
    }
    
    ShowWindow((const char *)"Grayimage", src_gray, 0, 0);
    
    int T =0;
    double Tmax;
    double Tmin;
    minMaxIdx(src_gray,&Tmin,&Tmax);
    T = ( Tmax + Tmin ) / 2;
    printf("Brightness MIN, MAX: %f, %f\n", Tmin, Tmax);
    
    while(true)
    {
        
        int Tosum =0,Tusum =0;
        int on = 0,un =0;
        for(int i = 0;i<numbricROI.rows;i++)
        {
            for(int j = 0 ;j <numbricROI.cols; j++)
            {
                if(numbricROI.at<uchar>(i,j) >= T )
                {
                    Tosum += numbricROI.at<uchar>(i,j);
                    on ++;
                }
                else
                {
                    Tusum += numbricROI.at<uchar>(i,j);
                    un ++;
                }
            }
        }
        if(on != 0)
        {
            Tosum /=on;
        }
        else
        {
            Tosum = 0;
        }
        if(un != 0)
        {
            Tusum /=un;
        }
        else
        {
            Tusum = 0;
        }
        
        if((Tosum+Tusum) /2  != T)
            T = (Tosum+Tusum) /2;
        else
            break;
    }
    
    cout << "T : " << T << endl;

    threshold(src_gray,dst,T + makeup,255,THRESH_BINARY);
	threshold(src_gray,thres,T + makeup,1,THRESH_BINARY);
    threshold(src_down,src_down,T + makeup,255,THRESH_BINARY);    
    ShowWindow((const char *)"ALthreshold", dst, WIDTH * 2, 0);
    ShowWindow((const char *)"src_down", src_down, WIDTH * 1, 0);
    
    Mat labelImg ;
    IcvprCcaByTwoPass(thres, labelImg) ;
    
    // show result
    Mat grayImg ;
    labelImg *= 10 ;
    labelImg.convertTo(grayImg, CV_8UC1) ;
    ShowWindow((const char *)"labelImg", grayImg, WIDTH * 2, HEIGHT * 2);
    
    Mat trainTempImg= Mat(Size(48,48),8,3);
    trainTempImg.setTo(Scalar::all(0));
    int counts = 0;
    map<int, ALRect>::iterator iter;
    
    printf("ROI X RANGE : %d, %d\n", ROILTX, ROIRDX);
    
    for(iter = component.begin(); iter != component.end(); iter++) {
        int x = iter->second._ltx;
        int y = iter->second._lty;
        int width = iter->second._width;
        int height = iter->second._height;
        int count = iter->second._count;
        int cy = HEIGHT / 2;
        bool isShow =  y <= cy+1 && y + height >= cy && count >= 25 && count <= 760 && height >=11 && height <= 46 && width >= 3 && width < 46 && x >= ROILTX && x + width <= ROIRDX + 5 ? true : false;
        char title[1000] ;
        cout << "source component ltx, lty, width, height, count : " << iter->second._ltx << ", " << iter->second._lty << ", " << iter->second._width << ", " << iter->second._height << ", " << iter->second._count << endl;
        if(isShow) {
            numeric.push_back(iter->second);
            sprintf(title, "component : %d", iter->first);
            cout << "component ltx, lty, width, height, count : " << iter->second._ltx << ", " << iter->second._lty << ", " << iter->second._width << ", " << iter->second._height << ", " << iter->second._count << endl;
            Mat roi = src_down( Rect(iter->second._ltx,iter->second._lty,iter->second._width,iter->second._height) );
			ShowWindow(title, roi, WIDTH * 4, 0 + roi.rows * ((counts++) * 3));
        }
    }
    
    //
    sort(numeric.begin(),numeric.end(),SortLtx);
    
    for(int i=0; i<numeric.size(); i++) {
        char title[1000] ;
        cout << "numeric ltx, lty, width, height, count : " << numeric[i]._ltx << ", " << numeric[i]._lty << ", " << numeric[i]._width << ", " << numeric[i]._height << ", " << numeric[i]._count << endl;
        sprintf(title, "numeric : %d", i);
        Mat roi = src_down( Rect(numeric[i]._ltx,numeric[i]._lty,numeric[i]._width,numeric[i]._height) );
        resize(roi,trainTempImg,Size(48,48));
        Mat trainRoi = Mat(48,48,CV_8U, Scalar(0));
        int x = (trainRoi.rows /2)-( numeric[i]._width/2);
        int y = (trainRoi.cols /2)-( numeric[i]._height/2);
        Mat roi2 = trainRoi(Rect(x,y,roi.cols,roi.rows));
        addWeighted(roi2,0,roi,1,0,roi2);
        
#ifdef SHOWWINDOW
        getcwd(title, 1000);
        GetCurrentDir(title, 1000);
        sprintf(title, "%s/train/tmp/%d_%d.bmp", title, type, rand());
        ShowWindow(title, trainRoi, WIDTH * 1.5, 0 + trainRoi.rows * ((i) * 2 ));
        imwrite(title, trainRoi);
#endif
        
        HOGDescriptor *hog= new HOGDescriptor (cvSize(48,48),cvSize(24,24),cvSize(12,12),cvSize(6,6),9);
        vector<float> descriptors;
        hog->compute(trainRoi,descriptors,Size(1,1),Size(0,0));
        printf("Hog dims: %d \n",descriptors.size());
        CvMat* SVMtrainMat = cvCreateMat(1,descriptors.size(),CV_32FC1);
        int n =0;
        for(vector<float>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)
        {
            cvmSet(SVMtrainMat,0,n,*iter);
            n++;
        }
        int ret = svm.predict(SVMtrainMat);
        
        result[i + 1] = ret + 48;
        
        if(i + 1 == numericMax) {
            result[0] = 0;
            break;
        }
    }
    //
    
    Mat colorLabelImg ;
    IcvprLabelColor(labelImg, colorLabelImg) ;
    ShowWindow((const char *)"colorImg", colorLabelImg, WIDTH * 2, 0 + HEIGHT * 4);
    
    printf("result: %c, %c, %c, %c, %c, %d \n", result[1], result[2], result[3], result[4], result[5], result[6]);
    
    
    return result;
};

void IcvprCcaByTwoPass(const Mat& _binImg, Mat& _lableImg)
{
    // connected component analysis (4-component)
    // use two-pass algorithm
    // 1. first pass: label each foreground pixel with a label
    // 2. second pass: visit each labeled pixel and merge neighbor labels
    //
    // foreground pixel: _binImg(x,y) = 1
    // background pixel: _binImg(x,y) = 0
    
    
    if (_binImg.empty() ||
        _binImg.type() != CV_8UC1)
    {
        return ;
    }
    
    // 1. first pass
    
    _lableImg.release() ;
    _binImg.convertTo(_lableImg, CV_32SC1) ;
    
    int label = 1 ;  // start by 2
    std::vector<int> labelSet ;
    labelSet.push_back(0) ;   // background: 0
    labelSet.push_back(1) ;   // foreground: 1
    
    int rows = _binImg.rows - 1 ;
    int cols = _binImg.cols - 1 ;
    for (int i = 1; i < rows; i++)
    {
        int* data_preRow = _lableImg.ptr<int>(i-1) ;
        int* data_curRow = _lableImg.ptr<int>(i) ;
        for (int j = 1; j < cols; j++)
        {
            if (data_curRow[j] == 1)
            {
                std::vector<int> neighborLabels ;
                //                neighborLabels.reserve(2) ;
                int leftPixel = data_curRow[j-1] ;
                int upPixel = data_preRow[j] ;
                if ( leftPixel > 1)
                {
                    neighborLabels.push_back(leftPixel) ;
                }
                if (upPixel > 1)
                {
                    neighborLabels.push_back(upPixel) ;
                }
                
                if (neighborLabels.empty())
                {
                    labelSet.push_back(++label) ;  // assign to a new label
                    data_curRow[j] = label ;
                    labelSet[label] = label ;
                }
                else
                {
                    std::sort(neighborLabels.begin(), neighborLabels.end()) ;
                    int smallestLabel = neighborLabels[0] ;
                    data_curRow[j] = smallestLabel ;
                    
                    // save equivalence
                    for (size_t k = 1; k < neighborLabels.size(); k++)
                    {
                        int tempLabel = neighborLabels[k] ;
                        int oldSmallestLabel = labelSet[tempLabel] ;
                        if (oldSmallestLabel > smallestLabel)
                        {
                            labelSet[oldSmallestLabel] = smallestLabel ;
                            //                            oldSmallestLabel = smallestLabel ;
                        }
                        else if (oldSmallestLabel < smallestLabel)
                        {
                            labelSet[smallestLabel] = oldSmallestLabel ;
                        }
                    }
                }
            }
        }
    }
    
    // update equivalent labels
    // assigned with the smallest label in each equivalent label set
    for (size_t i = 2; i < labelSet.size(); i++)
    {
        int curLabel = labelSet[i] ;
        int preLabel = labelSet[curLabel] ;
        while (preLabel != curLabel)
        {
            curLabel = preLabel ;
            preLabel = labelSet[preLabel] ;
        }
        labelSet[i] = curLabel ;
    }
    
    
    // 2. second pass
    for (int i = 0; i < rows + 1; i++)
    {
        int* data = _lableImg.ptr<int>(i) ;
        for (int j = 0; j < cols + 1; j++)
        {
            int& pixelLabel = data[j] ;
            pixelLabel = labelSet[pixelLabel] ;
            
            if(pixelLabel > 1) {
                if(component.find(pixelLabel) != component.end()) {
                    if(component[pixelLabel]._ltx > j)
                        component[pixelLabel].SetLtx(j);
                    if(component[pixelLabel]._lty > i)
                        component[pixelLabel].SetLty(i);
                    if(component[pixelLabel]._rdx < j)
                        component[pixelLabel].SetRdx(j);
                    if(component[pixelLabel]._rdy < i)
                        component[pixelLabel].SetRdy(i);
                    component[pixelLabel].AddCount(1);
                } else {
                    component[pixelLabel] = ALRect(j, i, 1, 1, 1);
                }
            }
        }
    }
    
    cout << "component : " << component.size() << endl;
    cout << "CCL : " << labelSet.size() << endl;
}

Scalar IcvprGetRandomColor()
{
    uchar r = 255 * (rand()/(1.0 + RAND_MAX));
    uchar g = 255 * (rand()/(1.0 + RAND_MAX));
    uchar b = 255 * (rand()/(1.0 + RAND_MAX));
    return Scalar(b,g,r) ;
}

void IcvprLabelColor(const Mat& _labelImg, Mat& _colorLabelImg)
{
    if (_labelImg.empty() ||
        _labelImg.type() != CV_32SC1)
    {
        return ;
    }
    
    std::map<int, Scalar> colors ;
    
    int rows = _labelImg.rows ;
    int cols = _labelImg.cols ;
    
    _colorLabelImg.release() ;
    _colorLabelImg.create(rows, cols, CV_8UC3) ;
    _colorLabelImg = Scalar::all(0) ;
    
    for (int i = 0; i < rows; i++)
    {
        const int* data_src = (int*)_labelImg.ptr<int>(i) ;
        uchar* data_dst = _colorLabelImg.ptr<uchar>(i) ;
        for (int j = 0; j < cols; j++)
        {
            int pixelValue = data_src[j] ;
            if (pixelValue > 1)
            {
                if (colors.count(pixelValue) <= 0)
                {
                    colors[pixelValue] = IcvprGetRandomColor() ;
                }
                Scalar color = colors[pixelValue] ;
                *data_dst++   = color[0] ;
                *data_dst++ = color[1] ;
                *data_dst++ = color[2] ;
            }
            else
            {
                data_dst++ ;
                data_dst++ ;
                data_dst++ ;
            }
        }
    }
}

short SetNumericMax(int type) {
    switch(type) {
        case 1:
        case 211:
            return 4;
            break;
        case 2:
        case 311:
            return 4;
            break;
        case 3:
        case 312:
            return 4;
            break;
        case 4:
        case 321:
            return 4;
            break;
        case 5:
        case 322:
            return 4;
            break;
        case 6:
        case 331:
            return 4;
            break;
        case 7:
        case 411:
            return 4;
            break;
        case 8:
        case 421:
            return 5;
            break;
        case 9:
        case 431:
            return 4;
            break;
        case 10:
        case 441:
            return 5;
            break;
        case 11:
        case 511:
            return 4;
            break;
        case 12:
        case 512:
            return 4;
            break;
        case 13:
        case 521:
            return 4;
            break;
        case 14:
        case 522:
            return 4;
            break;
        case 15:
        case 611:
            return 4;
            break;
        case 16:
        case 621:
            return 4;
            break;
        case 17:
        case 622:
            return 4;
            break;
		case 18:
		case 451:
			return 5;
			break;
		case 19:
		case 631:
			return 5;
			break;
		case 20:
		case 632:
			return 5;
			break;
        default:
            return 4;
    }
}

bool SortLtx(const ALRect lhs,const ALRect rhs)
{
    return lhs._ltx < rhs._ltx ;
}

void drawHistImg(const Mat &src,Mat &dst) {
    int histSize = 256;
    float histMaxValue =0;
    for(int i =0;i<histSize;i++){
        float tempValue = src.at<float>(i);
        if(histMaxValue < tempValue){
            histMaxValue = tempValue;
        }
    }
    float scale = (0.9*256)/histMaxValue;
    for(int i = 0;i<histSize;i++){
        int intensity = static_cast<int>(src.at<float>(i)*scale);
        line(dst,Point(i,255),Point(i,255-intensity),Scalar(0));
    }
}

void ShowWindow(const char *title, Mat src, int x, int y) {
#ifdef SHOWWINDOW
    imshow(title,src);
    moveWindow( title, x, y );
#endif
}

bool FindROI(const Mat& _srcImg,Mat& _roiImg)
{
    if (_srcImg.empty() ||
        _srcImg.type() != CV_8UC3)
    {
        return false;
    }
    Mat src_gray,src_label,src_color,showImg1,showImg2;
    _srcImg.copyTo(src_gray);
    cvtColor(src_gray,src_gray,COLOR_BGR2GRAY);
    medianBlur(src_gray,src_gray,3);
    medianBlur(src_gray,src_gray,5);
    //src_gray.convertTo(src_gray,-1,-1,255);
    int value1=0,value2=0;
    for(int h =0;h< src_gray.rows-2;h++)
    {
        for(int w=0;w<src_gray.cols;w++)
        {
            value1 = src_gray.at<uchar>(h+2,w) - src_gray.at<uchar>(h,w);
            value2 = src_gray.at<uchar>(h,w) - src_gray.at<uchar>(h+2,w);
            if(value1 > value2)
                if(value1 < 15)
                    src_gray.at<uchar>(h,w) =0;
                else
                    src_gray.at<uchar>(h,w) = 255;
                else
                    if(value2 < 15)
                        src_gray.at<uchar>(h,w) = 0;
                    else
                        src_gray.at<uchar>(h,w) = 255;
        }
    }
    ShowWindow((const char *)"src_gray", src_gray,300, HEIGHT * 2);
    
    threshold(src_gray,src_gray,125,1,THRESH_BINARY);
    dilate(src_gray,src_gray,Mat(),Point(-1,-1),1);
    erode(src_gray,src_gray,Mat(),Point(-1,-1),1);
    component.clear();
    IcvprCcaByTwoPass(src_gray, src_label) ;
    ShowWindow((const char *)"src_label", src_label, 1000, HEIGHT * 4);
    IcvprLabelColor(src_label,src_color) ;    
    src_color.copyTo(showImg1);
    src_color.copyTo(showImg2);
    cvtColor(showImg1,showImg1,COLOR_BGR2GRAY);
    cvtColor(showImg2,showImg2,COLOR_BGR2GRAY);
    threshold(showImg2,showImg2,0,255,THRESH_BINARY);
    ShowWindow((const char *)"showImg1", showImg1, 1000, HEIGHT * 2);
    map<int, ALRect>::iterator iter;
    vector<ALRect> roiic;
    for(iter = component.begin(); iter != component.end(); iter++) {
        int x = iter->second._ltx;
        int y = iter->second._lty;
		int ry = iter->second._rdy;
        int width = iter->second._width;
        int height = iter->second._height;
        int count = iter->second._count;
        int cy = HEIGHT / 2;
        int cx = WIDTH / 4;
        bool isShow =   count >= 80 &&  width > 80 && x < 100 && y >= 5 && ry <= HEIGHT-5? true : false;
        char title[1000] ;
        if(isShow) {
            int newheight = height;
            int newy = y;		
            int piexnumbermax = 0;
            cout << "oldcomponent ltx, lty, width, height, count : " << iter->second._ltx << ", " << iter->second._lty << ", " << iter->second._width << ", " << iter->second._height << ", " << iter->second._count << endl;
            for(int i =y;i < y + height;i++)
            {
                int piexnumber=0;
                for(int j = x;j< x + width;j++)
                {
                    if(showImg2.at<uchar>(i,j) == 255)
                        piexnumber++;
                }
                if(piexnumber < (width/3))
                {
                    newheight--;
                    for(int j = x;j< x + width;j++)
                    {
                        showImg2.at<uchar>(i,j) = 0;
                    }
                }
            }
            sprintf(title, "component : %d", iter->first);
            cout << "newcomponent ltx, lty, width, height, count : " << iter->second._ltx << ", " << iter->second._lty << ", " << iter->second._width << ", " << iter->second._height << ", " << iter->second._count << endl;
        }
        else
        {
            for(int i = y; i < y + height; i++)
            {
                for(int j = x; j < x +width;j++)
                {
                    showImg1.at<uchar>(i,j) = 0;
                    showImg2.at<uchar>(i,j) = 0;
                }
            }
        }
    }
    ShowWindow((const char *)"showImg1", showImg1, 300, HEIGHT * 4);
    ShowWindow((const char *)"showImg2", showImg2,300, HEIGHT * 6);
    
	component.clear();
	threshold(showImg2,showImg2,125,1,THRESH_BINARY);
    IcvprCcaByTwoPass(showImg2, src_label) ;
	IcvprLabelColor(src_label,src_color) ;
	ShowWindow((const char *)"showImg3", src_color, 1000, HEIGHT * 2);

	for(iter = component.begin(); iter != component.end(); iter++) {
        int x = iter->second._ltx;
        int y = iter->second._lty;
		int ry = iter->second._rdy;
        int width = iter->second._width;
        int height = iter->second._height;
        int count = iter->second._count;
        int cy = HEIGHT / 2;
        int cx = WIDTH / 4;
        bool isShow =   count >= 80 &&  width > 80 && x < 100 && y >= 5 && ry <= HEIGHT-5? true : false;
        char title[1000] ;
        if(isShow) {
            int newheight = height;
            int newy = y;		
            int piexnumbermax = 0;
			for(int i =y;i < y + height;i++)
            {
                int piexnumber=0;
                for(int j = x;j< x + width;j++)
                {
                    if(showImg2.at<uchar>(i,j) == 255)
                        piexnumber++;
                }
                if(piexnumber > piexnumbermax && y < HEIGHT/2 && newy < y)
                {
                    newy = y;
					piexnumbermax = piexnumber;
                }
				if(piexnumber > piexnumbermax && y > HEIGHT/2 && newy > y)
				{
					newy = y;
					piexnumbermax = piexnumber;
				}

            }
			iter->second.SetLty(newy);
            cout << "oldcomponent ltx, lty, width, height, count : " << iter->second._ltx << ", " << iter->second._lty << ", " << iter->second._width << ", " << iter->second._height << ", " << iter->second._count << endl;
            roiic.push_back(iter->second);
            sprintf(title, "component : %d", iter->first);
        }
    }
    int roitopy = 0,roibottomy= HEIGHT,roitopindex =-1,roibottomindex =-1;
    for(int i = 0;i < roiic.size();i++)
    {
        if(roiic[i]._lty < HEIGHT/2 && roitopy < roiic[i]._lty)
        {
            roitopy = roiic[i]._lty;
            roitopindex = i;
        }
        else if(roiic[i]._lty > HEIGHT/2 && roibottomy > roiic[i]._lty)
        {
            roibottomy = roiic[i]._lty;
            roibottomindex = i;
        }
    }
    if(roitopindex == -1 || roibottomindex == -1)
	{
		return false;
	}
	component.clear();
    cout << "top component ltx, lty, width, height : " << roiic[roitopindex]._ltx << ", " << roiic[roitopindex]._lty << ", " << roiic[roitopindex]._width << ", " << roiic[roitopindex]._height  << endl;
    cout << "bottom component ltx, lty, width, height : " << roiic[roibottomindex]._ltx << ", " << roiic[roibottomindex]._lty << ", " << roiic[roibottomindex]._width << ", " << roiic[roibottomindex]._height  << endl;
    
    Mat roi;
    if(roiic.size() >=2 && roitopindex >= 0 && roibottomindex >= 0)
    {
		int x = roiic[roitopindex]._ltx;
		int y = roiic[roitopindex]._lty;
		int height =  roiic[roibottomindex]._rdy - y ;
		int width = roiic[roitopindex]._width;
		if(width < roiic[roibottomindex]._width)
        {
            x = roiic[roibottomindex]._ltx;
            width = roiic[roibottomindex]._width;
		}

        cout << "component ltx, lty, width, height : " << x << ", " << y << ", " << width << ", " << height  << endl;
        
        ROILTX = x;
        ROILTY = y;
        ROIRDX = x + width;
        ROIRDY = y + height;
        _roiImg = _srcImg( Rect(x,y,width,height) );
        ShowWindow((const char *)"src_roi", _roiImg,300, HEIGHT * 2);
        return true;
    } else {
        _roiImg = _srcImg;
        return false;
    }
}
