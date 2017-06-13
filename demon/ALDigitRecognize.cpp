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
//#include <unistd.h>
#include "ALRect.hpp"
#include "ALDigitRecognize.hpp"

using namespace cv;
using namespace std;

int WIDTH = 320, HEIGHT = 140;
map<int, ALRect> component;
vector<ALRect> numeric;
CvSVM svm;

void IcvprCcaByTwoPass(const Mat& _binImg, Mat& _lableImg);
Scalar IcvprGetRandomColor();
void IcvprLabelColor(const Mat& _labelImg, Mat& _colorLabelImg);
bool SortLtx(const ALRect lhs,const ALRect rhs);
void drawHisImg(const Mat &src,Mat &dst);
short SetNumericMax(unsigned char type);
void ShowWindow(const char *title, Mat src, int x, int y);
void drawHistImg(const Mat &src, Mat &dst);

unsigned char *ALDigitRecognize(unsigned char type, unsigned char *imageBuf, char *svmFilePath) {
#ifdef SHOWWINDOW
    srand(time(NULL));
#endif
    static unsigned char result[7] = {0};
    component.clear();
    numeric.clear();
    result[0] = 1;
    svm.load(svmFilePath);
    Mat src_gray,dst,thres,src_down;
    int light=0;
    Mat src = Mat(HEIGHT, WIDTH, CV_8UC3, imageBuf);
    short numericMax = SetNumericMax(type);
    ShowWindow((const char *)"src", src, WIDTH * 1.5, HEIGHT * 3);
    
    cvtColor(src,src_gray,COLOR_BGR2GRAY);
    cvtColor(src,src_down,COLOR_BGR2GRAY);
    int histSize = 256;
    float rang[] = {0,255};
    const float* histRange = {rang};
    Mat histImg;
    calcHist(&src_gray,1,0,Mat(),histImg,1,&histSize,&histRange);
    Mat showHistImg(256,256,CV_8UC1,Scalar(255));
    drawHistImg(histImg,showHistImg);
    ShowWindow((const char *)"srcHistimg", showHistImg, 0, HEIGHT * 1.5);
    
    while(true)
    {
        int piexl[3] = {0};
        for(int i =0;i<src_gray.rows;i++)
        {
            for(int j = 0;j< src_gray.cols;j++)
            {
                if(src_gray.at<uchar>(i,j) < 120 ){
                    piexl[0] ++;}
                else if(src_gray.at<uchar>(i,j) <= 230){
                    piexl[1] ++;}
                else {
                    piexl[2] ++;}
            }
        }
        if(piexl[0] > (piexl[1]+(src_gray.rows *  src_gray.cols) / 6) && piexl[0] > piexl[2]){
            src_gray.convertTo(src_gray,-1,1,15);
            light -=15;
        }
        else if(piexl[2] > (piexl[1]+(src_gray.rows *  src_gray.cols) / 6) && piexl[2] > piexl[0]){
            src_gray.convertTo(src_gray,-1,1,-15);
            light +=15;
        }
        else
            break;
    }
    //equalizeHist(src_gray,src_gray);
    
    //dilate(src_gray,src_gray,Mat(),Point(-1,-1),1);
    calcHist(&src_gray,1,0,Mat(),histImg,1,&histSize,&histRange);
    showHistImg = Mat(256,256,CV_8UC1,Scalar(255));
    drawHistImg(histImg,showHistImg);
    ShowWindow((const char *)"srcHistimg2", showHistImg, 0, HEIGHT * 3);
    
    dilate(src_gray,src_gray,Mat(),Point(-1,-1),1);
    //medianBlur(src_gray,src_gray,3);
    ShowWindow((const char *)"Grayimage", src_gray, 0, 0);
    
    int T =0;
    double Tmax;
    double Tmin;;
    minMaxIdx(src_gray,&Tmin,&Tmax);
    T = ( Tmax + Tmin ) / 2;
    printf("Brightness MIN, MAX: %f, %f\n", Tmin, Tmax);
    
    while(true)
    {
        
        int Tosum =0,Tusum =0;
        int on = 0,un =0;
        for(int i = 0;i<src_gray.rows;i++)
        {
            for(int j = 0 ;j <src_gray.cols; j++)
            {
                if(src_gray.at<uchar>(i,j) >= T )
                {
                    Tosum += src_gray.at<uchar>(i,j);
                    on ++;
                }
                else
                {
                    Tusum += src_gray.at<uchar>(i,j);
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
    
    Mat test ;
    adaptiveThreshold(src_gray, test, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY,129, 0);
    //medianBlur(test,test,3);
    dilate(test,test,Mat(),Point(-1,-1),1);
    //imshow("adaptiv",test);
    threshold(src_gray,dst,T,255,THRESH_BINARY);
    threshold(src_gray,thres,T,1,THRESH_BINARY);
    threshold(src_down,src_down,T + light,255,THRESH_BINARY);
    ShowWindow((const char *)"ALthreshold", dst, WIDTH * 2, 0);
    
    Mat labelImg ;
    IcvprCcaByTwoPass(thres, labelImg) ;
    
    // show result
    Mat grayImg ;
    labelImg *= 10 ;
    labelImg.convertTo(grayImg, CV_8UC1) ;
    ShowWindow((const char *)"labelImg", grayImg, WIDTH * 2, HEIGHT * 2);
    
    /*CvSVM svm;
     svm.load("D:\\OCR\\digital-recognize\\demon\\gas.xml");*/
    Mat trainTempImg= Mat(Size(28,28),8,3);
    
    trainTempImg.setTo(Scalar::all(0));
    int counts = 0;
    map<int, ALRect>::iterator iter;
    for(iter = component.begin(); iter != component.end(); iter++) {
        int x = iter->second._ltx;
        int y = iter->second._lty;
        int width = iter->second._width;
        int height = iter->second._height;
        int count = iter->second._count;
        int cy = HEIGHT / 2;
        bool isShow =  y <= cy && y + height >= cy && count >= 70 && count <= 760 && height >=17 && height < 45 ? true : false;
        char title[1000] ;
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
        if (i >= numericMax) {
            result[0] = 7;
            break;
        } else if (i + 1 == numericMax) {
            result[0] = 0;
        }
        
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
        cout << title << "/train/tmp/" << endl;
        sprintf(title, "%s/train/tmp/%d_%d.bmp", title, 1, rand());
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
        
        if(i < numericMax) {
            result[i + 1] = ret + 48;
        }
        else{
            break;}
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

short SetNumericMax(unsigned char type) {
    switch(type) {
        case 1:
            return 4;
            break;
        case 4:
            return 4;
            break;
        case 5:
            return 4;
            break;
        case 6:
            return 4;
            break;
        case 7:
            return 4;
            break;
        case 8:
            return 5;
            break;
        case 9:
            return 4;
            break;
        case 10:
            return 5;
            break;
        case 11:
            return 4;
            break;
        case 12:
            return 4;
            break;
        case 13:
            return 4;
            break;
        case 14:
            return 4;
            break;
        case 15:
            return 4;
            break;
        case 16:
            return 4;
            break;
        case 17:
            return 4;
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

