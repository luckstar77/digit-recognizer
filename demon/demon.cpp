#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "ALRect.hpp"

using namespace cv;
using namespace std;

Mat erosion_dst, dilation_dst;
int WIDTH = 160, HEIGHT = 70;
map<int, ALRect> component;
vector<ALRect> numeric;
CvSVM svm;
//#define PRINTRESULT

unsigned char *DigitRecognize(unsigned char, unsigned char *);
unsigned char *ALDigitRecognize(unsigned char, unsigned char *);
void IcvprCcaByTwoPass(const cv::Mat&, cv::Mat&);
cv::Scalar IcvprGetRandomColor();
void IcvprLabelColor(const cv::Mat& _labelImg, cv::Mat& _colorLabelImg);
bool SortLtx(const ALRect lhs,const ALRect rhs);
unsigned char SetNumericMax(unsigned char type);

int main(int argc,char** argv)
{
	//≈™®˙πœ§˘
	Mat image = imread(argv[1],CV_LOAD_IMAGE_COLOR);
	if(!image.data)
	{
		return -1;
	}
    
    WIDTH = image.cols;
    HEIGHT = image.rows;
    
    svm.load(argv[2]);
    
    ALDigitRecognize(*argv[3], image.data);
    
#ifdef PRINTRESULT
    printf("%d, %d, %d, %d, %d", result[0], result[1], result[2], result[3], result[4]);
#else
    waitKey(0);
#endif
}

unsigned char *ALDigitRecognize(unsigned char type, unsigned char *imageBuf) {
    srand(time(NULL));
   
    Mat src_gray,dst,thres,src_down;
    unsigned char result[6];
	int light=0;
    Mat src = Mat(HEIGHT, WIDTH, CV_8UC3, imageBuf);
    unsigned char numericMax = SetNumericMax(type);
	
	
    cvtColor(src,src_gray,COLOR_BGR2GRAY);
	cvtColor(src,src_down,COLOR_BGR2GRAY);
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
	
	
	dilate(src_gray,src_gray,Mat(),Point(-1,-1),1);    
	//medianBlur(src_gray,src_gray,3);
	imshow("Grayimage",src_gray); 
	
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
    
    
    threshold(src_gray,dst,T,255,THRESH_BINARY);
    threshold(src_gray,thres,T,1,THRESH_BINARY);
	threshold(src_down,src_down,T + light,255,THRESH_BINARY);
    imshow("ALthreshold",dst);
    
    cv::Mat labelImg ;
    IcvprCcaByTwoPass(thres, labelImg) ;
    
    // show result
    cv::Mat grayImg ;
    labelImg *= 10 ;
    labelImg.convertTo(grayImg, CV_8UC1) ;
    cv::imshow("labelImg", grayImg) ;

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
        bool isShow =  y + height >= cy && count >= 100 && count <= 660 && height >=20 && height < 35 ? true : false;
		char title[1000] ;        
        if(isShow) {
            numeric.push_back(iter->second);
            sprintf(title, "component : %d", iter->first);
            cout << "component ltx, lty, width, height, count : " << iter->second._ltx << ", " << iter->second._lty << ", " << iter->second._width << ", " << iter->second._height << ", " << iter->second._count << endl;
            Mat roi = src_down( Rect(iter->second._ltx,iter->second._lty,iter->second._width,iter->second._height) );            
            imshow(title,roi);
            namedWindow( title, CV_WINDOW_AUTOSIZE );
            moveWindow( title, WIDTH * 4, 0 + roi.rows * ((counts++) * 3) );
        }
    }
    
    //
    sort(numeric.begin(),numeric.end(),SortLtx);
    
    for(int i=0; i<numericMax; i++) {
        char title[1000] ;
        cout << "numeric ltx, lty, width, height, count : " << numeric[i]._ltx << ", " << numeric[i]._lty << ", " << numeric[i]._width << ", " << numeric[i]._height << ", " << numeric[i]._count << endl;
        sprintf(title, "numeric : %d", i);
        Mat roi = src_down( Rect(numeric[i]._ltx,numeric[i]._lty,numeric[i]._width,numeric[i]._height) );
        resize(roi,trainTempImg,Size(28,28));
        
        sprintf(title, "/work/shintaogas/code/shintao-recognize/train/trainTempImg%d.bmp", rand());
        imshow(title,trainTempImg);
        namedWindow( title, CV_WINDOW_AUTOSIZE );
        moveWindow( title, WIDTH * 1.5, 0 + roi.rows * ((i) * 3 ));
        imwrite(title, trainTempImg);
        
        
        HOGDescriptor *hog= new HOGDescriptor (cvSize(28,28),cvSize(14,14),cvSize(7,7),cvSize(7,7),9);
        vector<float> descriptors;
        hog->compute(trainTempImg,descriptors,Size(1,1),Size(0,0));
        printf("Hog dims: %d \n",descriptors.size());
        CvMat* SVMtrainMat = cvCreateMat(1,descriptors.size(),CV_32FC1);
        int n =0;
        for(vector<float>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)
        {
            cvmSet(SVMtrainMat,0,n,*iter);
            n++;
        }
        int ret = svm.predict(SVMtrainMat);
        result[i + 1] = ret;

    }
    //
    
    cv::Mat colorLabelImg ;
    IcvprLabelColor(labelImg, colorLabelImg) ;
    cv::imshow("colorImg", colorLabelImg) ;
	
    namedWindow( "ALthreshold", CV_WINDOW_AUTOSIZE );
    namedWindow( "labelImg", CV_WINDOW_AUTOSIZE );
    namedWindow( "colorImg", CV_WINDOW_AUTOSIZE );
    
    moveWindow( "ALthreshold", WIDTH * 2, 0 + HEIGHT * 0 );
    moveWindow( "labelImg", WIDTH * 2, 0 + HEIGHT * 2 );
    moveWindow( "colorImg", WIDTH * 2, 0 + HEIGHT * 4 );

//    result[0] = 0;  //0:成功 非0:失敗
//    result[1] = 63; //0~9:辨識值 63:無法辨識
//    result[2] = 63; //0~9:辨識值 63:無法辨識
//    result[3] = 63; //0~9:辨識值 63:無法辨識
//    result[4] = 63; //0~9:辨識值 63:無法辨識
//    result[5] = 63; //0~9:辨識值 63:無法辨識
    printf("result: %d, %d, %d, %d \n", result[1], result[2], result[3], result[4]);

    
    return result;
};

void IcvprCcaByTwoPass(const cv::Mat& _binImg, cv::Mat& _lableImg)
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

cv::Scalar IcvprGetRandomColor()
{
    uchar r = 255 * (rand()/(1.0 + RAND_MAX));
    uchar g = 255 * (rand()/(1.0 + RAND_MAX));
    uchar b = 255 * (rand()/(1.0 + RAND_MAX));
    return cv::Scalar(b,g,r) ;
}


void IcvprLabelColor(const cv::Mat& _labelImg, cv::Mat& _colorLabelImg)
{
    if (_labelImg.empty() ||
        _labelImg.type() != CV_32SC1)
    {
        return ;
    }
    
    std::map<int, cv::Scalar> colors ;
    
    int rows = _labelImg.rows ;
    int cols = _labelImg.cols ;
    
    _colorLabelImg.release() ;
    _colorLabelImg.create(rows, cols, CV_8UC3) ;
    _colorLabelImg = cv::Scalar::all(0) ;
    
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
                cv::Scalar color = colors[pixelValue] ;
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

bool SortLtx(const ALRect lhs,const ALRect rhs)
{
    return lhs._ltx < rhs._ltx ;
}

unsigned char SetNumericMax(unsigned char type) {
    switch(type) {
        case 0:
            return 4;
            break;
        default:
            return 4;
    }
}
