#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "ALRect.hpp"

using namespace cv;
using namespace std;

Mat erosion_dst, dilation_dst;
const int WIDTH = 160, HEIGHT = 70;
map<int, ALRect> component;
//#define PRINTRESULT

unsigned char *DigitRecognize(unsigned char, unsigned char *);
unsigned char *ALDigitRecognize(unsigned char, unsigned char *);
void IcvprCcaByTwoPass(const cv::Mat&, cv::Mat&);
cv::Scalar IcvprGetRandomColor();
void IcvprLabelColor(const cv::Mat& _labelImg, cv::Mat& _colorLabelImg);

int main(int argc,char** argv)
{

	//≈™®˙πœ§˘
	Mat image = imread(argv[1],CV_LOAD_IMAGE_COLOR);
    unsigned char *result;
	if(!image.data)
	{
		return -1;
	}
    
    //result = DigitRecognize(0, image.data);
    ALDigitRecognize(0, image.data);
    
#ifdef PRINTRESULT
    printf("%d, %d, %d, %d, %d", result[0], result[1], result[2], result[3], result[4]);
#else
    waitKey(0);
#endif
}

unsigned char *DigitRecognize(unsigned char type, unsigned char *imageBuf) {
    Mat src_gray,dst;
    static unsigned char result[6];
    Mat src = Mat(HEIGHT, WIDTH, CV_8UC3, imageBuf);
    //¬‡¶«∂•πœ
    cvtColor(src,src_gray,COLOR_BGR2GRAY);
    imshow("Grayimage",src_gray);
    
    //©Ò§j
    /*pyrUp(src_gray,src_gray,Size(src.cols*2,src.rows*2));
     imshow("up",src_gray);*/
    
    //∞™¥µ¬o™i
    /*blur(src_gray,dst,Size(3,3));
     imshow("blur",dst);*/
    
    ////©Ò§j
    //pyrUp(src_gray,src_gray,Size(src.cols*2,src.rows*2));
    //imshow("up",src_gray);
    
    int T =0;
    double Tmax;
    double Tmin;;
    minMaxIdx(src_gray,&Tmin,&Tmax);
    T = ( Tmax + Tmin ) / 2;
    
    printf("Brightness MIN, MAX: %f, %f\n", Tmin, Tmax);
    
    while(true)
    {
        
        int Tosum =0,Tusum =0; //osum∂WπLT•[¡` usum §p©ÛT•[¡`
        int on = 0,un =0;  //on∂WπLT™∫¡`º∆ un §p©ÛT™∫¡`º∆
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
        Tosum /=on;
        Tusum /=un;
        if((Tosum+Tusum) /2  != T)
            T = (Tosum+Tusum) /2;
        else
            break;
    }
    
    
    threshold(src_gray,dst,T,255,THRESH_BINARY);
    //imshow("threshold",dst);
    
    //ø±µ»
    //erode(dst,dst,Mat(),Point(-1,-1),1);
    //dilate(dst,dst,Mat(),Point(-1,-1),1);
    //imshow("1",dst);
    
    int row = 1;
    row = dst.rows;
    int colum = 1;
    int** iarry = new int*[dst.rows];
    for(int i = 0;i<dst.rows;i++)
    {
        iarry[i] = new int [dst.cols];
    }
    for(int i = 0;i< dst.rows;i++)
    {
        for(int j = 0 ;j<dst.cols;j++)
        {
            iarry[i][j] = 0;
        }
    }
    
    int n = 1;
    
    for(int x = 0;x< 2;x++){
        for(int i = 1;i< dst.rows - 1;i++)   //§¿∏s
        {
            for(int j = 1 ;j<dst.cols-1;j++)
            {
                if(dst.at<uchar>(i,j) == 255)
                {
                    if(iarry[i-1][j+1] ==0 && iarry[i][j-1] ==0 && iarry[i-1][j] ==0)//(B L U)
                        iarry[i][j] = n++;  //N=new
                    else if(iarry[i-1][j+1] !=0 && iarry[i][j-1] ==0 && iarry[i-1][j] ==0 )
                        iarry[i][j] = iarry[i-1][j+1];  //N=B
                    else if(iarry[i][j-1] ==0 && iarry[i-1][j] !=0) //(L U)
                        iarry[i][j] = iarry[i-1][j];    //N=U
                    else if(iarry[i][j-1] !=0 && iarry[i-1][j] ==0) //(L U)
                        iarry[i][j] = iarry[i][j-1];    //N=L
                    else if(iarry[i][j-1] !=0 && iarry[i-1][j] !=0 && iarry[i][j-1] == iarry[i-1][j] )
                        iarry[i][j] = iarry[i][j-1];    //N=L
                    else if(iarry[i][j-1] !=0 && iarry[i-1][j] !=0 && iarry[i][j-1] != iarry[i-1][j])
                    {
                        iarry[i][j] = iarry[i][j-1];    //N=L
                        iarry[i][j-1] = iarry[i][j-1];
                    }
                }
            }
        }
    }
    
    printf("Label Numberic: %d\n", n);
    
    
    for(int i = 0;i< dst.rows;i++)  //¥˙∏’πœ
    {
        for(int j = 0 ;j<dst.cols;j++)
        {
            if(dst.at<uchar>(i,j) == 255)
            {
                dst.at<uchar>(i,j) = iarry[i][j]*10;
                
            }
            else
                dst.at<uchar>(i,j) = 0;
        }
    }
    //imshow("02",dst);
    
    int* sum = new int[n];
    for(int i =0;i<n;i++) //™Ï©l§∆
    {
        sum[i] =0;
    }
    for(int i = 0;i< dst.rows;i++)//≠p∫‚º–∞O∂q
    {
        for(int j = 0 ;j<dst.cols;j++)
        {
            sum[iarry[i][j]] ++;
        }
    }
    for(int i =0;i<n;i++)//øzøÔ
    {
        if(sum[i] > 75)
            sum[i] =0;
        if(sum[i]< 25)
            sum[i] = 0;
    }
    for(int i = 0;i< dst.rows;i++) // ≤M∞£øzøÔµ≤™G
    {
        for(int j = 0 ;j<dst.cols;j++)
        {
            if(sum[iarry[i][j]] == 0 )
            {
                iarry[i][j] = 0;
                dst.at<uchar>(i,j) = 255;
            }
            else
                dst.at<uchar>(i,j) = 0;
        }
    }
    //imshow("03",dst);
    
    Canny(dst,dst,0,50,5);  //´ÿ•ﬂΩ¸π¯Canny
    //imshow("canny",dst);
    
//    vector<vector<Point>> approx;  //¥Mß‰Ω¸π¯
//    findContours(dst,approx,CV_RETR_LIST,CV_CHAIN_APPROX_SIMPLE);
    
    int xx = 0;
    //for(size_t i = 0;i<approx.size()-1;i++)  //≠p∫‚•≠ß°Æyº–
    //{
    //	Point bb = approx[i][0];
    //	xx += bb.y;
    //int xx = 0;
    //for(size_t i = 0;i<approx.size()-1;i++)  //≠p∫‚•≠ß°Æyº–
    //{
    //	Point bb = approx[i][0];
    //	xx += bb.x;
    //}
    //   xx /= approx.size();
    //for(size_t i = 1;i<approx.size()-1;i++)
    //{
    //	Point bb = approx[i][0];
    //	if(abs(approx[i][0].x - approx[i-1][0].x) >5 && abs(xx - bb.y) < 10 )  //±¯•Û§@ §£≠n≠´Ω∆ ±¯•Û§G ¬˜º∆¶r§”ª∑
    //	{
    //		Mat roi = src_gray(Rect(bb.x-5,bb.y-5,12,20));
    //	if(abs(approx[i][0].x - approx[i-1][0].x) >10 && abs(xx - bb.x) < 60 )  //±¯•Û§@ §£≠n≠´Ω∆ ±¯•Û§G ¬˜º∆¶r§”ª∑
    //	{
    //		Mat roi = src_gray(Rect(bb.x-5,bb.y-5,25,40));
    //		char buffer[3];
    //		sprintf(buffer,"%d",i);
    //		imshow(buffer,roi);
    //	}
    //}
    
    namedWindow( "threshold", CV_WINDOW_AUTOSIZE );
    namedWindow( "02", CV_WINDOW_AUTOSIZE );
    namedWindow( "03", CV_WINDOW_AUTOSIZE );
    namedWindow( "canny", CV_WINDOW_AUTOSIZE );
    moveWindow( "threshold", 0, 0 + HEIGHT * 2 );
    moveWindow( "02", 0, 0 + HEIGHT * 4 );
    moveWindow( "03", 0, 0 + HEIGHT * 6 );
    moveWindow( "canny", 0, 0 + HEIGHT * 8 );
    
    result[0] = 0;  //0:成功 非0:失敗
    result[1] = 63; //0~9:辨識值 63:無法辨識
    result[2] = 63; //0~9:辨識值 63:無法辨識
    result[3] = 63; //0~9:辨識值 63:無法辨識
    result[4] = 63; //0~9:辨識值 63:無法辨識
    result[5] = 63; //0~9:辨識值 63:無法辨識
    
    return result;
};


unsigned char *ALDigitRecognize(unsigned char type, unsigned char *imageBuf) {
    Mat src_gray,dst,thres;
    static unsigned char result[6];
    Mat src = Mat(HEIGHT, WIDTH, CV_8UC3, imageBuf);
    
    cvtColor(src,src_gray,COLOR_BGR2GRAY);
    imshow("Grayimage",src_gray);
   
	pyrUp(src_gray,src_gray,Size(src.cols*2,src.rows*2));
	//pyrUp(src_gray,src_gray,Size(src_gray.cols*2,src_gray.rows*2));
    imshow("up",src_gray);
    
	dilate(src_gray,src_gray,Mat(),Point(-1,-1),1);
	erode(src_gray,src_gray,Mat(),Point(-1,-1),1);
    imshow("1",src_gray);

	//pyrDown(src_gray,src_gray,Size(src_gray.cols/2,src_gray.rows/2));
	pyrDown(src_gray,src_gray,Size(dst.cols/2,dst.rows/2));
    imshow("down",src_gray);

    int T =0;
    double Tmax;
    double Tmin;;
    minMaxIdx(src_gray,&Tmin,&Tmax);
    T = ( Tmax + Tmin ) / 2;
    
    printf("Brightness MIN, MAX: %f, %f\n", Tmin, Tmax);
    
    while(true)
    {
        
        int Tosum =0,Tusum =0; //osum∂WπLT•[¡` usum §p©ÛT•[¡`
        int on = 0,un =0;  //on∂WπLT™∫¡`º∆ un §p©ÛT™∫¡`º∆
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
        Tosum /=on;
        Tusum /=un;
        if((Tosum+Tusum) /2  != T)
            T = (Tosum+Tusum) /2;
        else
            break;
    }
    
    
    threshold(src_gray,dst,T,255,THRESH_BINARY);
    threshold(src_gray,thres,T,1,THRESH_BINARY);
    imshow("ALthreshold",dst);
    
    cv::Mat labelImg ;
    IcvprCcaByTwoPass(thres, labelImg) ;
    
    // show result
    cv::Mat grayImg ;
    labelImg *= 10 ;
    labelImg.convertTo(grayImg, CV_8UC1) ;
    cv::imshow("labelImg", grayImg) ;
    
    int counts = 0;
    map<int, ALRect>::iterator iter;
    for(iter = component.begin(); iter != component.end(); iter++) {
        int x = iter->second._ltx;
        int y = iter->second._lty;
        int width = iter->second._width;
        int height = iter->second._height;
        int count = iter->second._count;
        int cy = HEIGHT / 2;
        bool isShow = y <= cy && y + height >= cy && count >= 15 && count <= 150 && height >=10 && height < 20 ? true : false;
		char title[100] ;
        bool isShow = y <= cy && y + height >= cy && count >= 15 && count <= 100 ? true : false;
        char title[] = {};
        
        if(isShow) {
            sprintf(title, "component : %d", iter->first);

            cout << "component ltx, lty, width, height, count : " << iter->second._ltx << ", " << iter->second._lty << ", " << iter->second._width << ", " << iter->second._height << ", " << iter->second._count << endl;

            Mat roi = grayImg( Rect(iter->second._ltx,iter->second._lty,iter->second._width,iter->second._height) );
            
            imshow(title,roi);
            namedWindow( title, CV_WINDOW_AUTOSIZE );
            moveWindow( title, WIDTH * 4, 0 + HEIGHT * (counts++) );
        }
    }
    
    cv::Mat colorLabelImg ;
    IcvprLabelColor(labelImg, colorLabelImg) ;
    cv::imshow("colorImg", colorLabelImg) ;
    
    namedWindow( "ALthreshold", CV_WINDOW_AUTOSIZE );
    namedWindow( "labelImg", CV_WINDOW_AUTOSIZE );
    namedWindow( "colorImg", CV_WINDOW_AUTOSIZE );
    
    moveWindow( "ALthreshold", WIDTH * 2, 0 + HEIGHT * 0 );
    moveWindow( "labelImg", WIDTH * 2, 0 + HEIGHT * 2 );
    moveWindow( "colorImg", WIDTH * 2, 0 + HEIGHT * 4 );
    
    result[0] = 0;  //0:成功 非0:失敗
    result[1] = 63; //0~9:辨識值 63:無法辨識
    result[2] = 63; //0~9:辨識值 63:無法辨識
    result[3] = 63; //0~9:辨識值 63:無法辨識
    result[4] = 63; //0~9:辨識值 63:無法辨識
    result[5] = 63; //0~9:辨識值 63:無法辨識
    
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
