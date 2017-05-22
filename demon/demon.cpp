#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using namespace std;

Mat src_gray,dst;
Mat erosion_dst, dilation_dst;
const int WIDTH = 160, HEIGHT = 70;

unsigned char DigitRecognize(unsigned char, unsigned char *);

int main(int argc,char** argv)
{

	//≈™®˙πœ§˘
	Mat image = imread(argv[1],CV_LOAD_IMAGE_COLOR);
	if(!image.data)
	{
		return -1;
	}
    
    DigitRecognize(0, image.data);
}

unsigned char DigitRecognize(unsigned char type, unsigned char *imageBuf) {
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
    imshow("threshold",dst);
    
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
        for(int i = 1;i< dst.rows;i++)   //§¿∏s
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
    imshow("02",dst);
    
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
    imshow("03",dst);
    
    Canny(dst,dst,0,50,5);  //´ÿ•ﬂΩ¸π¯Canny
    imshow("canny",dst);
    
    //vector<vector<Point>> approx;  //¥Mß‰Ω¸π¯
    //findContours(dst,approx,CV_RETR_LIST,CV_CHAIN_APPROX_SIMPLE);
    
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
    
    
    waitKey(0);
    return 0;
};
