#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using namespace std;

Mat src, src_gray,dst;
Mat erosion_dst, dilation_dst;

int main(int argc,char** argv)
{
	Mat canny_output;

	//讀取圖片
	src = imread("D:\\ROC\\demon\\images\\sample.bmp",CV_LOAD_IMAGE_COLOR);
	if(!src.data)
	{
		return -1;
	}
	
	//轉灰階圖
    cvtColor(src,src_gray,COLOR_BGR2GRAY);
	imshow("Grayimage",src_gray);
	
	//放大
	pyrUp(src_gray,src_gray,Size(src.cols*2,src.rows*2));
	imshow("up",src_gray);

	//高斯濾波
	blur(src_gray,dst,Size(3,3));
	imshow("blur",dst);
    

	//二值化
	threshold(dst,dst,140,255,THRESH_BINARY);
	//imshow("threshold",dst);

	//膨脹	
	dilate(dst,dst,Mat(),Point(-1,-1),4);
	imshow("1",dst);
	
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
	
	for(int i = 1;i< dst.rows;i++)   //分群
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
					iarry[i][j] = iarry[i-1][j];    //N=U
				else if(iarry[i][j-1] !=0 && iarry[i-1][j] !=0 && iarry[i][j-1] != iarry[i-1][j])
					iarry[i][j] = iarry[i-1][j];    //N=U
			}			
		}
	}

 
	//for(int i = 0;i< dst.rows;i++)  //測試圖
	//{
	//	for(int j = 0 ;j<dst.cols;j++)
	//	{
	//		if(dst.at<uchar>(i,j) == 255)
	//		{
	//			dst.at<uchar>(i,j) = iarry[i][j]*100;
	//		 
	//		}
	//		else
	//			dst.at<uchar>(i,j) = 255;
	//	}
	//}
	//imshow("02",dst);

	int* sum = new int[n];
	for(int i =0;i<n;i++) //初始化
	{
		sum[i] =0;
	}
    for(int i = 0;i< dst.rows;i++)//計算標記量
	{
		for(int j = 0 ;j<dst.cols;j++)
		{
			sum[iarry[i][j]] ++;
		}
	}
	for(int i =0;i<n;i++)//篩選
	{
		if(sum[i] > 850)
			sum[i] =0;
		if(sum[i]< 320)
			sum[i] = 0;
	}
	for(int i = 0;i< dst.rows;i++) // 清除篩選結果
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

	Canny(dst,dst,0,50,5);  //建立輪廓Canny
	imshow("canny",dst);
	
	vector<vector<Point>> approx;  //尋找輪廓
	findContours(dst,approx,CV_RETR_LIST,CV_CHAIN_APPROX_SIMPLE);

	int xx = 0;
	for(size_t i = 0;i<approx.size()-1;i++)  //計算平均座標
	{
		Point bb = approx[i][0];
		xx += bb.x;
	}
    xx /= approx.size();
	for(size_t i = 1;i<approx.size()-1;i++)
	{
		Point bb = approx[i][0];
		if(abs(approx[i][0].x - approx[i-1][0].x) >10 && abs(xx - bb.x) < 60 )  //條件一 不要重複 條件二 離數字太遠
		{
			Mat roi = src_gray(Rect(bb.x-5,bb.y-5,25,40));
			char buffer[3];
			sprintf(buffer,"%d",i);
			imshow(buffer,roi);
		}
	}


	waitKey(0);
	return 0;
}

