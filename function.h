#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>

#include<opencv2\opencv.hpp>
#include<opencv2\highgui\highgui.hpp>

using namespace cv;
using namespace std;

#define PI 3.1415926

struct convlayer  //卷积层
{
	int outputmaps;
	int kernelsize;
	double *b;
	Mat *map;
	Mat **kernel;
};
struct samplelayer  //下采样层
{
	int scale;
	Mat *map;
};
struct cnn  //CNN网络
{
	convlayer cl[2];
	samplelayer sl[2];
	CvMat *ffW;
	Mat fv, ffb;
	Mat output;
};

//一些类型之间的转换
Mat Mat2Pt(CvMat* pointMat);
CvMat* Pt2Mat(Mat pointList);
Mat CvMat2Pt(CvMat *pointMat, int row, int col);
CvMat* Pt2CvMat(Mat pointList, int row, int col);
CvMat* Mat2CvMat(Mat mat);

//手掌方向检测和图像旋转
int PalmDirection(Mat image);
void rotateImage(Mat img, Mat &img_rotate, Mat &rot_mat, double angle);

//CNN
Mat sigm(Mat in);
void conv(Mat in, Mat mask, Mat &out);
void sampling(Mat in, int scale, Mat &out);
void ReadData(cnn &net);
void CnnSetup(cnn &net);
void CnnTest(cnn &net, Mat in);
Mat CNNout(Mat pointin, int row, int col);

//图像校正和提取ROI
void AdjustImage(Mat &trans, Mat &trans_inv, Mat point_std, Mat point_ori, Mat &point_out, int std_size[], int in_size[]);
Mat ExtractROIstd(Mat image_std, Mat point_std, Mat &point_out);
Mat ExtractROIin(Mat image_in, Mat point_ori, Mat &point_out);

//寻找特征点、图像校正、提取ROI
Mat main_preprocessing(Mat img_gray, bool stdORin);
