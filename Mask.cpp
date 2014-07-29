#include<opencv2\opencv.hpp>
#include<opencv2\highgui\highgui.hpp>

#include<iostream>
#include<vector>

using namespace cv;
using namespace std;

vector<vector<double>> block_searching(Mat, Mat, int);

void Mask(Mat* A, Mat* B, int numberOfFrames)
{
	Mat M = Mat::zeros(Size(A[0].cols, A[0].rows), CV_8U);
	vector<vector<double>> Colors(3, vector<double>(0));
	vector<vector<double>> Colors_t(3, vector<double>(0));
	int count = 0;
	for (; count < numberOfFrames; count++)
	{
		Colors = block_searching(Colors, A[count], B[count]);
		
		for (int k = 0; k < Colors[0].size(); k++)
		{
			Colors_t[0].push_back(Colors[0][k]);
			Colors_t[1].push_back(Colors[1][k]);
			Colors_t[2].push_back(Colors[2][k]);
		}
	}
	int total = Colors_t[0].size();
	Mat Data = Mat(total, 3, CV_8U);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < total; j++)
			Data.at<double>(Point(i,j)) = Colors_t[i][j];
	Mat to_progress = A[numberOfFrames - 1];
	
}

vector<vector<double>> block_searching(Mat original, Mat foreground)
{
	vector<vector<double>> Colors(3, vector<double>(0));
	int h = foreground.rows / 10;
	int w = foreground.cols / 10;
	int t1, t2;
	for (t1 = 0; t1 < h; t1++)
	{
		for (t2 = 0; t2 < w; t2++)
		{
			Rect r(t2 * 10, t1 * 10, 10, 10);
			Mat Sub = foreground(r);
			if (countNonZero(Sub) <= 10)
			{
				Mat Sub_C = original(r);
				Scalar M = mean(Sub_C);
				Colors[0].push_back(M.val[0]);
				Colors[1].push_back(M.val[1]);
				Colors[2].push_back(M.val[2]);
			}
		}
	}
	return Colors;
}
