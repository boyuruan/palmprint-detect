#include<opencv2\opencv.hpp>
#include<opencv2\highgui\highgui.hpp>

#include<iostream>
#include<vector>

using namespace cv;
using namespace std;

void block_searching(vector<vector<double>>, Mat, Mat, int);

void Mask(Mat* A, Mat* B, int numberOfFrames)
{
	Mat M = Mat::zeros(Size(A[0].cols, A[0].rows), CV_8U);
	vector<vector<double>> Colors(3, vector<double>(0));
	
	int count = 0;
	for (; count < numberOfFrames; count++)
	{
		block_searching(Colors, A[count], B[count], numberOfFrames);
	}
	int total = Colors[0].size();
	Mat Data = Mat(total, 3, CV_8U);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < total; j++)
			Data.at<double>(Point(i,j)) = Colors[i][j];
	Mat to_progress = A[numberOfFrames - 1];
	delete A;
	delete B;
}

void block_searching(vector<vector<double>> Colors, Mat original, Mat foreground, int numberOfFrames)
{
	int h = foreground.rows / 20;
	int w = foreground.cols / 20;
	int t1, t2;
	for (t1 = 0; t1 < h; t1++)
	{
		for (t2 = 0; t2 < w; t2++)
		{
			Rect r(t2 * 20, t1 * 20, 20, 20);
			Mat Sub = foreground(r);
			if (countNonZero(Sub) <= 40)
			{
				Mat Sub_C = original(r);
				for (int i = 0; i < 20; i++)
				{
					for (int j = 0; j < 20; j++)
					{
						Vec3f indensity = Sub_C.at<Vec3f>(Point(i, j));
						Colors[0].push_back(indensity.val[0]);
						Colors[1].push_back(indensity.val[1]);
						Colors[2].push_back(indensity.val[2]);
					}
				}
			}
		}
	}
}
