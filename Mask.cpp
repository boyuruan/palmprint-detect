#include"function.h"

vector<vector<double>> block_searching(Mat, Mat);

Mat Mask(Mat* A, Mat* B, int numberOfFrames)
{
	Mat M = Mat::zeros(Size(A[0].cols, A[0].rows), CV_64FC1);
	vector<vector<double>> Colors(3, vector<double>(0));
	vector<vector<double>> Colors_t(3, vector<double>(0));
	int count = 0;
	for (; count < numberOfFrames; count++)
	{
		Colors = block_searching(A[count], B[count]);
		
		for (int k = 0; k < Colors[0].size(); k++)
		{
			Colors_t[0].push_back(Colors[0][k]);
			Colors_t[1].push_back(Colors[1][k]);
			Colors_t[2].push_back(Colors[2][k]);
		}
	}
	int total = Colors_t[0].size();
	Mat Data = Mat(total, 3, CV_64F);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < total; j++)
			Data.at<double>(Point(i,j)) = Colors_t[i][j];
	Mat to_progress = A[numberOfFrames - 1];
	imwrite("to_progress.jpg", to_progress);
	Mat cov;
	Mat MM;
	calcCovarMatrix(Data, cov, MM, CV_COVAR_ROWS|CV_COVAR_NORMAL);
	cov = cov / (double)(total - 1);
	int x, y;
	for (x = 0; x < to_progress.cols; x++)
	{
		for (y = 0; y < to_progress.rows; y++)
		{
			Vec3b indensity = to_progress.at<Vec3b>(Point(x, y));
			uchar to_b = indensity.val[0];
			uchar to_g = indensity.val[1];
			uchar to_r = indensity.val[2];
			Mat E = Mat(3, 1, CV_64FC1);
			E.at<double>(Point(0, 0)) = (double)to_b - MM.at<double>(Point(0, 0));
			E.at<double>(Point(0, 1)) = (double)to_g - MM.at<double>(Point(1, 0));
			E.at<double>(Point(0, 2)) = (double)to_r - MM.at<double>(Point(2, 0));
			Mat temp(1, 1, CV_64FC1);
			temp = E.t()*cov.inv()*E;
			Scalar ind = temp.at<double>(0, 0);
			M.at<double>(Point(x, y)) = exp(ind.val[0]);
		}
	}
	return M;
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
