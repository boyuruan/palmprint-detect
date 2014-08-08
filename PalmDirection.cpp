#include "function.h"

//////////利用hough变换检测手掌方向
int PalmDirection(Mat image)  //输入二值化图像
{
	//霍夫变换，每一对极坐标参数(rho,theta)对应一条直线，保存到lines
	vector<cv::Vec2f> lines;
	HoughLines(image, lines, 1, PI / 180, 80);

	//统计直线角度的规律
	vector<cv::Vec2f>::const_iterator it = lines.begin();
	Mat angle = Mat::zeros(18, 1, CV_8UC1);    //将180度角分为18份，每一份对应10度角
	while (it != lines.end())
	{
		float theta = (*it)[1];
		angle.at<uchar>((int)(theta / PI * 18), 0)++;
		++it;
	}

	Point maxloc;
	minMaxLoc(angle, NULL, NULL, NULL, &maxloc);
	int PalmAngle = maxloc.y * 10 + 5;
	return PalmAngle;
}
