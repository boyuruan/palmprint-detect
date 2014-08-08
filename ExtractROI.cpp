#include "function.h"

//////////提取ROI区域
//几种类型之间的转换
Mat Mat2Pt(CvMat* pointMat)
{
	Mat pointList(8, 1, CV_64FC1);
	for (int i = 0; i < 4; i++)
	{
		pointList.at<double>(2 * i) = cvmGet(pointMat, 0, i);
		pointList.at<double>(2 * i + 1) = cvmGet(pointMat, 1, i);
	}
	return pointList;
}
CvMat* Pt2Mat(Mat pointList)
{
	CvMat *pointMat = cvCreateMat(3, 4, CV_64FC1);
	for (int i = 0; i < 4; i++)
	{
		cvmSet(pointMat, 0, i, pointList.at<double>(2 * i));
		cvmSet(pointMat, 1, i, pointList.at<double>(2 * i + 1));
		cvmSet(pointMat, 2, i, 1.0);
	}
	double x = cvmGet(pointMat, 0, 0);
	double y = pointList.at<double>(0);
	return pointMat;
}
CvMat* Pt2CvMat(Mat pointList, int row, int col)
{
	CvMat *pointMat = cvCreateMat(3, 4, CV_64FC1);
	for (int i = 0; i < 4; i++)
	{
		cvmSet(pointMat, 0, i, pointList.at<double>(2 * i) - col / 2);
		cvmSet(pointMat, 1, i, pointList.at<double>(2 * i + 1) - row / 2);
		cvmSet(pointMat, 2, i, 1.0);
	}
	return pointMat;
}
Mat CvMat2Pt(CvMat *pointMat, int row, int col)
{
	Mat pointList(8, 1, CV_64FC1);
	for (int i = 0; i < 4; i++)
	{
		pointList.at<double>(2 * i) = cvmGet(pointMat, 0, i) / cvmGet(pointMat, 2, i) + col / 2;
		pointList.at<double>(2 * i + 1) = cvmGet(pointMat, 1, i) / cvmGet(pointMat, 2, i) + row / 2;
	}
	return pointList;
}

//逆时针旋转图像degree角度（原尺寸）
void rotateImage(Mat img, Mat &img_rotate, Mat &rot_mat, double angle)  //angle:角度
{
	// 旋转矩阵
	Point center = Point(img.cols / 2, img.rows / 2);
	double scale = 1;
	rot_mat = getRotationMatrix2D(center, angle, scale);
	warpAffine(img, img_rotate, rot_mat, img.size());
}

//图像仿射校正过程
void AdjustImage(Mat &trans, Mat &trans_inv, Mat point_std, Mat point_ori, Mat &point_out, int std_size[], int in_size[])
{
	//计算输入图像到标准图像的变换矩阵
	CvMat *std = Pt2CvMat(point_std, std_size[0], std_size[1]);
	CvMat *ori = Pt2CvMat(point_ori, in_size[0], in_size[1]);
	CvMat *std_inv = cvCreateMat(4, 3, CV_64FC1);
	CvMat *tr_inv = cvCreateMat(3, 3, CV_64FC1);

	CvMat *tr = cvCreateMat(3, 3, CV_64FC1);
	cvInvert(std, std_inv, CV_SVD);
	cvMatMul(ori, std_inv, tr);
	cvInvert(tr, tr_inv, CV_SVD);
	trans = Mat(tr, 1);
	trans_inv = Mat(tr_inv, 1);

	//计算输出图像中的特征点位置
	CvMat *pCM_out = cvCreateMat(3, 4, CV_64FC1);
	cvMatMul(tr_inv, ori, pCM_out);
	point_out = CvMat2Pt(pCM_out, in_size[0], in_size[1]);

}

//输入标准图像，则对标准图像进行旋转校正
Mat ExtractROIstd(Mat image_std, Mat point_std, Mat &point_out)
{
	Mat ad_image, rot_mat, roi;
	double dist, angle;
	int middle[2];

	//旋转校正
	angle = atan((point_std.at<double>(7) - point_std.at<double>(3)) / (point_std.at<double>(6) - point_std.at<double>(2)));
	rotateImage(image_std, ad_image, rot_mat, angle * 180 / PI);

	CvMat* rotm = Mat2CvMat(rot_mat);
	CvMat* pointMatin = Pt2Mat(point_std);
	CvMat* pointMatout = cvCreateMat(2, 4, CV_64FC1);
	cvMatMul(rotm, pointMatin, pointMatout);
	point_out = Mat2Pt(pointMatout);

	//提取ROI区域
	middle[0] = (int)((point_out.at<double>(2) + point_out.at<double>(6)) / 2);
	middle[1] = (int)((point_out.at<double>(3) + point_out.at<double>(7)) / 2);
	dist = sqrt((point_out.at<double>(2) - point_out.at<double>(6))*(point_out.at<double>(2) - point_out.at<double>(6))
		+ (point_out.at<double>(3) - point_out.at<double>(7))*(point_out.at<double>(3) - point_out.at<double>(7)));
	int t = dist / 4;
	Mat temp = ad_image(Rect(middle[0] - t * 2, middle[1] + t, t * 4, t * 4));
	temp.copyTo(roi);

	//将标准图尺寸和特征点位置写入文件
	ofstream fout("Std.txt");
	fout << image_std.rows << " " << image_std.cols << endl;
	for (int i = 0; i < 8; i++)
		fout << point_std.at<double>(i) << " ";

	cvReleaseMat(&pointMatin);
	cvReleaseMat(&pointMatout);
	cvReleaseMat(&rotm);

	return  roi;
}

//只存在输入图像，则对输入图像进行仿射校正
Mat ExtractROIin(Mat image_in, Mat point_ori, Mat &point_out)
{
	Mat point_std(8, 1, CV_64FC1);
	int std_size[2];

	//从文件中读入标准图尺寸和特征点位置
	ifstream fin("Std.txt");
	fin >> std_size[0];
	fin >> std_size[1];
	for (int i = 0; i < 4; i++)
	{
		fin >> point_std.at<double>(2 * i);
		fin >> point_std.at<double>(2 * i + 1);
	}

	Mat ad_image(std_size[0], std_size[1], CV_8UC1);
	Mat roi;
	double dist;
	int middle[2];
	int in_size[2] = { image_in.rows, image_in.cols };

	//图像校正
	Mat trans(3, 3, CV_64FC1), trans_inv(3, 3, CV_64FC1);
	AdjustImage(trans, trans_inv, point_std, point_ori, point_out, std_size, in_size);

	Mat trans_mat = trans_inv(Rect(0, 0, 3, 2));
	for (int i = 0; i < 2; i++)
		trans_mat.at<double>(i, 2) = trans_mat.at<double>(i, 2) - trans_mat.at<double>(i, 0)*in_size[1 - i] / 2
		- trans_mat.at<double>(i, 1)*in_size[i] / 2 + std_size[1 - i] / 2;

	warpAffine(image_in, ad_image, trans_mat, ad_image.size());


	//提取ROI区域
	middle[0] = (int)((point_out.at<double>(2) + point_out.at<double>(6)) / 2);
	middle[1] = (int)((point_out.at<double>(3) + point_out.at<double>(7)) / 2);
	dist = sqrt((point_out.at<double>(2) - point_out.at<double>(6))*(point_out.at<double>(2) - point_out.at<double>(6))
		+ (point_out.at<double>(3) - point_out.at<double>(7))*(point_out.at<double>(3) - point_out.at<double>(7)));
	int t = dist / 4;
	Mat temp = ad_image(Rect(middle[0] - t * 2, middle[1] + t, t * 4, t * 4));
	temp.copyTo(roi);

	return roi;
}
