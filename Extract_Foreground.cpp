#include"function.h"

Mat Mask(Mat*, Mat*, int);
void clarify(Mat);
Mat MaskOnImage(Mat, Mat);

int main(int argc, char* argv[])
{
	if (argc != 2)
	{
		cout << "Input Error!" << endl;
		exit(0);
	}
	VideoCapture vid(argv[1]);
	BackgroundSubtractorMOG2 mog;
	int numberOfFrames = vid.get(CV_CAP_PROP_FRAME_COUNT);
	int count = 0;
	Mat* frame = new Mat[numberOfFrames];
	Mat* foreground = new Mat[numberOfFrames];
	Mat* original = new Mat[numberOfFrames];
	double alpha = 0.005;
	for (; count < numberOfFrames; count++)
	{
		vid.read(frame[count]);
		original[count] = frame[count].clone();
		cvtColor(frame[count], frame[count], CV_RGB2Lab);
		vector<Mat> channels;
		split(frame[count], channels);
		Mat L = Mat::zeros(Size(frame[count].cols, frame[count].rows), CV_8UC1);
		channels.at(0) = L;
		merge(channels, frame[count]);

		mog(frame[count], foreground[count], alpha);
		threshold(foreground[count], foreground[count], 250, 255, CV_THRESH_BINARY_INV);
	}
	Mat M = Mask(original, foreground, numberOfFrames);
	M.convertTo(M, CV_8UC1);
	threshold(M, M, 0, 255, CV_THRESH_BINARY);
	clarify(M);
	GaussianBlur(M, M, Size(5, 5), 0, 0, 4);
	Mat to_progress = imread("to_progress.jpg");
	Mat to_progress_po(to_progress.rows, to_progress.cols, CV_8UC3);
	to_progress_po = MaskOnImage(to_progress, M);
	
	int angle = PalmDirection(M);
	cout << angle << endl;
	Mat ad_image, rot, Mask_rot;
	rotateImage(to_progress_po, ad_image, rot, angle * 180 / PI);
	rotateImage(M, Mask_rot, rot, angle * 180 / PI);
	//******************************
	Rect r(150, 330, 225, 225);
	//******************************
	Mat tp = ad_image(r);
	Mat palm(300, 300, CV_8UC1);
	cvtColor(tp, palm, CV_RGB2GRAY);
	cnn net;
	CnnSetup(net);
	ReadData(net);
	Mat palm_cnn(40, 40, palm.type());
	resize(palm, palm_cnn, Size(40, 40));
	Mat cnn_in, point;
	palm_cnn.convertTo(cnn_in, CV_64FC1);
	CnnTest(net, cnn_in);
	point = CNNout(net.output, palm.rows, palm.cols);
	Mat point_out;
	Mat roi = ExtractROIstd(palm, point, point_out);
	imshow("roi", roi);
	waitKey();
	return 0;
}

void clarify(Mat M)
{
	int H = M.rows;
	int W = M.cols;
	Mat BW = Mat::zeros(Size(W, H), CV_8UC1);
	for (int i = W / 10; i < W * 9 / 10; i++)
	{
		for (int j = H * 1 / 9; j < H * 8 / 9; j++)
		{
			BW.at<uchar>(Point(i, j)) = 1;
		}
	}
	M = M.mul(BW);
}

Mat MaskOnImage(Mat to_progress, Mat mask)
{
	Mat result(to_progress.rows, to_progress.cols, CV_8UC3);
	for (int x = 0; x < to_progress.cols; x++)
	{
		for (int y = 0; y < to_progress.rows; y++)
		{
			uchar v = mask.at<uchar>(Point(x, y));
			v == 255 ? mask.at<uchar>(Point(x, y)) = 1 : mask.at<uchar>(Point(x, y)) = 0;
		}
	}
	vector<Mat> channels;
	split(to_progress, channels);
	channels.at(0) = channels.at(0).mul(mask);
	channels.at(1) = channels.at(1).mul(mask);
	channels.at(2) = channels.at(2).mul(mask);
	merge(channels, result);
	return result;
}
