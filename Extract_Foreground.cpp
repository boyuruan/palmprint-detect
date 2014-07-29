#include<opencv2\opencv.hpp>
#include<opencv2\video\video.hpp>
#include<opencv2\highgui\highgui.hpp>

#include<iostream>

using namespace cv;
using namespace std;

Mat Mask(Mat*, Mat*, int);

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
	double alpha = 0.003;
	for (; count < numberOfFrames; count++)
	{
		vid.read(frame[count]);
		original[count] = frame[count].clone();
		cvtColor(frame[count], frame[count], CV_RGB2Lab);
		vector<Mat> channels;
		split(frame[count], channels);
		Mat a = channels.at(1);
		Mat L = Mat::zeros(Size(a.cols, a.rows), CV_8U);
		channels.at(0) = L;
		merge(channels, frame[count]);

		mog(frame[count], foreground[count], alpha);
		threshold(foreground[count], foreground[count], 250, 255, CV_THRESH_BINARY_INV);
	}
	Mat M = Mask(original, foreground, numberOfFrames);
	imshow("Mask", M);
	waitKey();
	return 0;
}
