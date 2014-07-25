#include<opencv2\opencv.hpp>
#include<opencv2\video\video.hpp>
#include<opencv2\highgui\highgui.hpp>

#include<iostream>

using namespace cv;
using namespace std;

void imageout(Mat*, int);
int Modeling(Mat*, Mat*, int);

int main(int argc, char* argv[])
{
	if (argc != 2)
	{
		cout << "Input Error!" << endl;
		exit(0);
	}
	VideoCapture vid(argv[1]);
	//namedWindow("Extraced Foreground");
	BackgroundSubtractorMOG2 mog;
	int numberOfFrames = vid.get(CV_CAP_PROP_FRAME_COUNT);
	int count = 0;
	Mat* frame = new Mat[numberOfFrames];
	Mat* foreground = new Mat[numberOfFrames];
	double alpha = 0;
	cout << "Input alpha: ";
	cin >> alpha;
	for (; count < numberOfFrames; count++)
	{
		vid.read(frame[count]);

		mog(frame[count], foreground[count], alpha);
		//medianBlur(foreground, foreground, 7);
		//gmg(frame[count], foreground, alpha);
		cout << "current position: " << count + 1 << endl;
		threshold(foreground[count], foreground[count], 250, 255, CV_THRESH_BINARY_INV);
		//adaptiveThreshold(foreground, foreground, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 25, 10);
		//imshow("Extraced Foreground", foreground);
		//waitKey(5);
		/*
		char filename[50];
		_itoa(count + 1, filename, 10);
		strcat(filename, ".jpg");
		imwrite(filename, foreground);
		imwrite(filename, frame);
		*/
	}
	//double* model = Modeling(frame, foreground, numberOfFrames);
	int total = Modeling(frame, foreground, numberOfFrames);
	cout << total << endl;
	return 0;
}
