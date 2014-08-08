#include "function.h"

//////////利用CNN检测特征点

CvMat* Mat2CvMat(Mat mat)
{
	CvMat* cvmat = cvCreateMat(mat.rows,mat.cols,CV_64FC1);
	for(int i = 0; i < mat.rows; i++)
		for(int j = 0; j < mat.cols; j++)
			cvmSet(cvmat,i,j,mat.at<double>(i,j));

	return cvmat;
}

Mat sigm(Mat in)
{
	Mat out(in.size(),in.type());
	for(int i = 0; i < in.rows; i++)
		for(int j = 0; j < in.cols; j++)
			out.at<double>(i,j) = 1/(1+exp(-in.at<double>(i,j)));

	return out;
}

//实现卷积操作
void conv(Mat in, Mat mask, Mat &out)
{
	for(int i = 0; i < out.rows; i++)
		for(int j = 0; j < out.cols; j++)
		{
			double d = 0;
			for(int k = 0; k < mask.rows; k++)
				for(int l = 0; l < mask.cols; l++)
					d += in.at<double>(i+k,j+l) * mask.at<double>(mask.rows-1-k,mask.cols-1-l);
			out.at<double>(i,j) = d;
		}
}

//实现下采样操作
void sampling(Mat in, int scale, Mat &out)
{
	for(int i = 0; i < out.rows; i++)
		for(int j = 0; j < out.cols; j++)
		{
			double d = 0;
			for(int k = 0; k < scale; k++)
				for(int l = 0; l < scale; l++)
					d += in.at<double>(i*scale+k,j*scale+l);
			out.at<double>(i,j) = d/(scale*scale);
		}
}

//读入文件中的数据，构建CNN
void ReadData(cnn &net)
{
	ifstream fin("Data.txt");
	string name;
	int number;
	double data;

	while(1)
	{
		fin >> name;
		if(name == "cl1outputmaps")
			{   fin >> number;
				net.cl[0].outputmaps = number;   }
		if(name == "cl1kernelsize")
			{   fin >> number;
				net.cl[0].kernelsize = number;   }
		if(name == "cl1kernel")
			{   for(int i = 0; i < net.cl[0].outputmaps; i ++)
					for(int j = 0; j < net.cl[0].kernelsize; j ++)
						for(int k = 0; k < net.cl[0].kernelsize; k ++)
						{
							fin >> data;
							net.cl[0].kernel[0][i].at<double>(j,k) = data;
						}
			}
		if(name == "cl1b")
			{   for(int i = 0; i < net.cl[0].outputmaps; i ++)
			    {
					fin >> data;
					net.cl[0].b[i] = data;
			    }
			}
		if(name == "cl2outputmaps")
			{   fin >> number;
				net.cl[1].outputmaps = number;   }
		if(name == "cl2kernelsize")
			{   fin >> number;
				net.cl[1].kernelsize = number;   }
		if(name == "cl2kernel")
			{   for(int i = 0; i < net.cl[0].outputmaps; i ++)
					for(int j = 0; j < net.cl[1].outputmaps; j ++)
						for(int k = 0; k < net.cl[1].kernelsize; k ++)
							for(int l = 0; l < net.cl[1].kernelsize; l ++)
							{
								fin >> data;
								net.cl[1].kernel[i][j].at<double>(k,l) = data;
							}
			}
		if(name == "cl2b")
			{   for(int i = 0; i < net.cl[1].outputmaps; i ++)
			    {
					fin >> data;
					net.cl[1].b[i] = data;
			    }
			}
		if(name == "ffW")
			{   for(int i = 0; i < 8; i ++)
					for(int j = 0; j < 980; j ++)
					{
						fin >> data;
						cvmSet(net.ffW,i,j,data);
					}
			}
		if(name == "ffb")
			{   for(int i = 0; i < 8; i ++)
				{
					fin >> data;
					net.ffb.at<double>(i) = data;
				}
			}
		if(name == "end")
			{  break;  }
	}

}

//初始化CNN
void CnnSetup(cnn &net)
{
	int mapsize[2] = {40,40};

	//卷积层1
	net.cl[0].outputmaps = 10;
	net.cl[0].kernelsize = 5;
	net.cl[0].b = new double [net.cl[0].outputmaps];
	net.cl[0].map = new Mat [net.cl[0].outputmaps];
	net.cl[0].kernel = new Mat *[1];
	mapsize[0] = mapsize[0] - net.cl[0].kernelsize + 1;
	mapsize[1] = mapsize[1] - net.cl[0].kernelsize + 1;

	net.cl[0].kernel[0] = new Mat [net.cl[0].outputmaps];
	for(int j = 0; j < net.cl[0].outputmaps; j++)
	{
		net.cl[0].b[j] = 0;
		net.cl[0].map[j] = Mat(mapsize[0],mapsize[1],CV_64FC1);
		net.cl[0].kernel[0][j] = Mat(net.cl[0].kernelsize,net.cl[0].kernelsize,CV_64FC1);
	}

	//下采样层1
	net.sl[0].map = new Mat [net.cl[0].outputmaps];
	net.sl[0].scale = 2;
	mapsize[0] = mapsize[0]/net.sl[0].scale;
	mapsize[1] = mapsize[1]/net.sl[0].scale;

	for(int j = 0; j < net.cl[0].outputmaps; j++)
		net.sl[0].map[j] = Mat(mapsize[0],mapsize[1],CV_64FC1);

	//卷积层2
	net.cl[1].outputmaps = 20;
	net.cl[1].kernelsize = 5;
	net.cl[1].b = new double [net.cl[1].outputmaps];
	net.cl[1].map = new Mat [net.cl[1].outputmaps];
	net.cl[1].kernel = new Mat *[net.cl[0].outputmaps];
	mapsize[0] = mapsize[0] - net.cl[1].kernelsize + 1;
	mapsize[1] = mapsize[1] - net.cl[1].kernelsize + 1;

	for(int i = 0; i < net.cl[0].outputmaps; i++)
	{
		net.cl[1].kernel[i] = new Mat [net.cl[1].outputmaps];
		for(int j = 0; j < net.cl[1].outputmaps; j++)
			net.cl[1].kernel[i][j] = Mat(net.cl[1].kernelsize,net.cl[1].kernelsize,CV_64FC1);
	}
	for(int j = 0; j < net.cl[1].outputmaps; j++)
	{
		net.cl[1].b[j] = 0;
		net.cl[1].map[j] = Mat(mapsize[0],mapsize[1],CV_64FC1);
	}

	//下采样层2
	net.sl[1].map = new Mat [net.cl[1].outputmaps];
	net.sl[1].scale = 2;
	mapsize[0] = mapsize[0]/net.sl[1].scale;
	mapsize[1] = mapsize[1]/net.sl[1].scale;

	for(int j = 0; j < net.cl[1].outputmaps; j++)
		net.sl[1].map[j] = Mat(mapsize[0],mapsize[1],CV_64FC1);

	net.ffb = Mat(8,1,CV_64FC1);
	net.ffW = cvCreateMat(8,net.cl[1].outputmaps * mapsize[0] * mapsize[1], CV_64FC1);
	net.fv = Mat(net.cl[1].outputmaps * mapsize[0] * mapsize[1], 1, CV_64FC1);
	net.output = Mat(8,1,CV_64FC1);
}

//CNN的测试程序
void CnnTest(cnn &net, Mat in)
{
	//卷积层1
	for(int j = 0; j < net.cl[0].outputmaps; j++)
	{
		Mat z(in.rows-net.cl[0].kernelsize+1,in.cols-net.cl[0].kernelsize+1,CV_64FC1);
		conv(in, net.cl[0].kernel[0][j], z);
		z += net.cl[0].b[j];
		net.cl[0].map[j] = sigm(z);
	}

	//下采样层1
	for(int j = 0; j < net.cl[0].outputmaps; j++)
		sampling(net.cl[0].map[j], net.sl[0].scale, net.sl[0].map[j]);

	//卷积层2
	for(int j = 0; j < net.cl[1].outputmaps; j++)
	{
		Mat z, t;
		z = Mat::zeros(net.sl[0].map[0].rows-net.cl[1].kernelsize+1,net.sl[0].map[0].cols-net.cl[1].kernelsize+1,CV_64FC1);
		t = Mat(z.size(),CV_64FC1);
		for(int i = 0; i < net.cl[0].outputmaps; i++)
		{
			conv(net.sl[0].map[i], net.cl[1].kernel[i][j], t);
			add(z,t,z);
		}
		z += net.cl[1].b[j];
		net.cl[1].map[j] = sigm(z);
	}

	//下采样层2
	for(int j = 0; j < net.cl[1].outputmaps; j++)
		sampling(net.cl[1].map[j], net.sl[1].scale, net.sl[1].map[j]);

    //输出层
    for(int j = 0; j < net.cl[1].outputmaps; j ++)
	{
		int total = net.sl[1].map[j].rows * net.sl[1].map[j].cols;
        Mat z = net.sl[1].map[j].t();
		z = z.reshape(0, total);
		z.copyTo(net.fv(Rect(0,j*7*7,1,7*7)));
	}

	CvMat* ffv = Mat2CvMat(net.fv);
	CvMat* c1 = cvCreateMat(8,1,CV_64FC1);
	cvMatMul(net.ffW,ffv,c1);

	Mat c2 = Mat(c1,1);
	add(c2,net.ffb,c2);
    
	net.output = sigm(c2);

}

//将输出的0~1的数转化为图片上的坐标点
Mat CNNout(Mat pointin, int row, int col)
{
	Mat pointout(pointin.rows,pointin.cols,CV_64FC1);
	for(int i = 0; i < 4; i++)
	{
		pointout.at<double>(2*i) = pointin.at<double>(2*i) * col;
		pointout.at<double>(2*i+1) = pointin.at<double>(2*i+1) * row;
	}

	return pointout;
}
