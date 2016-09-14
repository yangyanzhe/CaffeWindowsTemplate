#include "ActivationDifference.h"
#include <iostream>
#include <string>
#include <fstream>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace cv;
using namespace std;

ActivationDifference::ActivationDifference(string modelPath, string prototxtPath)
{
	pm1 = new PoseMachine();
	pm2 = new PoseMachine();

	pm1->initModelParam(modelPath, prototxtPath);
	pm1->loadNet();

	pm2->initModelParam(modelPath, prototxtPath);
	pm2->loadNet();
}

ActivationDifference::~ActivationDifference()
{
	delete pm1;
	delete pm2;
}



int ActivationDifference::getLayerId(string name)
{
	int layerId = 0;
	auto layer_names = pm1->net_->layer_names();
	for (int i = 0; i<layer_names.size(); i++) {
		if (layer_names[i].compare(name) == 0) {
			layerId = i;
			break;
		}
	}
	return layerId;
}

void ActivationDifference::getLayerList()
{
	
	layerList.push_back(getLayerId("relu1_stage1"));
	layerList.push_back(getLayerId("relu2_stage1"));
	layerList.push_back(getLayerId("relu3_stage1"));
	layerList.push_back(getLayerId("relu4_stage1"));
	layerList.push_back(getLayerId("relu5_stage1"));
	layerList.push_back(getLayerId("relu6_stage1"));
	layerList.push_back(getLayerId("conv7_stage1"));

	for (int i = 2; i <= 6; i++)
	{
		layerList.push_back(getLayerId("Mconv1_stage" + to_string(i)));
		layerList.push_back(getLayerId("Mconv2_stage" + to_string(i)));
		layerList.push_back(getLayerId("Mconv3_stage" + to_string(i)));
		layerList.push_back(getLayerId("Mconv4_stage" + to_string(i)));
		layerList.push_back(getLayerId("Mconv5_stage" + to_string(i)));
	}
}

void ActivationDifference::calculateActivationDifference(string imgPath1, string imgPath2, string description)
{
	getLayerList();
	cout << "layer number = " << layerList.size() << endl;
	for (int i = 0; i < layerList.size(); i++)
	{
		cout << i << ": " << layerList[i] << endl;
	}

	// read image
	cout << "read image " << imgPath1 << endl;
	Mat img1 = imread(imgPath1);
	if (img1.empty())
	{
		std::cout << "cannot read image from file: " << imgPath1 << std::endl;
		return;
	}
	cout << "read image " << imgPath2 << endl;
	Mat img2 = imread(imgPath2);
	if (img2.empty())
	{
		std::cout << "cannot read image from file: " << imgPath2 << std::endl;
		return;
	}

	// preprocess
	vector<double> scales1;
	pm1->calculateScales(scales1, img1.rows);
	Mat sImg1 = pm1->padAround(img1, scales1[pm1->octave]);
	vector<cv::Mat> input_channels1;
	pm1->wrapInputLayer(&input_channels1);
	pm1->preprocess(sImg1, &input_channels1);
	auto layer_names = pm1->net_->layer_names();

	vector<double> scales2;
	pm2->calculateScales(scales2, img2.rows);
	Mat sImg2 = pm2->padAround(img2, scales2[pm2->octave]);
	vector<cv::Mat> input_channels2;
	pm2->wrapInputLayer(&input_channels2);
	pm2->preprocess(sImg2, &input_channels2);

	cout << "images have been processed. " << endl;
	
	int start = 0;
	int end = -1;
	for (int stage = 1; stage <= 6; stage++)
	{
		start = end + 1;
		if (stage == 1)
			end += 7;
		else
			end += 5;

		for (int i = start; i <= end; i++)
		{
			int layerId = layerList[i];
			// forward net
			pm1->net_->ForwardTo(layerId);
			pm2->net_->ForwardTo(layerId);

			// output
			int width = 0, height = 0, channel = 0;
			int area = 0;

			int  blobId1 = pm1->net_->top_ids(layerId)[0];
			auto blob1 = pm1->net_->blobs()[blobId1];
			const float* output_data1 = blob1->cpu_data();

			int  blobId2 = pm2->net_->top_ids(layerId)[0];
			auto blob2 = pm2->net_->blobs()[blobId2];
			const float* output_data2 = blob2->cpu_data();

			width = blob1->width();
			height = blob1->height();
			channel = blob1->channels();
			area = width * height;
			for (int j = 0; j < channel; j++)
			{
				float* temp1 = new float[area];
				memcpy(temp1, output_data1, sizeof(float)*area);
				Mat activation1(height, width, CV_32FC1, temp1);
				//Mat actNorm1(height, width, CV_32FC1);
				//normalizeMatrix(activation1, actNorm1);
				
				float* temp2 = new float[area];
				memset(temp2, 0, sizeof(float)*area);
				memcpy(temp2, output_data2, sizeof(float)*area);
				Mat activation2(height, width, CV_32FC1, temp2);
				//Mat actNorm2(height, width, CV_32FC1);
				//normalizeMatrix(activation2, actNorm2);
				
				if (channel == 15)
				{
					int boxsize = pm1->param->boxsize;
					resize(activation1, activation1, Size(boxsize, boxsize));
					pm1->resizeIntoScaledImg(activation1);
					resize(activation1, activation1, Size(img1.cols, img1.rows));

					resize(activation2, activation2, Size(boxsize, boxsize));
					pm2->resizeIntoScaledImg(activation2);
					resize(activation2, activation2, Size(img2.cols, img2.rows));
				}

				string filename1 = "result" + description + "/activation1/" + layer_names[layerId] + "_filter" + to_string(j) + ".jpg";
				imwrite(filename1, generateHeatMap(activation1));
				string filename2 = "result" + description + "/activation2/" + layer_names[layerId] + "_filter" + to_string(j) + ".jpg";
				imwrite(filename2, generateHeatMap(activation2));


				string filename = "result" + description + "/stage" + to_string(stage) + "/" + layer_names[layerId] + "_filter" + to_string(j) + ".jpg";
				imwrite(filename, generateDifferenceMap(activation1, activation2));
				cout << "output to " << filename << endl;

				//Mat result(height, width, CV_8UC1);
				//absdiff(actNorm1, actNorm2, result);

				//if (judgeToPrint(result))
				//{
					// output difference
				//	string filename = "result/stage" + to_string(stage) + "/" + layer_names[layerId] + "_filter" + to_string(j) + ".jpg";
				//	imwrite(filename, result);
				//	cout << "output to " << filename << endl;
				//}

				output_data1 += area;
				output_data2 += area;
				if (temp1 != NULL)
				{
					delete[] temp1;
					temp1 = NULL;
				}
				if (temp2 != NULL)
				{
					delete[] temp2;
					temp2 = NULL;
				}

			} // end iteration on channels
		} // end iteration on layers
	} // end iteration on stage

	
	vector<double>().swap(scales1); 
	vector<double>().swap(scales2);
}

Mat ActivationDifference::generateHeatMap(Mat& activation)
{
	int width = activation.cols;
	int height = activation.rows;
	Mat result(height, width, CV_8UC3);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			float value = activation.at<float>(i, j);
			float r = 0, g = 0, b = 0;
			float h = 0;
			float s = abs(value);
			float v = 1;

			HSVtoRGB(r, g, b, h, s, v);
			result.at<Vec3b>(i, j)[2] = static_cast<uint8_t>(r * 255);
			result.at<Vec3b>(i, j)[1] = static_cast<uint8_t>(g * 255);
			result.at<Vec3b>(i, j)[0] = static_cast<uint8_t>(b * 255);
		}
	}

	return result;
}

Mat ActivationDifference::generateDifferenceMap(Mat& map1, Mat& map2)
{
	int width = map1.cols;
	int height = map1.rows;
	Mat result(height, width, CV_8UC3);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			float value = map1.at<float>(i, j) - map2.at<float>(i, j);
			float r = 0, g = 0, b = 0;
			float h = 0, s = 0, v = 1;
			if (value > 0)	// red
			{
				h = 0;
				s = value;
				v = 1;
			}
			else            // blue
			{
				h = 240;
				s = abs(value);
				v = 1;
			}
			HSVtoRGB(r, g, b, h, s, v);
			result.at<Vec3b>(i, j)[2] = static_cast<uint8_t>(r * 255);
			result.at<Vec3b>(i, j)[1] = static_cast<uint8_t>(g * 255);
			result.at<Vec3b>(i, j)[0] = static_cast<uint8_t>(b * 255);
		}
	}

	return result;
}

bool ActivationDifference::judgeToPrint(Mat& result)
{
	int width = result.cols;
	int height = result.rows;
	bool flag = false;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int value = result.at<uchar>(i, j);
			if (value > 10)
			{
				flag = true;
			}
		}
	}

	return flag;
}

void ActivationDifference::normalizeMatrix(Mat& input, Mat& output)
{
	int width = input.cols;
	int height = input.rows;
	float max = 0;
	float min = 0;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			float value = input.at<float>(i, j);
			min = __min(min, value);
			max = __max(max, value);
		}
	}

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			float value = input.at<float>(i, j);
			output.at<float>(i, j) = (value - min) / (max - min);
		}
	}
}

void ActivationDifference::HSVtoRGB(float& fR, float& fG, float& fB, float& fH, float& fS, float& fV) {
	float hh, p, q, t, ff;
	int  i;

	if (fS <= 0.0) {       // < is bogus, just shuts up warnings
		fR = fV;
		fG = fV;
		fB = fV;
	}
	hh = fH;
	if (hh >= 360.0) hh = 0.0;
	hh /= 60.0;
	i = static_cast<int>(hh);
	ff = hh - i;
	p = fV * (1.0 - fS);
	q = fV * (1.0 - (fS * ff));
	t = fV * (1.0 - (fS * (1.0 - ff)));

	switch (i) {
	case 0:
		fR = fV;
		fG = t;
		fB = p;
		break;
	case 1:
		fR = q;
		fG = fV;
		fB = p;
		break;
	case 2:
		fR = p;
		fG = fV;
		fB = t;
		break;

	case 3:
		fR = p;
		fG = q;
		fB = fV;
		break;
	case 4:
		fR = t;
		fG = p;
		fB = fV;
		break;
	case 5:
	default:
		fR = fV;
		fG = p;
		fB = q;
		break;
	}
}

void ActivationDifference::run()
{
	for (int i = 0; i < pairList.size(); i++)
		calculateActivationDifference(pairList[i].imgPath1, pairList[i].imgPath2, pairList[i].description);
}