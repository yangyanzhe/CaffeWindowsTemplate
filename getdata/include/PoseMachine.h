#include <caffe\caffe.hpp>

#pragma once
#include <vector>
#include <memory>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include "Correlation.h"

using namespace caffe;
using std::vector;
using std::string;
using cv::Mat;

struct ModelParameter
{
	string caffemodel;
	string deployFile;
	string description;
	int boxsize;
	int padValue;
	int np;		// number of parts
	int sigma;
	int stage;
	vector<std::shared_ptr<vector<double>>> limbs;
	vector<string> part_str;
};

class PoseMachine
{
public:
	PoseMachine();
	~PoseMachine();

public:
	int* pad;
	int  octave;
	int  num_channels_;
	bool use_gpu;
	std::shared_ptr<Net<float> > net_;

	void initModelParam();
	void deleteParam();
	void calculateScales(vector<double>& scales, int height);
	void registerLayers();
	void wrapInputLayer(vector<Mat>* input_channels);
	void preprocess(Mat img, vector<Mat>* input_channels);
	void resizeIntoScaledImg(Mat& originImg);
	void outputMatrix(string name, Mat& matrix);
	Mat  padAround(Mat originImg, double scale);

	void getPredictLocation(float* data, int height, int width, MyPoint& p);
	string getStageEndLayer(int stageNum);

public:
	ModelParameter* param;
	void initModelParam(string mCaffemodel, string mDeployFile);
	void loadNet();
	vector<Mat> applyModel(string image);
	void getPrediction(string filename, int stage, MyPoint* predict);
};