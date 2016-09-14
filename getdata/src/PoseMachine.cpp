#include <fstream>
#include <iostream>
#include <cmath>
#include <ctime>
#include <string>
#include "PoseMachine.h"
#include "caffe\caffe.hpp"
#include "caffe\proto\caffe.pb.h"
#include "caffe\layer_factory.hpp"
#include "caffe\layer.hpp"
#include "caffe\layers\conv_layer.hpp"
#include "caffe\layers\pooling_layer.hpp"
#include "caffe\layers\relu_layer.hpp"
#include "caffe\layers\slice_layer.hpp"
#include "caffe\layers\split_layer.hpp"
#include "caffe\layers\concat_layer.hpp"
#include "caffe\layers\input_layer.hpp"
#include <opencv2\imgproc\imgproc.hpp>
#include "Correlation.h"

using namespace std;
using namespace cv;
using namespace caffe;

PoseMachine::PoseMachine()
{
	param = NULL;
	octave = 6;
	use_gpu = false;
	pad = new int[4];
	memset(pad, 0, sizeof(int) * 4);
	initModelParam();
}

PoseMachine::~PoseMachine()
{
	delete[]pad;
	deleteParam();
	net_ = nullptr;
}

void PoseMachine::deleteParam()
{
	if (param != NULL)
	{
		param->caffemodel.clear();
		param->deployFile.clear();
		param->description.clear();
		param->limbs.clear();
		param = NULL;
	}
}

void PoseMachine::initModelParam()
{
	if (param != NULL)
		deleteParam();

	param = new ModelParameter;
	param->caffemodel = "..\\models\\PoseModel\\pose_iter_985000_addLEEDS.caffemodel";
	param->deployFile = "..\\models\\PoseModel\\pose_deploy_centerMap.prototxt";
	param->description, "MPII+LSP 6-stage CPM";
	param->boxsize = 368;
	param->padValue = 128;
	param->np = 14;
	param->sigma = 21;
	param->stage = 6;
	for (int i = 0; i < 9; i++)
	{
		std::shared_ptr<vector<double>> p(new vector<double>);
		p->push_back(i * 2 + 1);
		p->push_back(i * 2 + 2);
		param->limbs.push_back(std::move(p));
	}
}

void PoseMachine::initModelParam(string mCaffemodel, string mDeployFile)
{
	if (param != NULL)
		deleteParam();

	param = new ModelParameter;
	param->caffemodel = mCaffemodel;
	param->deployFile = mDeployFile;
	param->description, "MPII+LSP 6-stage CPM";
	param->boxsize = 368;
	param->padValue = 128;
	param->np = 14;
	param->sigma = 21;
	param->stage = 6;
	for (int i = 0; i < 9; i++)
	{
		std::shared_ptr<vector<double>> p(new vector<double>);
		p->push_back(i * 2 + 1);
		p->push_back(i * 2 + 2);
		param->limbs.push_back(std::move(p));
	}
}

void PoseMachine::calculateScales(vector<double>& scales, int height)
{
	double middleRange = 1.2 * (height - 1) / height;
	double startRange = middleRange * 0.8;
	double endRange = middleRange * 3.0;
	int boxsize = param->boxsize;

	double startScale = boxsize / (height * endRange);
	double endScale = boxsize / (height * startRange);

	for (double i = log2(startScale); i < log2(endScale); i += (1 / (double)octave))
		scales.push_back(pow(2, i));
}

Mat PoseMachine::padAround(Mat originImg, double scale)
{
	Mat img = originImg.clone();
	int width = static_cast<int>(img.cols*scale);
	int height = static_cast<int>(img.rows*scale);
	resize(img, img, Size(width, height));

	// resize image to be (boxsize, boxsize)
	int boxsize = param->boxsize;
	int padValue = param->padValue;

	int top = __max(boxsize / 2 - height / 2, 0);
	int bottom = __max(boxsize / 2 + height / 2 - height, 0);
	int left = __max(boxsize / 2 - width / 2, 0);
	int right = __max(boxsize / 2 + width / 2 - width, 0);
	pad[0] = top;
	pad[1] = bottom;
	pad[2] = left;
	pad[3] = right;

	Scalar value = Scalar(padValue, padValue, padValue);
	copyMakeBorder(img, img, top, bottom, left, right, BORDER_CONSTANT, value);

	// crop if necessary
	int centerX = static_cast<int>(width / 2) + left;
	int centerY = static_cast<int>(height / 2) + top;
	int x = __max(centerX - boxsize / 2, 0);
	int y = __max(centerY - boxsize / 2, 0);
	cv::Rect myROI(x, y, boxsize, boxsize);
	img = img(myROI);
	return img;
}

void PoseMachine::resizeIntoScaledImg(Mat& originImg)
{
	int top = pad[0];
	int bottom = pad[1];
	int left = pad[2];
	int right = pad[3];

	int width = param->boxsize - left - right;
	int height = param->boxsize - top - bottom;
	Rect myROI(left, top, width, height);
	originImg = originImg(myROI);
}

inline void meshgrid(const cv::Mat &xgv, const cv::Mat &ygv,
	cv::Mat &X, cv::Mat &Y)
{
	cv::repeat(xgv.reshape(1, 1), ygv.total(), 1, X);
	cv::repeat(ygv.reshape(1, 1).t(), 1, xgv.total(), Y);
}

void PoseMachine::preprocess(Mat img, vector<Mat>* input_channels)
{
	Mat out(img.rows, img.cols, CV_32FC4);
	Mat X(img.rows, img.cols, CV_32FC1);
	Mat Y(img.rows, img.cols, CV_32FC1);
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			for (int k = 0; k < 3; k++)
				out.at<Vec4f>(i, j)[k] = img.at<Vec3b>(i, j)[k] / 256.0 - 0.5;
		}
	}

	int boxsize = param->boxsize;
	int sigma = param->sigma;
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			double xt = pow(i - boxsize / 2, 2);
			double yt = pow(j - boxsize / 2, 2);
			double exponent = (xt + yt) / (2 * param->sigma * param->sigma);
			out.at<Vec4f>(i, j)[3] = exp(-exponent);
		}
	}

	split(out, *input_channels);
}

void PoseMachine::wrapInputLayer(vector<Mat>* input_channels) {
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void PoseMachine::outputMatrix(string name, Mat& matrix)
{
	std::ofstream outFile(name, ios::out);
	int width = matrix.cols;
	int height = matrix.rows;
	int channels = matrix.channels();
	outFile.precision(2);
	if (channels != 1)
	{
		std::cout << "channels != 1" << std::endl;
		return;
	}

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			outFile << matrix.at<float>(y, x) << " ";
		}
		outFile << std::endl;
	}
}

void PoseMachine::loadNet()
{
	registerLayers();
	Caffe::set_mode(Caffe::CPU);
	net_.reset(new Net<float>(param->deployFile, TEST));
	net_->CopyTrainedLayersFrom(param->caffemodel);

	if (net_->num_inputs() != 1 || net_->num_outputs() != 1){
		std::cout << "Network should have exactly one input/output.";
	}
}

string PoseMachine::getStageEndLayer(int stageNum)
{
	string layername;
	if (stageNum == 1)
		layername = "conv7_stage1";
	else
		layername = "Mconv5_stage" + to_string(stageNum);

	return layername;
}

void PoseMachine::getPredictLocation(float* data, int height, int width, MyPoint& p)
{
	float maxValue = 0;
	p.x = 0;
	p.y = 0;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int index = i*width + j;
			if (data[index] > maxValue)
			{
				maxValue = data[index];
				p.x = j;
				p.y = i;
			}

		}
	}


}

void PoseMachine::getPrediction(string filename, int stage, MyPoint* predict)
{
	// read image
	cout << "filename=" << filename << endl;
	Mat img = imread(filename);
	if (img.empty())
	{
		std::cout << "cannot read image from file: " << filename << std::endl;
		return;
	}

	// preprocess
	vector<double> scales;
	calculateScales(scales, img.rows);
	Mat sImg = padAround(img, scales[octave]);
	vector<cv::Mat> input_channels;
	wrapInputLayer(&input_channels);
	preprocess(sImg, &input_channels);

	// find Layer id
	int layerId = 0;
	auto layer_names = net_->layer_names();
	for (int i = 0; i<layer_names.size(); i++) {
		if (layer_names[i].compare(getStageEndLayer(stage)) == 0) {
			layerId = i;
			break;
		}
	}
	cout << "getStageEndLayer(stage) = " << getStageEndLayer(stage) << endl;
	cout << "layerId = " << layerId << endl;

	// forward net
	net_->ForwardTo(layerId);

	// output
	int blobId = net_->top_ids(layerId)[0];
	auto blob = net_->blobs()[blobId];
	const float* output_data = blob->cpu_data();
	int width = blob->width();
	int height = blob->height();
	for (int j = 0; j < blob->channels(); j++)
	{
		if (j == 14)	// don't need the background layer
			break;

		float* temp = new float[height*width];
		memcpy(temp, output_data, sizeof(float)*height*width);
		getPredictLocation(temp, height, width, predict[j]);
		
		output_data += width * height;
		
		if (temp != NULL)
		{
			delete[] temp;
			temp = NULL;
		}
	}

	vector<double>().swap(scales);
}

vector<Mat> PoseMachine::applyModel(string image)
{
	loadNet();

	// read image
	std::cout << "[image] " << image << std::endl;
	Mat img = imread(image);
	if (img.empty())
	{
		std::cout << "cannot read image from file: " << image << std::endl;
		return img;
	}

	vector<double> scales;
	calculateScales(scales, img.rows);

	// calculate prediction and heatmaps
	for (int i = 0; i < static_cast<int>(scales.size()); i++)
	{
		// input
		Mat sImg = padAround(img, scales[i]);
		vector<cv::Mat> input_channels;
		wrapInputLayer(&input_channels);
		preprocess(sImg, &input_channels);
		net_->Forward();

		// output
		Blob<float>* output_layer = net_->output_blobs()[0];
		const float* output_data = output_layer->cpu_data();
		int width = output_layer->width();
		int height = output_layer->height();
		vector<Mat> output_channels;
		for (int j = 0; j < output_layer->channels(); j++)
		{
			float* temp = new float[height*width];
			memcpy(temp, output_data, sizeof(float)*height*width);
			Mat channel(height, width, CV_32FC1, temp);
			output_channels.push_back(channel);
			output_data += width * height;
		}

		// resize
		for (int j = 0; j < output_channels.size(); j++)
		{
			resize(output_channels[j], output_channels[j], Size(param->boxsize, param->boxsize));
			resizeIntoScaledImg(output_channels[j]);
			resize(output_channels[j], output_channels[j], Size(img.cols, img.rows));
		}

		return output_channels;
	}
}

namespace caffe
{
	// Get input layer according to engine
	template <typename Dtype>
	shared_ptr<Layer<Dtype> > GetInputLayer(const LayerParameter& param) {
		InputParameter input_param = param.input_param();
		return shared_ptr<Layer<Dtype> >(new InputLayer<Dtype>(param));
	}

	template <typename Dtype>
	shared_ptr<Layer<Dtype> > GetSliceLayer(const LayerParameter& param) {
		SliceParameter slice_param = param.slice_param();
		return shared_ptr<Layer<Dtype> >(new SliceLayer<Dtype>(param));
	}

	template <typename Dtype>
	shared_ptr<Layer<Dtype> > GetSplitLayer(const LayerParameter& param) {
		return shared_ptr<Layer<Dtype> >(new SplitLayer<Dtype>(param));
	}

	template <typename Dtype>
	shared_ptr<Layer<Dtype> > GetConcatLayer(const LayerParameter& param) {
		ConcatParameter concat_param = param.concat_param();
		return shared_ptr<Layer<Dtype> >(new ConcatLayer<Dtype>(param));
	}

	// Get convolution layer according to engine.
	template <typename Dtype>
	shared_ptr<Layer<Dtype> > GetConvolutionLayer(
		const LayerParameter& param) {
		ConvolutionParameter conv_param = param.convolution_param();
		ConvolutionParameter_Engine engine = conv_param.engine();
#ifdef USE_CUDNN
		bool use_dilation = false;
		for (int i = 0; i < conv_param.dilation_size(); ++i) {
			if (conv_param.dilation(i) > 1) {
				use_dilation = true;
			}
		}
#endif
		if (engine == ConvolutionParameter_Engine_DEFAULT) {
			engine = ConvolutionParameter_Engine_CAFFE;
#ifdef USE_CUDNN
			if (!use_dilation) {
				engine = ConvolutionParameter_Engine_CUDNN;
			}
#endif
		}
		if (engine == ConvolutionParameter_Engine_CAFFE) {
			return shared_ptr<Layer<Dtype> >(new ConvolutionLayer<Dtype>(param));
#ifdef USE_CUDNN
		}
		else if (engine == ConvolutionParameter_Engine_CUDNN) {
			if (use_dilation) {
				LOG(FATAL) << "CuDNN doesn't support the dilated convolution at Layer "
					<< param.name();
			}
			return shared_ptr<Layer<Dtype> >(new CuDNNConvolutionLayer<Dtype>(param));
#endif
		}
		else {
			LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
		}
	}

	// Get pooling layer according to engine.
	template <typename Dtype>
	shared_ptr<Layer<Dtype> > GetPoolingLayer(const LayerParameter& param) {
		PoolingParameter_Engine engine = param.pooling_param().engine();
		if (engine == PoolingParameter_Engine_DEFAULT) {
			engine = PoolingParameter_Engine_CAFFE;
#ifdef USE_CUDNN
			engine = PoolingParameter_Engine_CUDNN;
#endif
		}
		if (engine == PoolingParameter_Engine_CAFFE) {
			return shared_ptr<Layer<Dtype> >(new PoolingLayer<Dtype>(param));
#ifdef USE_CUDNN
		}
		else if (engine == PoolingParameter_Engine_CUDNN) {
			if (param.top_size() > 1) {
				LOG(INFO) << "cuDNN does not support multiple tops. "
					<< "Using Caffe's own pooling layer.";
				return shared_ptr<Layer<Dtype> >(new PoolingLayer<Dtype>(param));
			}
			// CuDNN assumes layers are not being modified in place, thus
			// breaking our index tracking for updates in some cases in Caffe.
			// Until there is a workaround in Caffe (index management) or
			// cuDNN, use Caffe layer to max pooling, or don't use in place
			// layers after max pooling layers
			if (param.pooling_param().pool() == PoolingParameter_PoolMethod_MAX) {
				return shared_ptr<Layer<Dtype> >(new PoolingLayer<Dtype>(param));
			}
			else {
				return shared_ptr<Layer<Dtype> >(new CuDNNPoolingLayer<Dtype>(param));
			}
#endif
		}
		else {
			LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
		}
	}

	// Get relu layer according to engine.
	template <typename Dtype>
	shared_ptr<Layer<Dtype> > GetReLULayer(const LayerParameter& param) {
		ReLUParameter_Engine engine = param.relu_param().engine();
		if (engine == ReLUParameter_Engine_DEFAULT) {
			engine = ReLUParameter_Engine_CAFFE;
#ifdef USE_CUDNN
			engine = ReLUParameter_Engine_CUDNN;
#endif
		}
		if (engine == ReLUParameter_Engine_CAFFE) {
			return shared_ptr<Layer<Dtype> >(new ReLULayer<Dtype>(param));
#ifdef USE_CUDNN
		}
		else if (engine == ReLUParameter_Engine_CUDNN) {
			return shared_ptr<Layer<Dtype> >(new CuDNNReLULayer<Dtype>(param));
#endif
		}
		else {
			LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
		}
	}
}

void PoseMachine::registerLayers()
{
	REGISTER_LAYER_CREATOR(Input, GetInputLayer);
	REGISTER_LAYER_CREATOR(Slice, GetSliceLayer);
	REGISTER_LAYER_CREATOR(Split, GetSplitLayer);
	REGISTER_LAYER_CREATOR(Concat, GetConcatLayer);
	REGISTER_LAYER_CREATOR(Convolution, GetConvolutionLayer);
	REGISTER_LAYER_CREATOR(Pooling, GetPoolingLayer);
	REGISTER_LAYER_CREATOR(ReLU, GetReLULayer);
}

