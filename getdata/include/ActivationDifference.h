#ifndef ACTIVATION_DIFFERENCE_H
#define ACTIVATION_DIFFERENCE_H

#include "PoseMachine.h"
#include <string>
using std::string;

class DiffPairs
{
public:
	DiffPairs(string a, string b, string c){ imgPath1 = a; imgPath2 = b; description = c; }
	~DiffPairs(){}

	string imgPath1;
	string imgPath2;
	string description;
};

class ActivationDifference
{
public:
	ActivationDifference(string modelPath, string prototxtPath);
	~ActivationDifference();
	

private:
	vector<int> layerList;
	

	PoseMachine* pm1;
	PoseMachine* pm2;
	int getLayerId(string name);
	void getLayerList();
	bool judgeToPrint(Mat& result);
	void normalizeMatrix(Mat& input, Mat& output);
	Mat generateHeatMap(Mat& activation);
	Mat generateDifferenceMap(Mat& map1, Mat& map2);
	// fR, fG, fB \in 0-1, fG [0, 369] fS [0,1] fV [0,1]
	void HSVtoRGB(float& fR, float& fG, float& fB, float& fH, float& fS, float& fV);

public:
	vector<DiffPairs> pairList;
	void calculateDifference();
	void calculateActivationDifference(string imgPath1, string imgPath2, string description);
	void run();
};

#endif
