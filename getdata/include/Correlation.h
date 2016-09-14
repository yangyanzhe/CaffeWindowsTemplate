#ifndef CORRELATION_H
#define CORRELATION_H

#include <opencv2\highgui\highgui.hpp>

class MyPoint
{
public:
	int x;
	int y;
	MyPoint(){ x = 0; y = 0; };
	~MyPoint(){};
};

class PointPair
{
public:
	int x1;
	int y1;
	int x2;
	int y2;
	PointPair(){ x1 = 0; y1 = 0; x2 = 0; y2 = 0; };
	~PointPair(){};
};

class Correlation
{
public:
	Correlation(int partNum, int imageNum);
	~Correlation();

private:
	int mPartNum;
	int mImageNum;
	float **mPartCorr;
	MyPoint **mPredictPre;
	MyPoint **mPredictFol;
	std::vector<std::string> mPartStr;
	

public:
	void printPointMap();
	void storePredict(MyPoint* predictPre, MyPoint* predictFol, int imageId);
	void resetContent();
	void outputDataJs(int stagePre, int stageFol);
	void calculateCorr(int partPre, int partFol);
	void calculateCorrAll();
};

#endif