#include "Correlation.h"
#include <opencv2\highgui\highgui.hpp>
#include <string>
#include <iostream>
#include <fstream>

#define epsilon 1e-4

using namespace std;

Correlation::Correlation(int partNum, int imageNum)
{
	mPartNum = partNum;
	mImageNum = imageNum;
	mPartCorr = new float*[mPartNum];
	mPredictPre = new MyPoint*[mPartNum];
	mPredictFol = new MyPoint*[mPartNum];
	for (int i = 0; i < mPartNum; i++)
	{
		mPartCorr[i] = new float[mPartNum];
		mPredictPre[i] = new MyPoint[mImageNum];
		mPredictFol[i] = new MyPoint[mImageNum];
		memset(mPartCorr[i], 0, sizeof(float)*mPartNum);
		memset(mPredictPre[i], 0, sizeof(MyPoint)*mImageNum);
		memset(mPredictFol[i], 0, sizeof(MyPoint)*mImageNum);
	}

	mPartStr.push_back("head");
	mPartStr.push_back("neck");
	mPartStr.push_back("Rsho");
	mPartStr.push_back("Relb");
	mPartStr.push_back("Rwri");
	mPartStr.push_back("Lsho");
	mPartStr.push_back("Lelb");
	mPartStr.push_back("Lwri");
	mPartStr.push_back("Rhip");
	mPartStr.push_back("Rkne");
	mPartStr.push_back("Rank");
	mPartStr.push_back("Lhip");
	mPartStr.push_back("Lkne");
	mPartStr.push_back("Lank");
}

Correlation::~Correlation()
{
	for (int i = 0; i < mPartNum; i++)
	{
		if (mPartCorr[i] != NULL)
		{
			delete[](mPartCorr[i]);
			mPartCorr[i] = NULL;
		}
		if (mPredictPre[i] != NULL)
		{
			delete[](mPredictPre[i]);
			mPredictPre[i] = NULL;
		}
		if (mPredictFol[i] != NULL)
		{
			delete[](mPredictFol[i]);
			mPredictFol[i] = NULL;
		}
	}

	if (mPartCorr != NULL)
	{
		delete[]mPartCorr;
		mPartCorr = NULL;
	}
	if (mPredictFol != NULL)
	{
		delete[]mPredictFol;
		mPredictFol = NULL;
	}
	if (mPredictFol != NULL)
	{
		delete[]mPredictFol;
		mPredictFol = NULL;
	}
}

void Correlation::outputDataJs(int stagePre, int stageFol)
{
	string variable = "stage_" + to_string(stagePre * 10 + stageFol);
	string filename = variable + ".js";
	ofstream output(filename, ios::out);

	output << "var " << variable << " = [";
	for (int i = 0; i < mPartNum; i++)
	{
		for (int j = 0; j < mPartNum; j++)
		{
			int corr = static_cast<int>(mPartCorr[i][j] * 1000);
			output << "['" << mPartStr[i] << "','" << mPartStr[j] << "'," << corr << "]";
			if (i == mPartNum - 1 && j == mPartNum - 1)
				output << endl << "];";
			else
				output << "," << endl;
		}
	}

	output.close();
	cout << "output to " << filename << endl;

}

void Correlation::resetContent()
{
	for (int i = 0; i < mPartNum; i++)
	{
		memset(mPartCorr[i], 0, sizeof(float)*mPartNum);
		memset(mPredictPre[i], 0, sizeof(MyPoint)*mImageNum);
		memset(mPredictFol[i], 0, sizeof(MyPoint)*mImageNum);
	}
}

void Correlation::calculateCorr(int partPre, int partFol)
{
	float meanPreX = 0;
	float meanPreY = 0;
	float meanFolX = 0;
	float meanFolY = 0;
	float num = 0;
	float den = 0;
	for (int i = 0; i < mImageNum; i++)
	{
		meanPreX += mPredictPre[partPre][i].x;
		meanPreY += mPredictPre[partPre][i].y;
		meanFolX += mPredictFol[partFol][i].x;
		meanFolY += mPredictFol[partFol][i].y;
	}
	meanPreX /= mImageNum;
	meanPreY /= mImageNum;
	meanFolX /= mImageNum;
	meanFolY /= mImageNum;

	float v1s = 0, v2s = 0;
	for (int i = 0; i < mImageNum; i++)
	{
		float v1 = pow(mPredictPre[partPre][i].x - meanPreX, 2) + 
			       pow(mPredictPre[partPre][i].y - meanPreY, 2);
		float v2 = pow(mPredictFol[partFol][i].x - meanFolX, 2) +
				   pow(mPredictFol[partFol][i].y - meanFolY, 2);
		v1s += v1;
		v2s += v2;
		num = num + sqrt(v1)*sqrt(v2);
	}

	den = sqrt(v1s * v2s);
	mPartCorr[partPre][partFol] = num / den;
}

void Correlation::calculateCorrAll()
{
	for (int i = 0; i < mPartNum; i++)
	{
		for (int j = 0; j < mPartNum; j++)
		{
			calculateCorr(i, j);
		}
	}
}

void Correlation::storePredict(MyPoint* predictPre, MyPoint* predictFol, int imageId)
{
	for (int i = 0; i < mPartNum; i++)
	{
		mPredictPre[i][imageId].x = predictPre[i].x;
		mPredictPre[i][imageId].y = predictPre[i].y;
		mPredictFol[i][imageId].x = predictFol[i].x;
		mPredictFol[i][imageId].y = predictFol[i].y;
	}
}

void Correlation::printPointMap()
{
	for (int i = 0; i < mPartNum; i++)
	{
		for (int j = 0; j < mImageNum; j++)
		{
			cout << "Part " << i << ", Image " << j << ", Predicted Position (" << mPredictPre[i][j].x << ", " << mPredictPre[i][j].y << ")" << endl;
		}
	}
}