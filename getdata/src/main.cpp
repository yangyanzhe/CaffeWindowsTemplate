#include <iostream>
#include "PoseMachine.h"
#include "Correlation.h"
#include "ImageList.h"
#include "ActivationDifference.h"

using namespace std;

void resetPointArray(MyPoint* list, int length)
{
	for (int i = 0; i < length; i++)
	{
		list[i].x = 0;
		list[i].y = 0;
	}
}

int main(int argc, char* argv[])
{
	if (argc < 5)
	{
		cout << "Error in parameter input." << endl;
		return -1;
	}

	string imgFolder = argv[1];
	string imgListPath = argv[2];
	string prototxtPath = argv[3];
	string modelPath = argv[4];

	// load model
	ActivationDifference diff(modelPath, prototxtPath);
	//diff.pairList.push_back(DiffPairs("1.jpg", "2.jpg", "_left_leg"));
	diff.pairList.push_back(DiffPairs("1.jpg", "3.jpg", "_right_leg"));
	diff.pairList.push_back(DiffPairs("1.jpg", "4.jpg", "_left_arm"));
	diff.pairList.push_back(DiffPairs("1.jpg", "5.jpg", "_right_arm"));
	diff.pairList.push_back(DiffPairs("1.jpg", "6.jpg", "_back_forth"));
	diff.run();
	//diff.calculateActivationDifference("1.jpg", "2.jpg");
}