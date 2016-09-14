#include <iostream>
#include <fstream>
#include "ImageList.h"

using namespace std;

ImageList::ImageList(const char* basePath, const char* filename) {
	string line;
	ifstream input(filename);

	mBasePath = basePath;

	while (getline(input, line))
	{
		mImagePaths.push_back(line);
	}
	input.close();

	length = mImagePaths.size();
	cout << "Loaded image with " << mImagePaths.size() << " images. " << endl;
}