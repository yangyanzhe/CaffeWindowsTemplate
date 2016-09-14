#ifndef IMAGELIST_H
#define IMAGELIST_H

#include <string>
#include <vector>

class ImageList
{
public:
	ImageList(const char* basePath, const char* filename);

	std::string mBasePath;
	std::vector<std::string> mImagePaths;
	int length;
};

#endif