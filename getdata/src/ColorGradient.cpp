#include "ColorGradient.h"
#include <iostream>

using namespace std;

ColorGradient::ColorGradient()
{

}

ColorGradient::~ColorGradient()
{
	vector<ColorPoint>().swap(color);
}

void ColorGradient::createDefaultHeatMapGradient()
{
	color.push_back(ColorPoint(1, 0, 0, 1.0f));		// Red
	color.push_back(ColorPoint(1, 1, 1, 0.0f));		// white
	color.push_back(ColorPoint(0, 1, 0, -1.0f));		// Green
}

void ColorGradient::getColorAtValue(const float value, float &red, float &green, float &blue)
{
	if (color.size() == 0)
		return;

	for (int i = 0; i<color.size(); i++)
	{
		ColorPoint &currC = color[i];
		if (value < currC.val)
		{
			ColorPoint &prevC = color[__max(0, i - 1)];
			float valueDiff = (prevC.val - currC.val);
			float fractBetween = (valueDiff == 0) ? 0 : (value - currC.val) / valueDiff;
			red = (prevC.r - currC.r)*fractBetween + currC.r;
			green = (prevC.g - currC.g)*fractBetween + currC.g;
			blue = (prevC.b - currC.b)*fractBetween + currC.b;
			return;
		}
	}
	red = color.back().r;
	green = color.back().g;
	blue = color.back().b;
}