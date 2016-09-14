#ifndef COLOR_GRADIENT_H
#define COLOR_GRADIENT_H

#include <vector>
using std::vector;

class ColorGradient
{
private:
	struct ColorPoint  // Internal class used to store colors at different points in the gradient.
	{
		float h, s, v;      // Red, green and blue values of our color.
		float val;        // Position of our color along the gradient (between 0 and 1).
		ColorPoint(float red, float green, float blue, float value)
			: r(red), g(green), b(blue), val(value) {}
	};
	vector<ColorPoint> color;      // An array of color points in ascending value.


public:
	ColorGradient();
	~ColorGradient();

	void createDefaultHeatMapGradient();
	void getColorAtValue(const float value, float &red, float &green, float &blue);

};

#endif