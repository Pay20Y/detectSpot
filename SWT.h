
#ifndef TEXTDETECTION_H
#define TEXTDETECTION_H

#include <opencv2/core/core.hpp>

namespace swt{

struct SWTPoint2d {
    int x;
    int y;
    float swtValue;
};

struct Ray {
        SWTPoint2d p;
        SWTPoint2d q;
        std::vector<SWTPoint2d> points;
};

class SWT{
	public:
		void StrokeWidthTransform(const cv::Mat& edgeImage,cv::Mat& gradientX,cv::Mat& gradientY,bool dark_on_light,cv::Mat& SWTImage,std::vector<Ray> & rays);
		void SWTMedianFilter (cv::Mat& SWTImage, std::vector<Ray> & rays);
		std::vector< std::vector<SWTPoint2d>> findLegallyConnectedComponents (cv::Mat& SWTImage, std::vector<Ray> & rays);
		std::vector<std::vector<SWTPoint2d>> filterComponents(cv::Mat& SWTImage,std::vector<std::vector<SWTPoint2d> > & components);
};

}//end of namespace
#endif