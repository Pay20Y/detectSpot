#include <iostream>
#include "SWT.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>  
#include <vector>
#include <string> 
#include <fstream>

using namespace std;
using namespace cv;
using namespace swt;

struct histPoint
{
	int x;
	int y;
	float histValue;
};

Mat loadImage(int num){
    string filename = to_string(num) + "_1.jpg";
    // if(num < 58){
    //     filename = to_string(num) + "_1.jpg"; //marked sample
    // }else{
    //     filename = to_string(num) + "_0.jpg"; //unmarked sample
    // }
    string imagepath = "/home/george/data/nBlog/"+filename;
    //string imagepath = "/home/george/data/USTB-SV1K_V1/training/0001.jpg";
	Mat image = imread(imagepath);
    if (image.empty()) {
        cout<<"read image failed"<<endl;
        return Mat();
    }
    cvtColor(image, image, CV_BGR2RGB);
    return image;
}

Mat convert2gray(Mat input){
	// cout<<"convert to gray--->"<<endl;
	Mat grayImage( input.size(), CV_8UC1 ); //新建一个叫grayImage的变量用来存放灰度图，数据的类型为8U 一个通道
    cvtColor ( input, grayImage, CV_RGB2GRAY );
    return grayImage;
}

Mat doCanny(Mat& grayImage){
    // cout<<"Do canny--->"<<endl;
    double threshold_low = 150; 
    double threshold_high = 250;
    Mat edgeImage( grayImage.size(),CV_8UC1 );

    Canny(grayImage, edgeImage, threshold_low, threshold_high, 3);
    return edgeImage;
}	

void plotComponents(std::vector<std::vector<SWTPoint2d>>& components,Mat& componentsImage){
	for(std::vector<std::vector<SWTPoint2d>>::const_iterator it1 = components.begin();it1 != components.end();it1++){

		for(std::vector<SWTPoint2d>::const_iterator it2 = it1->begin();it2 != it1->end();it2++){
			componentsImage.at<uchar>(it2->y,it2->x) = 255;
		}
	}
}

void plotComponents(std::vector<SWTPoint2d> components,Mat& componentsImage){
	for(std::vector<SWTPoint2d>::const_iterator it2 = components.begin();it2 != components.end();it2++){
			componentsImage.at<uchar>(it2->y,it2->x) = 255;
	}
}

Mat doInject(Mat& input,std::vector<std::vector<SWTPoint2d>>& components){
	Mat inject = Mat::zeros(input.size(),CV_8UC3);
	inject = ~inject;
	for(std::vector<std::vector<SWTPoint2d>>::const_iterator it1 = components.begin();it1 != components.end();it1++){
		for(std::vector<SWTPoint2d>::const_iterator it2 = it1->begin();it2 != it1->end();it2++){
			inject.at<Vec3b>(it2->y,it2->x)[2] = input.at<Vec3b>(it2->y,it2->x)[0];
			inject.at<Vec3b>(it2->y,it2->x)[1] = input.at<Vec3b>(it2->y,it2->x)[1];
			inject.at<Vec3b>(it2->y,it2->x)[0] = input.at<Vec3b>(it2->y,it2->x)[2];
		}
	}
	return inject;
}

bool histSort (const histPoint &lhs, const histPoint &rhs) {
    return lhs.histValue > rhs.histValue;
}

std::vector<SWTPoint2d> calcHistFromInput(Mat& input,std::vector<std::vector<SWTPoint2d>> components){
	Mat input_rgb = input.clone();
	//input.convertTo(input_hsv,CV_RGB2HSV);
	input_rgb.convertTo(input_rgb,CV_32F);
	int histSize[2];  
    //float rranges[2];
    float granges[2];
    float branges[2];
    const float *ranges[3];
    int channels[2];
    int dims;

    //histSize[0]	= 30; 
    histSize[0]	= 255;
    histSize[1] = 255;
    //rranges[0] = 0; rranges[1] = 256;
    granges[0] = 0; granges[1] = 255;
    branges[0] = 0; branges[1] = 255;
    //ranges[0]=rranges;
    ranges[0] = granges;
    ranges[1] = branges;
    channels[0] = 0;
    channels[1] = 1;
    //channels[2]=2;
    dims = 2;

    Mat hist;

    calcHist(&input_rgb,1,channels,Mat(),hist,dims,histSize,ranges,true,false);

    cout<<"input'size is: "<<input.size()<<endl;
    cout<<"hist: "<<hist.depth()<<" "<<hist.channels()<<" "<<hist.size()<<endl;

    double maxValue,minValue;
    Point minPoint,maxPoint;
    minMaxLoc(hist,&minValue,&maxValue,&minPoint,&maxPoint);

    cout<<"max: "<<endl;
    cout<<"maxValue: "<<maxValue<<endl;
    
    //find the 2nd max value of hist start!
    Mat sortID(hist.size(),CV_8UC1);
    sortIdx(hist,sortID,SORT_EVERY_ROW + SORT_DESCENDING);
  
    std::vector<histPoint> v;
    for(int row = 0;row < sortID.rows;row++){
    	for(int col = 0;col < sortID.cols;col++){
    		//cout<<"hist: "<<hist.at<float>(row,col);
    		if((sortID.at<uchar>(row,col) == 0) && (hist.at<float>(row,col) != 0)){
    			histPoint hp;
    			hp.x = col;
    			hp.y = row;
    			hp.histValue = hist.at<float>(row,col);
    			v.push_back(hp);
    		}
    	}
    }
	cout<<"v.size is: "<<v.size()<<endl;

	sort(v.begin(),v.end(),&histSort);
 	maxValue = v[2].histValue; //now maxValue is the 2nd max
 	Point maxPoint_now(v[2].x,v[2].y);
 	cout<<"2nd max value is: "<<maxValue<<endl;
 	cout<<"G & B: "<<maxPoint_now.x<<" "<<maxPoint_now.y<<endl;
    
    std::vector<SWTPoint2d> targetComponents;
    for(std::vector<std::vector<SWTPoint2d>>::const_iterator it1 = components.begin();it1 != components.end();it1++){
    	for(std::vector<SWTPoint2d>::const_iterator it2 = it1->begin();it2 != it1->end();it2++){
    		int blueValue = input.at<Vec3b>(it2->y,it2->x)[0];
    		int greenValue = input.at<Vec3b>(it2->y,it2->x)[1];
    		//int redValue = input_rgb.at<Vec3b>(it2->y,it2->x)[2];
    		//cout<<blueValue<<" "<<greenValue<<endl;
    		
    		if((blueValue == maxPoint_now.x) && (greenValue == maxPoint_now.y)){
    			targetComponents.push_back(*it2);
    		}
    		
    	}
    }

    cout<<"after find the targetComponents's size is: "<<targetComponents.size()<<endl;
    return targetComponents;
}
/*
void countHist(Mat& input,std::vector<std::vector<SWTPoint2d>> components){
	int x = 255,y=255,z=255;

	for(std::vector<std::vector<SWTPoint2d>>::const_iterator it1 = components.begin();it1 != components.end();it1++){
		for(std::vector<SWTPoint2d>::const_iterator it2 = it1->begin();it2 != it1->end();it2++){

		}
	}
}
*/
int main(){
	for(int num = 11;num < 12;num++){
		cout<<"process in image "<<num<<"..."<<endl;
		Mat input = loadImage(num);
		Mat grayImage = convert2gray(input);
	    Mat edgeImage = doCanny(grayImage);

	    SWT swt;

	    Mat gradientX( input.size(), CV_32FC1 );
	    Mat gradientY( input.size(), CV_32FC1 );
	    Mat gaussianImage( grayImage.size(), CV_32FC1); // 元素类型为 32位float型，一个通道
	    grayImage.convertTo(gaussianImage, CV_32FC1, 1./255.);
	        
	    GaussianBlur( gaussianImage, gaussianImage, Size(5, 5), 0); // blurs an image using a Gaussian filter
	    Scharr(gaussianImage, gradientX, -1, 1, 0); // calculates the x- image derivative 
	    Scharr(gaussianImage, gradientY, -1, 0, 1);  // calculates the y- image derivative
	    GaussianBlur(gradientX, gradientX, Size(3, 3), 0);
	    GaussianBlur(gradientY, gradientY, Size(3, 3), 0);
	    std::vector<Ray> rays;
	        // std::vector<Ray> raysOut;

	    Mat SWTImage( input.size(), CV_32FC1 );
	    for( int row = 0; row < input.rows; row++ ){
	        float* ptr = (float*)SWTImage.ptr(row);   // cv::Mat::ptr() : return a pointer to the specified matrix row
	        for ( int col = 0; col < input.cols; col++ ){
	            *ptr++ = -1;
	        }
	    }

	    swt.StrokeWidthTransform(edgeImage,gradientX,gradientY,1,SWTImage,rays);
	    swt.SWTMedianFilter(SWTImage,rays);
	    // imshow("..",SWTImage);
	    // waitKey(-1);

	    std::vector<std::vector<SWTPoint2d>> components = swt.findLegallyConnectedComponents(SWTImage,rays);
	    std::vector<std::vector<SWTPoint2d>> filterComponents = swt.filterComponents(SWTImage,components);
	    Mat validImage = Mat::zeros(input.size(),CV_8UC1);

	    plotComponents(filterComponents,validImage);
	    
	    Mat inject = doInject(input,filterComponents);

	    // imshow("..",inject);
	    // waitKey(-1);
	    Mat componentsImage = Mat::zeros(input.size(),CV_8UC1);

	    Mat intermediateResult = Mat::zeros(input.rows , 3*input.cols + 20, CV_8UC1);

	    std::vector<SWTPoint2d> targetPoints;
	    targetPoints = calcHistFromInput(inject,filterComponents);
	    plotComponents(targetPoints,componentsImage);
	    // imshow("wish",componentsImage);
	    // waitKey(-1);
	    validImage.copyTo(intermediateResult(Rect(0 , 0, validImage.cols, validImage.rows)));
	    componentsImage.copyTo(intermediateResult(Rect(validImage.cols+10 , 0, componentsImage.cols, componentsImage.rows)));

	    cvtColor (intermediateResult, intermediateResult, CV_GRAY2RGB);

	    input.copyTo(intermediateResult(Rect(2*validImage.cols+20 , 0, input.cols, input.rows)));
	    imwrite("data/marked/"+to_string(num)+".jpg",intermediateResult);
	    cout<<"Image "<<num<<" end!"<<endl;
	}
}