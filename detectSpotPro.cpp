#include <iostream>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/unordered_map.hpp>
#include <boost/graph/floyd_warshall_shortest.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/features2d.hpp"  
#include <math.h>
#include <vector>
#include <string>
#include <fstream>
#include "SWT.h"

using namespace swt;
using namespace cv;
using namespace std;

#define PI 3.14159265

struct colorInfo
{
    int blueValue;
    int greenValue;
    int redValue;
};
Mat loadImage(int num){
    string filename;
    //string filename = to_string(num) + "_1.jpg";
    /*
    if(num < 58){
        filename = to_string(num) + "_1.jpg"; //marked sample
    }else{
        filename = to_string(num) + "_0.jpg"; //unmarked sample
    }*/
    filename = to_string(num) + ".jpg";
    //string imagepath = "/home/george/data/nBlog/"+filename;
    string imagepath = "/home/george/data/dotlines/" + filename;
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

std::vector<std::vector<Point>> afterSWT(Mat& input,Mat& edgeImage,Mat& gradientX,Mat& gradientY,Mat& validComponentsImage,Mat& SWTImage){
   
    std::vector<Ray> rays;

    // ofstream fout;
    // fout.open("point.txt");

    SWT swt;
    swt.StrokeWidthTransform(edgeImage,gradientX,gradientY,1,SWTImage,rays);
    swt.SWTMedianFilter(SWTImage, rays);

    std::vector<std::vector<SWTPoint2d>> components = swt.findLegallyConnectedComponents(SWTImage,rays);
    std::vector<std::vector<SWTPoint2d>> validComponents = swt.filterComponents(SWTImage,components);

    std::vector<std::vector<Point>> gotComponents;
    gotComponents.reserve(validComponents.size());

    for(std::vector<std::vector<SWTPoint2d>>::const_iterator it1 = validComponents.begin();it1 != validComponents.end();it1++){
        if(it1->size() > 0){
            std::vector<Point> temp_comp;
            temp_comp.reserve(it1->size());
            for(std::vector<SWTPoint2d>::const_iterator it2 = it1->begin();it2 != it1->end();it2++){
                validComponentsImage.at<uchar>(it2->y,it2->x) = 255;
                Point p(it2->x,it2->y);
                // fout<<"("<<it2->x<<","<<it2->y<<") ";
                temp_comp.push_back(p);
            }
            // fout<<"\n";
            gotComponents.push_back(temp_comp);
        }
    }
    // fout.close();
    cout<<"Let's check the validComponents's size is: "<<validComponents.size()<<" --- the gotComponents'size is: "<<gotComponents.size()<<endl;

    return gotComponents;
}

void doDetect(std::vector<std::vector<Point>>& components,Mat& input,std::vector<std::vector<Point>>& hullfilter){
    for(std::vector<std::vector<Point>>::const_iterator it1 = components.begin();it1 != components.end();it1++){
        Mat temp_convex = Mat::zeros(input.size(),CV_8UC1);

        float solidity = 0.0;
        std::vector<colorInfo> colorInfos;
        colorInfos.reserve(it1->size());

        std::vector<float> rgbRatio;

        for(std::vector<Point>::const_iterator it2 = it1->begin();it2 != it1->end(); it2++){
            temp_convex.at<unsigned char>(it2->y,it2->x) = 255;
            //colorInfo ci;
            //ci.redValue = input.at<Vec3b>(it2->y,it2->x)[0];
            //ci.greenValue = input.at<Vec3b>(it2->y,it2->x)[1];
            //ci.blueValue = input.at<Vec3b>(it2->y,it2->x)[2];
            //colorInfos.push_back(ci);

            int redValue = input.at<Vec3b>(it2->y,it2->x)[0];
            int greenValue = input.at<Vec3b>(it2->y,it2->x)[1];
            int blueValue = input.at<Vec3b>(it2->y,it2->x)[2];

            float ratio1 = redValue/(greenValue + 0.001);
            float ratio2 = redValue/(blueValue + 0.001);
            float ratio3 = greenValue/(blueValue + 0.001);

            rgbRatio.push_back(ratio1);
            rgbRatio.push_back(ratio2);
            rgbRatio.push_back(ratio3);
        }

        float maxRatio = *max_element(rgbRatio.begin(),rgbRatio.end());
        float minRatio = *min_element(rgbRatio.begin(),rgbRatio.end());

        bool colorNo = true;
        
        if((maxRatio > 2) || (minRatio < 0.5)){
            colorNo = false;
        }
        float compArea = countNonZero(temp_convex);

        vector<vector<Point>> contours;
        findContours( temp_convex, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );

        Rect bRect = boundingRect(contours[0]);

        if( !contours.empty() ) {
            vector<vector<Point>> hull(1);
            convexHull( contours[0], hull[0] );
            // cout<<contourArea(hull[0])<<endl;
            solidity = compArea / (contourArea( hull[0] ) + 1);
        }
        bool tooSmall = true;
        if((max(bRect.width,bRect.height)>120) && (min(bRect.height,bRect.width) > 30)){
            tooSmall = false;
        }
        //cout<<"-->solidity is: "<<solidity<<endl;
        //there will be 3 conditions for detect marked line:area ratio,width&height ratio,color var
       if((solidity < 0.4) && !tooSmall){
            hullfilter.push_back(*it1);
            
            //out<<"var-->"<<varience<<endl;
        }else if((max(bRect.width,bRect.height) / min(bRect.width,bRect.height) > 3) && !tooSmall){
            hullfilter.push_back(*it1);
        }else if(!colorNo && !tooSmall){
            //cout<<"add by color"<<endl;
            hullfilter.push_back(*it1);
        }/*else{
            cout<<"max ratio: "<<maxRatio<<" min ratio: "<<minRatio<<endl;
        }*/
    }
    cout<<"hullfilter'size is: "<<hullfilter.size()<<endl;
}

void doDetectHSV(std::vector<std::vector<Point>>& components,Mat& input,std::vector<std::vector<Point>>& hullfilter){
    Mat inputHSV;
    cvtColor(input,inputHSV,CV_RGB2HSV);
    for(std::vector<std::vector<Point>>::const_iterator it1 = components.begin();it1 != components.end();it1++){
        Mat temp_convex = Mat::zeros(input.size(),CV_8UC1);

        float solidity = 0.0;

        std::vector<int> hValue;
        std::vector<int> sValue;
        std::vector<int> vValue;
        std::vector<int> rgbRatio;
        for(std::vector<Point>::const_iterator it2 = it1->begin();it2 != it1->end(); it2++){
            int redValue;
            int greenValue;
            int blueValue;

            temp_convex.at<unsigned char>(it2->y,it2->x) = 255;
            hValue.push_back(inputHSV.at<Vec3b>(it2->y,it2->x)[0]);
            sValue.push_back(inputHSV.at<Vec3b>(it2->y,it2->x)[1]);
            vValue.push_back(inputHSV.at<Vec3b>(it2->y,it2->x)[2]);

            redValue = input.at<Vec3b>(it2->y,it2->x)[0];
            greenValue = input.at<Vec3b>(it2->y,it2->x)[1];
            blueValue = input.at<Vec3b>(it2->y,it2->x)[2];

            int ratio1 = abs(redValue - greenValue);
            int ratio2 = abs(redValue - blueValue);
            int ratio3 = abs(greenValue - blueValue);
            rgbRatio.push_back((ratio1 + ratio2 + ratio3)/3);

        }
        int maxHValue = *max_element(hValue.begin(),hValue.end());
        int minHValue = *min_element(hValue.begin(),hValue.end());
        // int maxSValue = *max_element(sValue.begin(),sValue.end());
        // int maxVValue = *max_element(vValue.begin(),vValue.end());
        float maxRatio = *max_element(rgbRatio.begin(),rgbRatio.end());
        bool hsvJudge = false;

        int disH = ((maxHValue - minHValue) <= 90)?(maxHValue - minHValue):(180 - maxHValue + minHValue);

        if((maxRatio > 20) || (disH > 60)){
            hsvJudge = true;
        }
        float compArea = countNonZero(temp_convex);

        vector<vector<Point>> contours;
        findContours( temp_convex, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );

        Rect bRect = boundingRect(contours[0]);

        if( !contours.empty() ) {
            vector<vector<Point>> hull(1);
            convexHull( contours[0], hull[0] );
            // cout<<contourArea(hull[0])<<endl;
            solidity = compArea / (contourArea( hull[0] ) + 1);
        }
        bool tooSmall = true;
        if((max(bRect.width,bRect.height)>120) && (min(bRect.height,bRect.width) > 80)){
            tooSmall = false;
        }
        bool notCharacter = false;
        bool notSpot = false;
        if(max(bRect.width,bRect.height)>45 && (min(bRect.height,bRect.width)>45)){
            notSpot = true;
        }
        if(max(bRect.width,bRect.height)>80){
            notCharacter = true;
        }
        //cout<<"-->solidity is: "<<solidity<<endl;
        //there will be 4 conditions for detect marked line:area ratio,width&height ratio,color var
        
        if((solidity <0.4) && notCharacter){
            hullfilter.push_back(*it1);
        }else if((max(bRect.width,bRect.height) / min(bRect.width,bRect.height) > 3) && notCharacter){
            hullfilter.push_back(*it1);
        }else if(hsvJudge && notCharacter){
            hullfilter.push_back(*it1);
            //cout<<"max H value is: "<<disH<<endl;
        }else if(((bRect.width / input.cols) > 0.3) || ((bRect.height / input.rows) > 0.3)){
            hullfilter.push_back(*it1);
        }
        //hullfilter.push_back(*it1);
    }
    cout<<"hullfilter'size is: "<<hullfilter.size()<<endl;

}

void plotComponents_rgb(std::vector<std::vector<Point>>& components,Mat& showComponents){
    assert (showComponents.channels() == 3);

    int count = 0;
    for(std::vector<std::vector<Point>>::const_iterator it1 = components.begin();it1 != components.end();it1++){
        for(std::vector<Point>::const_iterator it2 = it1->begin();it2 != it1->end();it2++){
            if(count % 6 == 0){
                showComponents.at<Vec3b>(it2->y,it2->x)[0] = 255;
                showComponents.at<Vec3b>(it2->y,it2->x)[1] = 0;
                showComponents.at<Vec3b>(it2->y,it2->x)[2] = 0;
            }else if(count % 6 == 1){
                showComponents.at<Vec3b>(it2->y,it2->x)[0] = 0;
                showComponents.at<Vec3b>(it2->y,it2->x)[1] = 255;
                showComponents.at<Vec3b>(it2->y,it2->x)[2] = 0;
            }else if(count % 6 == 2){
                showComponents.at<Vec3b>(it2->y,it2->x)[0] = 0;
                showComponents.at<Vec3b>(it2->y,it2->x)[1] = 0;
                showComponents.at<Vec3b>(it2->y,it2->x)[2] = 255;
            }else if(count % 6 == 3){
                showComponents.at<Vec3b>(it2->y,it2->x)[0] = 255;
                showComponents.at<Vec3b>(it2->y,it2->x)[1] = 255;
                showComponents.at<Vec3b>(it2->y,it2->x)[2] = 0;
            }else if(count % 6 == 4){
                showComponents.at<Vec3b>(it2->y,it2->x)[0] = 255;
                showComponents.at<Vec3b>(it2->y,it2->x)[1] = 0;
                showComponents.at<Vec3b>(it2->y,it2->x)[2] = 255;
            }else{
                showComponents.at<Vec3b>(it2->y,it2->x)[0] = 0;
                showComponents.at<Vec3b>(it2->y,it2->x)[1] = 255;
                showComponents.at<Vec3b>(it2->y,it2->x)[2] = 255;
            }
        }
        count++;
    }

}

void plotComponents_black(std::vector<std::vector<Point>>& components,Mat& showComponents){
    for(std::vector<std::vector<Point>>::const_iterator it1 = components.begin();it1 != components.end();it1++){
        for(std::vector<Point>::const_iterator it2 = it1->begin();it2 != it1->end();it2++){
            showComponents.at<uchar>(it2->y,it2->x) = 255;
        }
    }
}

int main(){
    for(int num = 55;num < 56;num++){
        cout<<"Now process the No. "<<num<<" Image"<<endl;
        Mat input = loadImage(num);
        Mat grayImage = convert2gray(input);
        Mat edgeImage = doCanny(grayImage);
        
        Mat inputHSV;
        cvtColor(input,inputHSV,CV_RGB2HSV);
        
        Mat gradientX( input.size(), CV_32FC1 );
        Mat gradientY( input.size(), CV_32FC1 );
        Mat gaussianImage( grayImage.size(), CV_32FC1); // 元素类型为 32位float型，一个通道
        grayImage.convertTo(gaussianImage, CV_32FC1, 1./255.);

        GaussianBlur( gaussianImage, gaussianImage, Size(5, 5), 0); // blurs an image using a Gaussian filter
        Scharr(gaussianImage, gradientX, -1, 1, 0); // calculates the x- image derivative 
        Scharr(gaussianImage, gradientY, -1, 0, 1);  // calculates the y- image derivative
        GaussianBlur(gradientX, gradientX, Size(3, 3), 0);
        GaussianBlur(gradientY, gradientY, Size(3, 3), 0); 

        Mat validComponentsImage = Mat::zeros(input.size(),CV_8UC1);
        Mat SWTImage( input.size(), CV_32FC1 );
        for( int row = 0; row < input.rows; row++ ){
            float* ptr = (float*)SWTImage.ptr(row);   // cv::Mat::ptr() : return a pointer to the specified matrix row
            for ( int col = 0; col < input.cols; col++ ){
                *ptr++ = -1;
            }
        }

        std::vector<std::vector<Point>> swtPoints = afterSWT(input,edgeImage,gradientX,gradientY,validComponentsImage,SWTImage);

        //test the HSV value of the component
        /*
        Mat testComponent(input.size(),CV_8UC1);
        testComponent.setTo(255); //just for indicate which component
        std::vector<Point> testVector = swtPoints[13];
        cout<<"HSV Info show here:"<<endl;
        for(std::vector<Point>::const_iterator it = testVector.begin();it != testVector.end();it++){
            testComponent.at<uchar>(it->y,it->x) = 0;
            cout<<"H is: "<<(int)inputHSV.at<Vec3b>(it->y,it->x)[0]<<endl;
            // cout<<"S is: "<<(int)inputHSV.at<Vec3b>(it->y,it->x)[1]<<endl;
            // cout<<"V is: "<<(int)inputHSV.at<Vec3b>(it->y,it->x)[2]<<endl;
        }
        imshow("test",testComponent);
        waitKey(-1);
        cout<<"HSV Info show end!"<<endl;
        */
        
        Mat validComponentsImage_rgb = Mat::zeros(input.size(),CV_8UC3);
        validComponentsImage_rgb = ~validComponentsImage_rgb;
        plotComponents_rgb(swtPoints,validComponentsImage_rgb);
        imwrite("../data/SWT/"+to_string(num)+".jpg",validComponentsImage_rgb);
        cout<<"SWT end!"<<endl;
        
        std::vector<std::vector<Point>> hullfilter;
        //doDetect(swtPoints,input,hullfilter);
        doDetectHSV(swtPoints,input,hullfilter);
        Mat showHullResult = Mat::zeros(input.size(),CV_8UC1);
        plotComponents_black(hullfilter,showHullResult);

        Mat SWT2Show = 255 * SWTImage;
        //output result
        Mat intermediateResult = Mat::zeros(2 * input.rows + 10, 2 * input.cols + 10, CV_8UC1);
        SWT2Show.copyTo(intermediateResult(Rect(0 , 0, SWTImage.cols, SWTImage.rows)));
        validComponentsImage.copyTo(intermediateResult(Rect(SWTImage.cols + 10,0 , validComponentsImage.cols, validComponentsImage.rows)));
        showHullResult.copyTo(intermediateResult(Rect(0 ,SWTImage.rows + 10, showHullResult.cols, showHullResult.rows)));
        cvtColor (intermediateResult, intermediateResult, CV_GRAY2RGB);
        input.copyTo(intermediateResult(Rect(showHullResult.cols + 10, SWTImage.rows + 10, input.cols, input.rows)));
        imwrite("../data/marked/"+to_string(num)+".jpg",intermediateResult);
        // imshow("result",showHullResult);
        // waitKey(-1);
        
        cout<<"Image "<<num<<" end!"<<endl;
    }
}