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
#include <sys/types.h> //to get file from folder
#include <dirent.h>

using namespace std;
using namespace cv;

Mat loadImage(int num){
    string filename;
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

Mat loadImageFolder(string filename){
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

/*
std::vector<std::vector<Point>> getComponents_rgb(Mat& binaryImage,Mat& input){
	boost::unordered_map<int,int> map;
	boost::unordered_map<int,Point> revmap;

	typedef boost::adjacency_list<boost::vecS,boost::vecS,boost::undirectedS> Graph;
	int num_vertices = 0;
	for( int row = 0; row < binaryImage.rows; row++ ){
    	uchar * ptr = (uchar*)binaryImage.ptr(row);
    	for (int col = 0; col < binaryImage.cols; col++ ){
            if (*ptr > 0) {
                map[row * binaryImage.cols + col] = num_vertices;
                Point p;
                p.x = col;
                p.y = row;
                revmap[num_vertices] = p;
                num_vertices++;
            }
            ptr++;
        }
    }

    Graph g(num_vertices);

    for( int row = 0; row < binaryImage.rows; row++ ){
        uchar * ptr = (uchar*)binaryImage.ptr(row);
        // Vec3b * ptrG = (Vec3b*)input.ptr(row);        
        for (int col = 0; col < binaryImage.cols; col++ ){
        	int redValue = input.at<Vec3b>(row,col)[0];
        	int greenValue = input.at<Vec3b>(row,col)[1];
        	int blueValue = inputa.at<Vec3b>(row,col)[2];
            if (*ptr > 0) {
                // check pixel to the right, right-down, down, left-down
                int this_pixel = map[row * binaryImage.cols + col];
                if (col+1 < binaryImage.cols) {
                    uchar right = binaryImage.at<uchar>(row, col+1);
                    int grayValue_right = grayImage.at<uchar>(row,col+1);
                    int grayValue_ratio = abs(grayValue_right - grayValue);
                    if ((right > 0) && (grayValue_ratio <= 12))
                        boost::add_edge(this_pixel, map.at(row * binaryImage.cols + col + 1), g);
               	}
                if (row+1 < binaryImage.rows) {
                    if (col+1 < binaryImage.cols) {
                        uchar right_down = binaryImage.at<uchar>(row+1, col+1);
                        int grayValue_rightdown = grayImage.at<uchar>(row+1,col+1);
                        int grayValue_ratio = abs(grayValue_rightdown - grayValue);
                        if ((right_down > 0) && (grayValue_ratio <= 12))
                            boost::add_edge(this_pixel, map.at((row+1) * binaryImage.cols + col + 1), g);
                    }
                    uchar down = binaryImage.at<uchar>(row+1, col);
                    int grayValue_down = grayImage.at<uchar>(row+1,col);
                    int grayValue_ratio = abs(grayValue_down - grayValue);
                    if ((down > 0) && (grayValue_ratio <= 12))
                        boost::add_edge(this_pixel, map.at((row+1) * binaryImage.cols + col), g);
                    if (col-1 >= 0) {
                        uchar left_down = binaryImage.at<uchar>(row+1, col-1);
                        int grayValue_leftdown = grayImage.at<uchar>(row+1,col-1);
                        int grayValue_ratio = abs(grayValue_leftdown - grayValue);
                        if ((left_down > 0) && (grayValue_ratio <= 12))
                            boost::add_edge(this_pixel, map.at((row+1) * binaryImage.cols + col - 1), g);
                    }
                }
            }
            ptr++;
            
        }
    }
}
*/

std::vector<vector<Point>> getComponents(Mat& binaryImage,Mat& grayImage,Mat& edgeImage){
	boost::unordered_map<int, int> map;
    boost::unordered_map<int, Point> revmap;

    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> Graph;
    int num_vertices = 0;
    // Number vertices for graph.  Associate each point with number
    for( int row = 0; row < binaryImage.rows; row++ ){
    	uchar * ptr = (uchar*)binaryImage.ptr(row);
    	const uchar* ptrE = (const uchar*)edgeImage.ptr(row);
    	for (int col = 0; col < binaryImage.cols; col++ ){
            if ((*ptr > 0) && !(*ptrE > 0)) {
                map[row * binaryImage.cols + col] = num_vertices;
                Point p;
                p.x = col;
                p.y = row;
                revmap[num_vertices] = p;
                num_vertices++;
            }
            ptr++;
            ptrE++;
        }
    }

    Graph g(num_vertices);

    for( int row = 0; row < binaryImage.rows; row++ ){
        uchar * ptr = (uchar*)binaryImage.ptr(row);
        uchar * ptrG = (uchar*)grayImage.ptr(row); 
        const uchar* ptrE = (const uchar*)edgeImage.ptr(row);       
        for (int col = 0; col < binaryImage.cols; col++ ){
        	int grayValue = grayImage.at<uchar>(row,col);
            if ((*ptr > 0) && !(*ptrE > 0)) {
                // check pixel to the right, right-down, down, left-down
                int this_pixel = map[row * binaryImage.cols + col];
                if (col+1 < binaryImage.cols) {
                    uchar right = binaryImage.at<uchar>(row, col+1);
                    uchar edge_right = edgeImage.at<uchar>(row,col+1);
                    int grayValue_right = grayImage.at<uchar>(row,col+1);
                    int grayValue_ratio = abs(grayValue_right - grayValue);
                    if ((right > 0) && (grayValue_ratio <= 12) && !(edge_right > 0))
                        boost::add_edge(this_pixel, map.at(row * binaryImage.cols + col + 1), g);
               	}
                if (row+1 < binaryImage.rows) {
                    if (col+1 < binaryImage.cols) {
                        uchar right_down = binaryImage.at<uchar>(row+1, col+1);
                        uchar edge_rightdown = edgeImage.at<uchar>(row+1,col+1);
                        int grayValue_rightdown = grayImage.at<uchar>(row+1,col+1);
                        int grayValue_ratio = abs(grayValue_rightdown - grayValue);
                        if ((right_down > 0) && (grayValue_ratio <= 12) && !(edge_rightdown > 0))
                            boost::add_edge(this_pixel, map.at((row+1) * binaryImage.cols + col + 1), g);
                    }
                    uchar down = binaryImage.at<uchar>(row+1, col);
                    uchar edge_down = edgeImage.at<uchar>(row+1,col);
                    int grayValue_down = grayImage.at<uchar>(row+1,col);
                    int grayValue_ratio = abs(grayValue_down - grayValue);
                    if ((down > 0) && (grayValue_ratio <= 12) && !(edge_down > 0))
                        boost::add_edge(this_pixel, map.at((row+1) * binaryImage.cols + col), g);
                    if (col-1 >= 0) {
                        uchar left_down = binaryImage.at<uchar>(row+1, col-1);
                        uchar edge_leftdown = edgeImage.at<uchar>(row+1,col-1);
                        int grayValue_leftdown = grayImage.at<uchar>(row+1,col-1);
                        int grayValue_ratio = abs(grayValue_leftdown - grayValue);
                        if ((left_down > 0) && (grayValue_ratio <= 12) && !(edge_leftdown > 0))
                            boost::add_edge(this_pixel, map.at((row+1) * binaryImage.cols + col - 1), g);
                    }
                }
            }
            ptr++;
            ptrE++;
        }
    }

    std::vector<int> c(num_vertices);

    int num_comp = connected_components(g, &c[0]);

    std::vector<std::vector<Point> > components;
    components.reserve(num_comp);
    std::cout << "Before filtering, " << num_comp << " components and " << num_vertices << " vertices" << std::endl;
    for (int j = 0; j < num_comp; j++) {
        std::vector<Point> tmp;
        components.push_back( tmp );
    }
    for (int j = 0; j < num_vertices; j++) {
        Point p = revmap[j];
        (components[c[j]]).push_back(p);
    }

    return components;
}

// std::vector<std::vector<Point>> filterSmall(std::vector<std::vector<Point>>& components,Mat& binaryImage){
std::vector<std::vector<Point>> filterComponents(std::vector<std::vector<Point>>& components,Mat& binaryImage){
	std::vector<std::vector<Point>> afterFilter;
	for(std::vector<std::vector<Point>>::const_iterator it1 = components.begin();it1 != components.end();it1++){
		Mat temp_convex = Mat::zeros(binaryImage.size(),CV_8UC1);

		for(std::vector<Point>::const_iterator itp = it1->begin();itp != it1->end();itp++){
			temp_convex.at<uchar>(itp->y,itp->x) = 255;
		}

		float compArea = countNonZero(temp_convex);
		std::vector<std::vector<Point>> contours;
		findContours( temp_convex, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );
		// cout<<"contours' size is: "<<contours.size()<<endl;
		// cout<<"1st vector of contours'size is: "<<contours[0].size()<<endl;
		/*
		Mat contoursImage = Mat::zeros(binaryImage.size(),CV_8UC1);
		contoursImage = ~contoursImage;
		for(std::vector<std::vector<Point>>::const_iterator it1 = contours.begin();it1 != contours.end();it1++){
			for(std::vector<Point>::const_iterator it2 = it1->begin();it2 != it1->end();it2++){
				contoursImage.at<uchar>(it2->y,it2->x) = 0;
			}
		}
		imshow("contours",contoursImage);
		waitKey(-1);
		*/
		/*
		std::vector<Rect> boundingRects;
		boundingRects.reserve(contours.size());
		for(std::vector<std::vector<Point>>::const_iterator itc = contours.begin();itc != contours.end();itc++){
			Rect bRect = boundingRect(*itc);
			boundingRects.push_back(bRect);
		}*/
		Rect bRect = boundingRect(contours[0]);

		float solidity = compArea / bRect.area();
		float width_heightRatio = max(bRect.width / bRect.height , bRect.height / bRect.width);

		if(max(bRect.width,bRect.height) < 80){
			continue;
		}
		if((solidity <= 0.4) || (width_heightRatio > 10)){
			afterFilter.push_back(*it1);
		}



	}

	return afterFilter;

}

bool infoDivergence(Mat& binaryImage,Mat& binaryImageReg,Mat& grayImage,Mat& edgeImage,std::vector<std::vector<Point>>& components1,std::vector<std::vector<Point>>& components2){
	components1 = getComponents(binaryImage,grayImage,edgeImage);
	components2 = getComponents(binaryImageReg,grayImage,edgeImage);

	return (components1.size() > components2.size()) ? true : false;
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

int main(){
	DIR* pDir;
    struct dirent* ptr; 
    if(!(pDir = opendir("/home/george/data/dotlines")))
        return -1;
    int num = 0;
    while((ptr = readdir(pDir)) != 0){
    	string filename = ptr->d_name;
        
        if(ptr->d_type == 4){
            cout<<"a dir continue..."<<endl;
            continue;
        }
        
        cout<<"Now process "<<filename<<" ..."<<endl;
        // cout<<ptr->d_type<<" "<<ptr->d_ino<<endl;
        Mat input = loadImageFolder(filename);
        Mat grayImage = convert2gray(input);
        Mat edgeImage = doCanny(grayImage);
        // Mat gaussianImage( grayImage.size(), CV_32FC1); // 元素类型为 32位float型，一个通道
        // grayImage.convertTo(gaussianImage, CV_32FC1, 1./255.);
        // GaussianBlur( gaussianImage, gaussianImage, Size(5, 5), 0); // blurs an image using a Gaussian filter
        // gaussianImage = gaussianImage * 255;

        Mat binaryImage(input.size(),CV_8UC1);
        threshold(grayImage,binaryImage,150,255,THRESH_BINARY);
        // threshold(gaussianImage,binaryImage,150,255,THRESH_BINARY);
        Mat binaryImageReg = ~binaryImage;

        std::vector<std::vector<Point>> components_binary;
        std::vector<std::vector<Point>> components_binaryReg;
        //test the 2 binary which adaptive to correct scene
        /*
        bool trick_binary = infoDivergence(binaryImage,binaryImageReg,components_binary,components_binaryReg); 
        
        std::vector<std::vector<Point>> components;
        if(trick_binary)
        	components = components_binary;
        else
        	components = components_binaryReg;
		*/
        std::vector<std::vector<Point>> components = getComponents(binaryImageReg,grayImage,edgeImage);
        // std::vector<std::vector<Point>> components = getComponents(binaryImageReg,gaussianImage);
        Mat binaryComponentsImage = Mat::zeros(input.size(),CV_8UC3);
        binaryComponentsImage = ~binaryComponentsImage;

        plotComponents_rgb(components,binaryComponentsImage);

        std::vector<std::vector<Point>> afterFilter = filterComponents(components,binaryImageReg);
        
        Mat filterComponentsImage = Mat::zeros(input.size(),CV_8UC3);
        filterComponentsImage = ~filterComponentsImage;

        plotComponents_rgb(afterFilter,filterComponentsImage);
        
        Mat combine2show(2 * input.rows + 10,3 * input.cols + 20,CV_8UC1);
        combine2show.setTo(0);
        //cvtColor(combine2show,combine2show,CV_GRAY2RGB);
        
        binaryImage.copyTo(combine2show(Rect(2 * input.cols + 20,0,binaryImage.cols,binaryImage.rows)));
        grayImage.copyTo(combine2show(Rect(input.cols + 10,0,grayImage.cols,grayImage.rows)));
        // gaussianImage.copyTo(combine2show(Rect(input.cols + 10,0,gaussianImage.cols,gaussianImage.rows)));
		cvtColor(combine2show,combine2show,CV_GRAY2RGB);
        input.copyTo(combine2show(Rect(0,0,input.cols,input.rows)));
        binaryComponentsImage.copyTo(combine2show(Rect(0,input.rows + 10,binaryComponentsImage.cols,binaryComponentsImage.rows)));
        filterComponentsImage.copyTo(combine2show(Rect(input.cols + 10,input.rows + 10,filterComponentsImage.cols,filterComponentsImage.rows)));
        
        imwrite("../data/binaryImage/" + to_string(num) + ".jpg",combine2show);
        num++;
    }
    closedir(pDir);
}