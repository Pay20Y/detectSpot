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

std::vector<vector<Point>> getComponents(Mat binaryImage){
	boost::unordered_map<int, int> map;
    boost::unordered_map<int, Point> revmap;

    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> Graph;
    int num_vertices = 0;
    // Number vertices for graph.  Associate each point with number
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
        for (int col = 0; col < binaryImage.cols; col++ ){
            if (*ptr > 0) {
                // check pixel to the right, right-down, down, left-down
                int this_pixel = map[row * binaryImage.cols + col];
                if (col+1 < binaryImage.cols) {
                    uchar right = binaryImage.at<uchar>(row, col+1);
                    if (right > 0)
                        boost::add_edge(this_pixel, map.at(row * binaryImage.cols + col + 1), g);
               	}
                if (row+1 < binaryImage.rows) {
                    if (col+1 < binaryImage.cols) {
                        uchar right_down = binaryImage.at<uchar>(row+1, col+1);
                        if (right_down > 0)
                            boost::add_edge(this_pixel, map.at((row+1) * binaryImage.cols + col + 1), g);
                    }
                    uchar down = binaryImage.at<uchar>(row+1, col);
                    if (down > 0)
                        boost::add_edge(this_pixel, map.at((row+1) * binaryImage.cols + col), g);
                    if (col-1 >= 0) {
                        uchar left_down = binaryImage.at<uchar>(row+1, col-1);
                        if (left_down > 0)
                            boost::add_edge(this_pixel, map.at((row+1) * binaryImage.cols + col - 1), g);
                    }
                }
            }
            ptr++;
            
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

        Mat binaryImage(input.size(),CV_8UC1);
        threshold(grayImage,binaryImage,150,255,THRESH_BINARY);

        std::vector<std::vector<Point>> components = getComponents(binaryImage);
        Mat components2show(input.size(),CV_8UC3);

        plotComponents_rgb(components,components2show);
        imshow("components",components2show);
        waitKey(-1);
        Mat combine2show(input.rows,2 * input.cols + 10,CV_8UC1);
        cvtColor(combine2show,combine2show,CV_GRAY2RGB);
        input.copyTo(combine2show(Rect(0,0,input.cols,input.rows)));
        cvtColor(combine2show,combine2show,CV_RGB2GRAY);
        binaryImage.copyTo(combine2show(Rect(input.cols + 10,0,binaryImage.cols,binaryImage.rows)));
        imwrite("../data/binaryImage/" + to_string(num) + ".jpg",combine2show);
        num++;
    }
    closedir(pDir);
}