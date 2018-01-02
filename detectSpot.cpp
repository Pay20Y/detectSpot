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

using namespace cv;
using namespace std;

#define PI 3.14159265

struct SWTPoint2d {
    int x;
    int y;
    float SWT;
};
struct Ray {
        SWTPoint2d p;
        SWTPoint2d q;
        std::vector<SWTPoint2d> points;
};

Mat loadImage(int num){
    string filename;
    if(num < 58){
        filename = to_string(num) + "_1.jpg"; //marked sample
    }else{
        filename = to_string(num) + "_0.jpg"; //unmarked sample
    }
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

void doGradient(Mat& grayImage,Mat& gradientX,Mat& gradientY){
	Mat gaussianImage( grayImage.size(), CV_32FC1); // 元素类型为 32位float型，一个通道
    grayImage.convertTo(gaussianImage, CV_32FC1, 1./255.);
    // convertTo 是改变元素的
    GaussianBlur( gaussianImage, gaussianImage, Size(5, 5), 0); // blurs an image using a Gaussian filter
    Scharr(gaussianImage, gradientX, -1, 1, 0); // calculates the x- image derivative 
    Scharr(gaussianImage, gradientY, -1, 0, 1);  // calculates the y- image derivative
    GaussianBlur(gradientX, gradientX, Size(3, 3), 0);
    GaussianBlur(gradientY, gradientY, Size(3, 3), 0);
}

void strokeWidthTransform (const Mat& edgeImage,Mat& gradientX,Mat& gradientY,bool dark_on_light,Mat& SWTImage,std::vector<Ray> & rays,std::vector<Ray> & raysOut) {
    // First pass
    float prec = .05;
    //Mat visitedMat(edgeImage.size(),CV_8UC1);
    //visitedMat.setTo(-1);
    for( int row = 0; row < edgeImage.rows; row++ )
    {
        const uchar* ptr = (const uchar*)edgeImage.ptr(row);
        //cout<<"ptr is: "<<(*ptr)<<"\n";
        for ( int col = 0; col < edgeImage.cols; col++ )
        {
            if (*ptr > 0) 
            {   // 有边缘的地方
                Ray r;
                //Ray rB;
                SWTPoint2d p;
                p.x = col;
                p.y = row;
                r.p = p;
                //rB.p = p;
                std::vector<SWTPoint2d> points;
                points.push_back(p);
                //visitedMat.at<float>(row,col) = 1;
                float curX = (float)col + 0.5;  // 为什么要加0.5？？？？？
                float curY = (float)row + 0.5;
                int curPixX = col;
                int curPixY = row;
                float G_x = gradientX.at<float>(row, col);
                float G_y = gradientY.at<float>(row, col);
                // normalize gradient
                float mag = sqrt( (G_x * G_x) + (G_y * G_y) ); // 梯度的大小
                //cout<<"gradient mag is: "<<mag<<" ";
                if (dark_on_light){    // 对x,y方向上的梯度进行归一化
                    G_x = -G_x/mag;
                    G_y = -G_y/mag;
                } else {
                    G_x = G_x/mag;
                    G_y = G_y/mag;
                }
                while (true)   // move on 
                {   
                    //if(mag >= 0.1){
                    curX += G_x*prec;   // 这个prec应该是沿着梯度方向前进的距离
                    curY += G_y*prec;
                    if ((int)(floor(curX)) != curPixX || (int)(floor(curY)) != curPixY) 
                    {   
                        /*
                        if(mag <= 2){
                            break;
                        }*/
                        // double floor(double x) : return the largest integral value that is not greater than x
                        curPixX = (int)(floor(curX));
                        curPixY = (int)(floor(curY));
                        // check if pixel is outside boundary of image
                        if (curPixX < 0 || (curPixX >= SWTImage.cols) || curPixY < 0 || (curPixY >= SWTImage.rows)) {
                            break;
                        }
                        SWTPoint2d pnew;
                        pnew.x = curPixX;
                        pnew.y = curPixY;
                        points.push_back(pnew);
                        //visitedMat.at<float>(row,col) = 1;
                        if (edgeImage.at<uchar>(curPixY, curPixX) > 0) //on edge
                        {
                            r.q = pnew;
                            // dot product
                            float G_xt = gradientX.at<float>(curPixY,curPixX);
                            float G_yt = gradientY.at<float>(curPixY,curPixX);
                            mag = sqrt( (G_xt * G_xt) + (G_yt * G_yt) );
                            if (dark_on_light) {
                                G_xt = -G_xt / mag;
                                G_yt = -G_yt / mag;
                            } else {
                                G_xt = G_xt / mag;
                                G_yt = G_yt / mag;

                            }
                            // double acos(double x) return the principal value of the arc cosine of x, expressed in radians
                            // acos就是反余弦
                            //if (acos(G_x * -G_xt + G_y * -G_yt) < PI/4.0 )   //zQiao 17.10.27
                            if (acos(G_x * -G_xt + G_y * -G_yt) < PI/2.0 ) // yzhou 17.08.24 change angel, the origin is PI/2.0
                            {
                                float length = sqrt( ((float)r.q.x - (float)r.p.x)*((float)r.q.x - (float)r.p.x) + ((float)r.q.y - (float)r.p.y)*((float)r.q.y - (float)r.p.y));
                                
                                // yzhou 17.08.22 begin. If the length of the ray is too big, it is be removed.
                                int threshold_L = 0.2 * (edgeImage.cols < edgeImage.rows ? edgeImage.cols : edgeImage.rows);
                                if (length > threshold_L){
                                    r.points = points;
                                    raysOut.push_back(r);
                                    break;
                                }
                                // yzhou 17.08.22 end

                                for (std::vector<SWTPoint2d>::iterator pit = points.begin(); pit != points.end(); pit++) 
                                {
                                    if (SWTImage.at<float>(pit->y, pit->x) < 0) 
                                    {
                                        SWTImage.at<float>(pit->y, pit->x) = length;
                                    } 
                                    else 
                                    {
                                        SWTImage.at<float>(pit->y, pit->x) = std::min(length, SWTImage.at<float>(pit->y, pit->x));
                                    }
                                } // end for pit
                                r.points = points;
                                rays.push_back(r);
                            } // end if acos
                            break;
                        }// end if edgeImage
                    } // end curX & curPixX
                } // end while
            } // end if ptr
            ptr++;
        } // end for col
    } // end for row
    cout<<"rays' size is: "<<rays.size()<<"\n";
}// end func SWT

void normalizeImage (const Mat& input, Mat& output) {
    assert ( input.depth() == CV_32F );
    assert ( input.channels() == 1 );
    assert ( output.depth() == CV_32F );
    assert ( output.channels() == 1 );

    float maxVal = 0;
    float minVal = 1e100;
    for ( int row = 0; row < input.rows; row++ ){
        const float* ptr = (const float*)input.ptr(row);
        for ( int col = 0; col < input.cols; col++ ){
            if (*ptr < 0) { }
            else {
                maxVal = std::max(*ptr, maxVal);
                minVal = std::min(*ptr, minVal);
            }
            ptr++;
        }
    }

    float difference = maxVal - minVal;
    for ( int row = 0; row < input.rows; row++ ) {
        const float* ptrin = (const float*)input.ptr(row);
        float* ptrout = (float*)output.ptr(row);
        for ( int col = 0; col < input.cols; col++ ) {
            if (*ptrin < 0) {
                *ptrout = 1;
            } else {
                *ptrout = ((*ptrin) - minVal)/difference;
            }
            ptrout++;
            ptrin++;
        }
    }
}

bool Point2dSort (const SWTPoint2d &lhs, const SWTPoint2d &rhs) {
    return lhs.SWT < rhs.SWT;
}
void SWTMedianFilter (Mat& SWTImage, std::vector<Ray> & rays) {
    for (auto& rit : rays) {
        for (auto& pit : rit.points) {
            pit.SWT = SWTImage.at<float>(pit.y, pit.x);
        }
        std::sort(rit.points.begin(), rit.points.end(), &Point2dSort);
        float median = (rit.points[rit.points.size()/2]).SWT;
        for (auto& pit : rit.points) {
            SWTImage.at<float>(pit.y, pit.x) = std::min(pit.SWT, median);
        }
    }
}

std::vector<std::vector<Point>> findComponent(Mat& edgeImage,Mat& binaryImage){
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
                    uchar rightc = edgeImage.at<uchar>(row,col+1);
                    if ((right > 0) && !(rightc > 0))
                        boost::add_edge(this_pixel, map.at(row * binaryImage.cols + col + 1), g);
                }
                if (row+1 < binaryImage.rows) {
                    if (col+1 < binaryImage.cols) {
                        uchar right_down = binaryImage.at<uchar>(row+1, col+1);
                        uchar right_downc = edgeImage.at<uchar>(row+1, col+1);
                        if (right_down > 0 && !(right_downc > 0))
                            boost::add_edge(this_pixel, map.at((row+1) * binaryImage.cols + col + 1), g);
                    }
                    uchar down = binaryImage.at<uchar>(row+1, col);
                    uchar downc = edgeImage.at<uchar>(row+1, col);
                    if (down > 0 && !(downc > 0))
                        boost::add_edge(this_pixel, map.at((row+1) * binaryImage.cols + col), g);
                    if (col-1 >= 0) {
                        uchar left_down = binaryImage.at<uchar>(row+1, col-1);
                        uchar left_downc = edgeImage.at<uchar>(row+1, col-1);
                        if (left_down > 0 && !(left_downc > 0))
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

void plotComponents(std::vector<std::vector<Point>> components,Mat& showComponents){
    int count = 0;
    for(std::vector<std::vector<Point>>::const_iterator it1 = components.begin();it1 != components.end();it1++){
        for(std::vector<Point>::const_iterator it2 = it1->begin();it2 != it1->end();it2++){
            if(count % 3 == 0){
                showComponents.at<Vec3f>(it2->y,it2->x)[0] = 255;
                showComponents.at<Vec3f>(it2->y,it2->x)[1] = 0;
                showComponents.at<Vec3f>(it2->y,it2->x)[2] = 0;
            }else if(count % 3 == 1){
                showComponents.at<Vec3f>(it2->y,it2->x)[0] = 0;
                showComponents.at<Vec3f>(it2->y,it2->x)[1] = 255;
                showComponents.at<Vec3f>(it2->y,it2->x)[2] = 0;
            }else{
                showComponents.at<Vec3f>(it2->y,it2->x)[0] = 0;
                showComponents.at<Vec3f>(it2->y,it2->x)[1] = 0;
                showComponents.at<Vec3f>(it2->y,it2->x)[2] = 255;
            }
        }
        count++;
    }
    imshow("components",showComponents);
    waitKey(-1);
}

void plotComponents_black(std::vector<std::vector<Point>>& components,Mat& showComponents){
    for(std::vector<std::vector<Point>>::const_iterator it1 = components.begin();it1 != components.end();it1++){
        for(std::vector<Point>::const_iterator it2 = it1->begin();it2 != it1->end();it2++){
            showComponents.at<uchar>(it2->y,it2->x) = 255;
        }
    }
}

void rectComponents(Mat& binaryImage,Mat& showComponents,Mat& input){
    Mat temp = input.clone();
    int count = 0;
    std::vector<std::vector<cv::Point> > plate_contours;  
    findContours(binaryImage, plate_contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    for(std::vector<std::vector<Point>>::const_iterator it1 = plate_contours.begin();it1 != plate_contours.end();it1++){
        cv::Rect rect = cv::boundingRect(*it1);  
        if(count % 3 == 0){
            rectangle(temp,rect.tl(),rect.br(),cv::Scalar(255,0,0),1,1,0);
        }else if(count % 3 == 1){
            rectangle(temp,rect.tl(),rect.br(),cv::Scalar(0,255,0),1,1,0);
        }else{
            rectangle(temp,rect.tl(),rect.br(),cv::Scalar(0,0,255),1,1,0);    
        } 
        count++;
    }
    imshow("components",temp);
    waitKey(-1);
} 

void tryMSER(Mat& grayImage,Mat&mserImage){
    std::vector<std::vector<Point> > regContours;
    std::vector<Rect> bboxes;  
    Ptr<MSER> mesr = MSER::create(2, 10, 5000, 0.5, 0.2);
    mesr->detectRegions(grayImage, regContours, bboxes);
      
    for (int i = (int)regContours.size() - 1; i >= 0; i--){  
        // 根据检测区域点生成mser+结果  
        const std::vector<Point>& r = regContours[i];  
        for (int j = 0; j < (int)r.size(); j++){  
            Point pt = r[j];  
            mserImage.at<unsigned char>(pt) = 255;  
        }  
    }  
    // imshow("mser",mserImage);
    // waitKey(-1);

}

void tryHull(std::vector<std::vector<Point>>& components,Mat& input,std::vector<std::vector<Point>>& hullfilter){
    for(std::vector<std::vector<Point>>::const_iterator it1 = components.begin();it1 != components.end();it1++){
        Mat temp_convex = Mat::zeros(input.size(),CV_8UC1);

        float solidity = 0.0;
        for(std::vector<Point>::const_iterator it2 = it1->begin();it2 != it1->end(); it2++){
            temp_convex.at<unsigned char>(it2->y,it2->x) = 255;
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

        //cout<<"-->solidity is: "<<solidity<<endl;
        if((solidity < 0.4) && (bRect.width > 60) && (bRect.height > 120)){
            hullfilter.push_back(*it1);
        }else if((max(bRect.width,bRect.height) / min(bRect.width,bRect.height) > 3) && (bRect.width > 60) && (bRect.height > 120)){
            hullfilter.push_back(*it1);
        }
    }
}

int main(){
    int got = 0;
    for(int num = 3; num < 4;num++){
        cout<<"Now process the No. "<<num<<" image..."<<endl;
    	Mat input = loadImage(num);
    	Mat grayImage = convert2gray(input);
        Mat edgeImage = doCanny(grayImage);
        // imshow("Before",grayImage);
        // waitKey(-1);

        /*
        Mat binaryImage(input.size(),CV_8UC1);
        threshold(grayImage, binaryImage, 150, 255, CV_THRESH_BINARY); //the treshold maybe change again
        binaryImage = ~binaryImage;

        imshow("2",binaryImage);
        waitKey(-1);
        //equalizeHist(grayImage,grayImage);
        
       
        // imshow("edge",edgeImage);
        // waitKey(-1);

        std::vector<std::vector<Point>> components = findComponent(edgeImage,binaryImage);
        
        std::vector<std::vector<Point>> hullfilter;
        tryHull(components,input,hullfilter);
        Mat showComponents = Mat::zeros(input.size(),CV_8UC1);
        plotComponents_black(hullfilter,showComponents);
        cout<<"=====Result show here======"<<endl;
        if (countNonZero(showComponents) != 0){ //means there are some lines
            if(num < 58){
                got++;
            }
            cout<<"image-"+to_string(num)+" is marked"<<endl;
            imwrite("data/marked/"+to_string(num)+".jpg",showComponents);
        }else{
            if(num >= 58){
                got++;
            }
            cout<<"image-"+to_string(num)+" is unmarked"<<endl;
            imwrite("data/marked/"+to_string(num)+".jpg",showComponents);
        }
        
        // imshow("hull",showComponents);
        // waitKey(-1);
    Mat distanceMap(input.size(),CV_32FC1);
    distanceTransform(binaryImage,distanceMap,CV_DIST_L2,3);
   
        // Mat mserImage = Mat::zeros(input.size(),CV_8UC1);
        // tryMSER(grayImage,mserImage);
        
        // Mat inject = mserImage & binaryImage;
        // imwrite("data/mser/"+to_string(num)+".jpg",inject);
    }
    float rotio = got/71.0;
    rotio = rotio * 100;
    cout<<"accurance of the marked sample is "<<rotio<<"%"<<endl;
    //Mat showComponents = Mat::zeros(input.size(),CV_8UC3);
    */
    /*
    std::vector<float> maxDiss;
    maxDiss.reserve(components.size());
    for(std::vector<std::vector<Point>>::const_iterator it1 = components.begin();it1 != components.end();it1++){
        float maxDis = 0.0;
        for(std::vector<Point>::const_iterator it2 = it1->begin();it2 != it1->end();it2++){
            if(distanceMap.at<float>(it2->y,it2->x) > maxDis)
                maxDis = distanceMap.at<float>(it2->y,it2->x);
        }
        //cout<<"maxDis is: "<<maxDis<<endl;
        maxDiss.push_back(maxDis);
    }
    */
    /*
    Mat boneImage = Mat::zeros(input.size(),CV_8UC1);
    int index = 0;
    for(std::vector<std::vector<Point>>::const_iterator it3 = components.begin();it3 != components.end();it3++){
        for(std::vector<Point>::const_iterator it4 = it3->begin();it4 != it3->end();it4++){
            float dis_lt = distanceMap.at<float>(it4->y-1,it4->x-1);
            float dis_t  = distanceMap.at<float>(it4->y-1,it4->x);
            float dis_rt = distanceMap.at<float>(it4->y-1,it4->x+1);
            float dis_l  = distanceMap.at<float>(it4->y,it4->x-1);
            float dis_r  = distanceMap.at<float>(it4->y,it4->x+1);
            float dis_lb = distanceMap.at<float>(it4->y+1,it4->x-1);
            float dis_b  = distanceMap.at<float>(it4->y+1,it4->x); 
            float dis_rb = distanceMap.at<float>(it4->y+1,it4->x+1);
            float dis_now = distanceMap.at<float>(it4->y,it4->x);
            if((dis_now >= dis_lt) && (dis_now >= dis_t) && (dis_now >= dis_rt) && (dis_now >= dis_l) && (dis_now >= dis_r) && (dis_now >= dis_lb) && (dis_now >= dis_b) && (dis_now >= dis_rb)){
                boneImage.at<uchar>(it4->y,it4->x) = 255;
            }
        }
        index++;
    }

    imshow("bone",boneImage);
    waitKey(-1);
    */
    //showComponents = ~showComponents;
    //rectComponents(binaryImage,showComponents,input);
    //plotComponents(components,showComponents);

    	Mat gradientX( input.size(), CV_32FC1 );
        Mat gradientY( input.size(), CV_32FC1 );
        Mat gaussianImage( grayImage.size(), CV_32FC1); // 元素类型为 32位float型，一个通道
        grayImage.convertTo(gaussianImage, CV_32FC1, 1./255.);
        // convertTo 是改变元素的
        GaussianBlur( gaussianImage, gaussianImage, Size(5, 5), 0); // blurs an image using a Gaussian filter
        Scharr(gaussianImage, gradientX, -1, 1, 0); // calculates the x- image derivative 
        Scharr(gaussianImage, gradientY, -1, 0, 1);  // calculates the y- image derivative
        GaussianBlur(gradientX, gradientX, Size(3, 3), 0);
        GaussianBlur(gradientY, gradientY, Size(3, 3), 0);
        //doGradient(grayImage,gradientX,gradientX);

        std::vector<Ray> rays;
        std::vector<Ray> raysOut;

        Mat SWTImage( input.size(), CV_32FC1 );
        for( int row = 0; row < input.rows; row++ ){
            float* ptr = (float*)SWTImage.ptr(row);   // cv::Mat::ptr() : return a pointer to the specified matrix row
            for ( int col = 0; col < input.cols; col++ ){
                *ptr++ = -1;
            }
        }   // this for 
        strokeWidthTransform(edgeImage,gradientX,gradientY,1,SWTImage,rays,raysOut);
        SWTMedianFilter ( SWTImage, rays );
        //SWTWithDis(MSER_result,SWTImage);
        Mat output2( input.size(), CV_32FC1 );
        normalizeImage (SWTImage, output2);
        Mat saveSWT( input.size(), CV_8UC1 );
        output2.convertTo(saveSWT, CV_8UC1, 255); // gray image
        imshow("swt",saveSWT);
        waitKey(-1);

    }
}