#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/unordered_map.hpp>
#include <boost/graph/floyd_warshall_shortest.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <cassert>
#include <cmath>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/features2d.hpp"  
#include <math.h>
#include <time.h>
#include <utility>
#include <algorithm>
#include <vector>
#include <string> 

#include "SWT.h"

using namespace cv;
using namespace std;
using namespace swt;

#define PI 3.14159265

bool Point2dSort (const SWTPoint2d &lhs, const SWTPoint2d &rhs) {
    return lhs.swtValue < rhs.swtValue;
}

void SWT::StrokeWidthTransform(const Mat& edgeImage,Mat& gradientX,Mat& gradientY,bool dark_on_light,Mat& SWTImage,std::vector<Ray> & rays){
    // First pass
    float prec = .05;
    for( int row = 0; row < edgeImage.rows; row++ )
    {
        const uchar* ptr = (const uchar*)edgeImage.ptr(row);
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
                            if (acos(G_x * -G_xt + G_y * -G_yt) < PI/2.0 ) // yzhou 17.08.24 change angel, the origin is PI/2.0
                            {
                                float length = sqrt( ((float)r.q.x - (float)r.p.x)*((float)r.q.x - (float)r.p.x) + ((float)r.q.y - (float)r.p.y)*((float)r.q.y - (float)r.p.y));
                                
                                int threshold_L = 0.2 * (edgeImage.cols < edgeImage.rows ? edgeImage.cols : edgeImage.rows);
                                if (length > threshold_L){
                                    r.points = points;
                                    break;
                                }

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
}

void SWT::SWTMedianFilter (Mat& SWTImage, vector<Ray> & rays){
    for (auto& rit : rays) {
        for (auto& pit : rit.points) {
            pit.swtValue = SWTImage.at<float>(pit.y, pit.x);
        }
        std::sort(rit.points.begin(), rit.points.end(), &Point2dSort);
        float median = (rit.points[rit.points.size()/2]).swtValue;
        for (auto& pit : rit.points) {
            SWTImage.at<float>(pit.y, pit.x) = std::min(pit.swtValue, median);
        }
    }
}

std::vector<std::vector<SWTPoint2d>> SWT::findLegallyConnectedComponents (cv::Mat& SWTImage, std::vector<Ray> & rays){
        boost::unordered_map<int, int> map;
        boost::unordered_map<int, SWTPoint2d> revmap;

        typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> Graph;
        int num_vertices = 0;
        // Number vertices for graph.  Associate each point with number
        for( int row = 0; row < SWTImage.rows; row++ ){
            float * ptr = (float*)SWTImage.ptr(row);
            for (int col = 0; col < SWTImage.cols; col++ ){
                if (*ptr > 0) {
                    map[row * SWTImage.cols + col] = num_vertices;
                    SWTPoint2d p;
                    p.x = col;
                    p.y = row;
                    revmap[num_vertices] = p;
                    num_vertices++;
                }
                ptr++;
            }
        }

        Graph g(num_vertices);

        for( int row = 0; row < SWTImage.rows; row++ ){
            float * ptr = (float*)SWTImage.ptr(row);
            for (int col = 0; col < SWTImage.cols; col++ ){
                if (*ptr > 0) {
                    // check pixel to the right, right-down, down, left-down
                    int this_pixel = map[row * SWTImage.cols + col];
                    // yzhou 17.08.24. The original ratio is 3.0.
                    float ratio_swtV = 2.0;
                    if (col+1 < SWTImage.cols) {
                        float right = SWTImage.at<float>(row, col+1);
                        if (right > 0 && ((*ptr)/right <= ratio_swtV || right/(*ptr) <= ratio_swtV))
                           boost::add_edge(this_pixel, map.at(row * SWTImage.cols + col + 1), g);
                    }
                    if (row+1 < SWTImage.rows) {
                        if (col+1 < SWTImage.cols) {
                            float right_down = SWTImage.at<float>(row+1, col+1);
                            if (right_down > 0 && ((*ptr)/right_down <= ratio_swtV || right_down/(*ptr) <= ratio_swtV))
                                boost::add_edge(this_pixel, map.at((row+1) * SWTImage.cols + col + 1), g);
                        }
                        float down = SWTImage.at<float>(row+1, col);
                        if (down > 0 && ((*ptr)/down <= ratio_swtV || down/(*ptr) <= ratio_swtV))
                            boost::add_edge(this_pixel, map.at((row+1) * SWTImage.cols + col), g);
                        if (col-1 >= 0) {
                            float left_down = SWTImage.at<float>(row+1, col-1);
                            if (left_down > 0 && ((*ptr)/left_down <= ratio_swtV || left_down/(*ptr) <= ratio_swtV))
                                boost::add_edge(this_pixel, map.at((row+1) * SWTImage.cols + col - 1), g);
                        }
                    }
                }
                ptr++;
            }
        }

        std::vector<int> c(num_vertices);

        int num_comp = connected_components(g, &c[0]);

        std::vector<std::vector<SWTPoint2d> > components;
        components.reserve(num_comp);
        std::cout << "Before filtering, " << num_comp << " components and " << num_vertices << " vertices" << std::endl;
        for (int j = 0; j < num_comp; j++) {
            std::vector<SWTPoint2d> tmp;
            components.push_back( tmp );
        }
        for (int j = 0; j < num_vertices; j++) {
            SWTPoint2d p = revmap[j];
            (components[c[j]]).push_back(p);
        }

        return components;
}

void componentStats(Mat& SWTImage,const std::vector<SWTPoint2d> & component,float & mean, float & variance,int & minx, int & miny, int & maxx, int & maxy)
{
        std::vector<float> temp;
        temp.reserve(component.size());
        mean = 0;
        variance = 0;
        minx = 1000000;
        miny = 1000000;
        maxx = 0;
        maxy = 0;
        for (std::vector<SWTPoint2d>::const_iterator it = component.begin(); it != component.end(); it++) {
                float t = SWTImage.at<float>(it->y, it->x);
                mean += t;
                temp.push_back(t);
                miny = std::min(miny,it->y);
                minx = std::min(minx,it->x);
                maxy = std::max(maxy,it->y);
                maxx = std::max(maxx,it->x);
        }
        mean = mean / ((float)component.size());
        for (std::vector<float>::const_iterator it = temp.begin(); it != temp.end(); it++) {
            variance += (*it - mean) * (*it - mean);
        }
        variance = variance / ((float)component.size());
        std::sort(temp.begin(),temp.end());
}

void filterRay(Mat& SWTImage,std::vector<SWTPoint2d>& components,std::vector<SWTPoint2d>& temp){
    std::vector<SWTPoint2d> afterFilter;
    for(std::vector<SWTPoint2d>::const_iterator it = components.begin();it != components.end();it++){
        if(SWTImage.at<float>(it->y,it->x) <= 20){
            afterFilter.push_back(*it);
        }
    }
    // return afterFilter;
}

//the filter conditions just for detect spot lines!
std::vector<std::vector<SWTPoint2d>> SWT::filterComponents(Mat& SWTImage,std::vector<std::vector<SWTPoint2d> > & components){
        std::vector<std::vector<SWTPoint2d>> validComponents;
        validComponents.reserve(components.size());

        for (std::vector<std::vector<SWTPoint2d> >::iterator it = components.begin(); it != components.end();it++) 
        {
            // compute the stroke width mean, variance, median
            float mean, variance;
            int minx, miny, maxx, maxy;
            /*
            componentStats(SWTImage, (*it), mean, variance, minx, miny, maxx, maxy);
            
            if(mean > 40){
                continue;
            }
            
            if (variance > 2 * mean) {   
                  continue;
            }
            */
            //std::vector<SWTPoint2d> temp = *it;
            
            std::vector<SWTPoint2d> temp;
            temp.reserve(it->size());
            for(std::vector<SWTPoint2d>::const_iterator it2 = it->begin();it2 != it->end();it2++){
                float swtValue = SWTImage.at<float>(it2->y,it2->x);
                if(swtValue <= 30){
                    temp.push_back(*it2);
                }
            }
            
            // filterRay(SWTImage,(*it),temp);
            /*
            float length = (float)(maxx-minx+1);
            float width = (float)(maxy-miny+1);
            */
            // check font height
            //this condition can filter the marked line so abandon it!
            /*
            if ((width > 300) || ((width < 10) && (length < 10)) || (length > 300) ) {
                continue;
            }
            */
            /*
            float area = length * width;
            float rminx = (float)minx;
            float rmaxx = (float)maxx;
            float rminy = (float)miny;
            float rmaxy = (float)maxy;
            // compute the rotated bounding box
            float increment = 1./36.;
            for (float theta = increment * PI; theta<PI/2.0; theta += increment * PI) {
                float xmin,xmax,ymin,ymax,xtemp,ytemp,ltemp,wtemp;
                    xmin = 1000000;
                    ymin = 1000000;
                    xmax = 0;
                    ymax = 0;
                for (unsigned int i = 0; i < (*it).size(); i++) {
                    xtemp = (*it)[i].x * cos(theta) + (*it)[i].y * -sin(theta);
                    ytemp = (*it)[i].x * sin(theta) + (*it)[i].y * cos(theta);
                    xmin = std::min(xtemp,xmin);
                    xmax = std::max(xtemp,xmax);
                    ymin = std::min(ytemp,ymin);
                    ymax = std::max(ytemp,ymax);
                }
                ltemp = xmax - xmin + 1;
                wtemp = ymax - ymin + 1;
                if (ltemp*wtemp < area) {
                    area = ltemp*wtemp;
                    length = ltemp;
                    width = wtemp;
                }
            }
            */
            /*
            if (length/width < 1./15. || length/width > 15.) {
                continue;
            }*/
            //float ratio_text = (float)(*it).size() / (float)area;

            //this conditon can filter the marked line too
            /*
            if ((ratio_text < 0.2) || (ratio_text > 0.8)){ //
                continue;
            }*/
            // create graph representing components
            //validComponents.push_back(temp);
            //validComponents.push_back(temp);
            validComponents.push_back(*it);
        }

        std::vector<std::vector<SWTPoint2d > > tempComp;
        tempComp.reserve(validComponents.size());

        for (unsigned int i = 0; i < validComponents.size(); i++) {
            tempComp.push_back(validComponents[i]);
        }
        validComponents = tempComp;
        
        validComponents.reserve(tempComp.size());

        std::cout << "After filtering " << validComponents.size() << " components" << std::endl;
        return validComponents;
}