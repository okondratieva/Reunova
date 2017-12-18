#include "opencv2\opencv.hpp"
#include "opencv\highgui.h"
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;
int borders(int value, int min, int max)
{
	return (value < min) ? min : ((value > max) ? max : value);
}
int main()
{
	Mat src=imread(/*"lena1.png"*/"original.jpg"/*, IMREAD_GRAYSCALE*/);
	Mat gray;
	Mat grad;
	Mat edges;
	Mat dist;
	Mat result = Mat(src.rows, src.cols, src.type());
	if (!src.data)
	{
		cout << "Image loading error!" << endl;
		waitKey();
		return -1;
	}
	//original image
	namedWindow("Source image", WINDOW_NORMAL);
	imshow("Source image", src);


	//grayscale image
	//use CV_BGR2GRAY because of imread has BGR default channel order in case of color images
	cvtColor(src, gray, CV_BGR2GRAY);
	namedWindow("Grayscale image", WINDOW_NORMAL);
	imshow("Grayscale image", gray);

	//gradients
	Mat gx, gy;
	Mat abs_gx, abs_gy;
	//reduce the noise
	GaussianBlur(gray, gray, Size(3, 3), 0, 0, BORDER_DEFAULT);
	//derivatives in x and y directions
	Sobel(gray, gx, CV_16S, 1, 0);
	Sobel(gray, gy, CV_16S, 0, 1);
	//convert results back to CV_8U
	convertScaleAbs(gx, abs_gx);
	convertScaleAbs(gy, abs_gy);
	//approximate the gradient by adding both directional gradients 
	addWeighted(abs_gx, 0.5, abs_gy, 0.5, 0, grad);
	namedWindow("Gradients", WINDOW_NORMAL);
	imshow("Gradients", grad);
	
	//Canny edge detection
	Canny(gray, edges,1,60);
	namedWindow("Canny", WINDOW_NORMAL);
	imshow("Canny", edges);

	//distance field
	//inverse canny edges for correct calculation the distance field
	edges = 1 - edges;
	distanceTransform(edges, dist, CV_DIST_L2, 3);


	Mat integral_img;
	int k=5;
	integral(src, integral_img, CV_32F);
	for(int i = 0;  i < src.cols; i++)
		for (int j = 0; j < src.rows; j++)
		{
			int tool = k*dist.at<float>(Point(i, j));
			if (tool >= 1)
			{
				//the function "borders" is needed to avoid out of bounds of the image
				int n = ((borders(i + tool, 0, integral_img.cols - 1) - borders(i - tool, 0, integral_img.cols - 1)) *
					(borders(j + tool, 0, integral_img.rows - 1) - borders(j - tool, 0, integral_img.rows - 1)));
				//apply of convolution
				result.at<Vec3b>(Point(i, j)) = (
					  integral_img.at<Vec3f>(Point(borders(i - tool, 0, integral_img.cols - 1), borders(j - tool, 0, integral_img.rows - 1)))
					+ integral_img.at<Vec3f>(Point(borders(i + tool, 0, integral_img.cols - 1), borders(j + tool, 0, integral_img.rows - 1)))
					- integral_img.at<Vec3f>(Point(borders(i - tool, 0, integral_img.cols - 1), borders(j + tool, 0, integral_img.rows - 1)))
					- integral_img.at<Vec3f>(Point(borders(i + tool, 0, integral_img.cols - 1), borders(j - tool, 0, integral_img.rows - 1)))
					) / n;
			}
			else
				result.at<Vec3b>(Point(i, j)) = src.at<Vec3b>(Point(i, j));
		}

	//Result
	namedWindow("Result", WINDOW_NORMAL);
	imshow("Result", result);
		
	waitKey();
	
	return 0;
}