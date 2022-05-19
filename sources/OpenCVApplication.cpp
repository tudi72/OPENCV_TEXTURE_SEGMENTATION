// OpenCVApplication.cpp : Defines the entry point for the console application.
//
#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif
#include <iostream>
#include <cstdlib>
#include <random>
default_random_engine gen;
uniform_int_distribution<int> d(0, 255);
#include <cmath>
#include "stdafx.h"
#include "common.h"
#include <cmath>
using namespace std;


typedef struct {
	Mat_<uchar> dilateImg;
	Mat_<uchar> erodeImg;
	Mat_<uchar> openImg;
	Mat_<uchar> closeImg;
}Morphology;

bool isInside(Mat img, int i, int j) {
	if (i >= 0 && i < img.rows && j >= 0 && j < img.cols)
		return true;
	else return false;
}

pair<int, int>* compute4Neighbours(Mat_<uchar> img, pair<int, int> q) {

	int di[4] = { -1,0,1,0 };
	int dj[4] = { 0,-1,0,1 };
	int i = q.first;
	int j = q.second;
	pair<int, int> neighbours[4];
	for (int k = 0; k < 4; k++)
		neighbours[k] = { i + di[k], j + dj[k] };
	return neighbours;
}

Morphology computeMorphology(Mat_<uchar> img, int label) {

	Morphology m;
	//Morphology* m = (Morphology*)malloc(sizeof(Morphology));
	Mat_<uchar> dilateImg = img.clone();
	Mat_<uchar> erodeImg = img.clone();
	Mat_<uchar> openImg = img.clone();
	Mat_<uchar> closeImg = img.clone();


	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {

			pair<int, int>* structuralElement = compute4Neighbours(img, { i,j });
			uchar minimum = 0, maximum = 255;
			for (int k = 0; k < 4; k++) {
				int ii = structuralElement[k].first - 1;
				int jj = structuralElement[k].second - 1;

				if (isInside(img, ii, jj) && minimum < img(ii, jj)) minimum = img(ii, jj);
				if (isInside(img, ii, jj) && maximum > img(ii, jj)) maximum = img(ii, jj);
			}
			dilateImg(i, j) = maximum;
			erodeImg(i, j) = minimum;
			openImg(i, j) = maximum;
			closeImg(i, j) = minimum;
		}
	}

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {

			pair<int, int>* structuralElement = compute4Neighbours(img, { i,j });
			uchar minimum = 0, maximum = 255;
			for (int k = 0; k < 4; k++) {
				int ii = structuralElement[k].first - 1;
				int jj = structuralElement[k].second - 1;

				if (isInside(img, ii, jj) && minimum < img(ii, jj)) minimum = img(ii, jj);
				if (isInside(img, ii, jj) && maximum > img(ii, jj)) maximum = img(ii, jj);
			}
			openImg(i, j) = minimum;
			closeImg(i, j) = maximum;
		}
	}
	m.erodeImg = erodeImg;
	m.openImg = openImg;
	m.closeImg = closeImg;
	m.dilateImg = dilateImg;

	return m;
}

Mat_<uchar> computeNMorphologies(Mat_<uchar>img, int n, int type) {
	Mat_<uchar> testImg = img.clone();
	Mat_<uchar> dilateImg;
	Mat_<uchar> erodeImg;
	Mat_<uchar> openImg;
	Mat_<uchar> closeImg;
	char* buffer = (char*)malloc(sizeof(char));
	for (int i = 0; i < n; i++) {
		if (type == 1) {
			erodeImg = computeMorphology(testImg, 0).erodeImg;
			testImg = erodeImg.clone();
		}
		else if (type == 0) {
			testImg = computeMorphology(testImg, 0).dilateImg;
		}
		else if (type == 2) {
			testImg = computeMorphology(testImg, 0).openImg;
		}
		else if (type == 3) {
			testImg = computeMorphology(testImg, 0).closeImg;
		}
	}
	free(buffer);
	if (type == 0) imshow("dilation ", testImg);
	else if (type == 1) imshow("erosion ", testImg);
	else if (type == 2) imshow("opening ", testImg);
	else imshow("closing", testImg);
	return testImg;
}

int* computeHistogram(Mat_<uchar> img) {

	int* h = (int*)calloc(256, sizeof(int));
	for (int i = 0; i < 256, h[i] = 0; i++);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			h[img(i, j)]++;
		}
	}
	return h;
}

double* computePDF(Mat_<uchar> img) {
	double* p = (double*)calloc(256, sizeof(double));

	int* h = computeHistogram(img);
	int M = img.rows * img.cols;

	for (int i = 0; i < 256; i++) {
		p[i] = (double)h[i] / (double)M;
	}
	return p;
}


Mat_<uchar> computeAdjacentMatrix(Mat_<uchar> img, pair<int, int> q) {
	
	int ii = q.first;
	int jj = q.second;
	Mat_<uchar> adjImg(3, 3,255);
	for (int i = 0; i < 3; i++) {
		for (int j = 0 ; j < 3; j++) {
			if(isInside(img,i + ii - 1,j + jj - 1))
				adjImg(i, j) = img(i + ii - 1,j + jj - 1);
			else {
				adjImg(i, j) = 0;
			}
		}
	}
	return adjImg;
}


double computeEntropy(Mat_<uchar> img) {
	

	double* p = computePDF(img);
	int M = img.rows * img.cols;
	double entropy = 0.0f;

	for (int i = 0; i < 256; i++) {
		if(p[i] > 1e-6)
			entropy = entropy + (p[i]) * log2(p[i]);
	}
	entropy = -entropy;
	return entropy;
}



Mat GrayToBinaryMatrix(unsigned char threshold,Mat_<uchar> img) {
	Mat grayImg = img;
	int height = grayImg.rows;
	int width = grayImg.cols;
	Mat binaryImg = Mat(height, width, CV_8UC1);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (grayImg.at<unsigned char>(i, j) < threshold)
				binaryImg.at<unsigned char>(i, j) = 0;
			else
				binaryImg.at<unsigned char>(i, j) = 255;

		}
	}

	imshow("binary image", binaryImg);
	return binaryImg;
}


Mat_<uchar> computeEntropyFilter(Mat_<uchar> img) {

	
	Mat_<double> entropyImg(img.rows, img.cols, CV_64F), normImg;
	Mat_<uchar> filter(img.rows, img.cols, 255);
	double maxEntropy = 0;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {

			entropyImg(i, j) = computeEntropy(computeAdjacentMatrix(img, { i,j }));
	
			if (entropyImg(i, j) > maxEntropy) maxEntropy = entropyImg(i, j);
		}
	}

	entropyImg.convertTo(normImg, CV_64F, 1.0 / maxEntropy, 0);
	normImg.convertTo(filter, CV_8UC1, 255, 0);
	imshow("entropy filter", filter);
	return filter;
}

Mat_<uchar> computeStandardDevFilter(Mat_<uchar> img) {

	Mat_<double> stddevImg(img.rows, img.cols, CV_64F);
	Mat meanTemp, stddevTemp,normImg;
	Mat_<uchar> filter(img.rows, img.cols, 255);

	double max = 0;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {


			meanStdDev(computeAdjacentMatrix(img, { i,j }), meanTemp, stddevTemp);
			double stddev = stddevTemp.at<double>(0, 0);
			stddevImg(i, j) = computeEntropy(computeAdjacentMatrix(img, { i,j }));

			if (stddevImg(i, j) > max) max = stddevImg(i, j);
		}
	}

	stddevImg.convertTo(normImg, CV_64F, 1.0 / max, 0);
	normImg.convertTo(filter, CV_8UC1, 255, 0);
	imshow("standard dev filter", filter);
	return filter;

}

int computeThreshold(Mat_<uchar> img, double C) {

	// oldT  - threshold between TR and BR at the beginning of while loop
	// newT - newly computed threshold at the end of while loop
	// miuTR - target region average grey level
	// miuBR - background image average grey level
	double oldT = 0.0, newT = 128.0;
	double miuTR = 0.0, miuBR = 255.0;

	//nBR,nTR - number of pixel in BR,TR regions
	int nBR, nTR;

	// TR - target region image
	// BR - background image 
	Mat_<uchar> TR = Mat(img.rows, img.cols, CV_8UC1);
	Mat_<uchar> BR = Mat(img.rows, img.cols, CV_8UC1);

	// repeat until different between old and new threshold <= C
	while (fabs(oldT - newT) > C) {

		//oldT - updates value with the one computed previous iteration
		oldT = newT;
		// reinitialize number of pixels in each region
		nBR = 0;
		nTR = 0;

		// TR,BR - images initialized as blank
		TR = Mat::zeros(TR.size(), TR.type());
		BR = Mat::zeros(BR.size(), BR.type());

		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {

				// targe region pixel found
				if (img(i, j) >= oldT) {
					miuTR += img(i, j);
					TR(i, j) = img(i, j);
					nTR++;
				}

				//background pixel found
				if (img(i, j) < oldT) {
					miuBR += img(i, j);
					BR(i, j) = img(i, j);
					nBR++;
				}
			}
		}
		// compute average grey level
		miuTR = miuTR / nTR;
		miuBR = miuBR / nBR;

		// newT - updates the newly computed threshold as a mean between background and target region
		newT = (miuTR + miuBR) / 2.0;
	}

	//imshow("Background", BR);
	//imshow("Target region", TR);
	//printf("newT\t\t:\t%lf\n", newT);

	return (int)newT;
}

Mat_<uchar> computeComplement(Mat_<uchar> img) {
	Mat_<uchar> compImg = img.clone();
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			compImg(i, j) = 255 - img(i, j);
		}
	}
	return compImg;
}

Mat_<uchar> computeIntersection(Mat_<uchar> dilateImg, Mat_<uchar> compImg) {
	Mat_<uchar> tempImg(compImg.rows, compImg.cols, 255);

	for (int i = 0; i < dilateImg.rows; i++) {
		for (int j = 0; j < dilateImg.cols; j++) {
			if (dilateImg(i, j) == compImg(i, j) && dilateImg(i, j) == 0)
				tempImg(i, j) = 0;
			else tempImg(i, j) = 255;
		}
	}
	return tempImg;
}

bool isEqual(Mat_<uchar> src, Mat_<uchar> dst) {

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src(i, j) != dst(i, j))
				return false;
		}
	}
	return true;
}

pair<int, int> computeFirstRegionPixel(Mat_<uchar> img,int label) {
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) == label) {
				if (isInside(img, i + 1, j) && img(i + 1, j) == 255)
					return { i + 1,j };
			}
		}
	}

	return { -1,-1 };
}

Mat_<uchar> computeDilationForRegion(Mat_<uchar> img) {
	Mat_<uchar> dilate(img.rows, img.cols, 255);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) == 0) {
				dilate(i, j) = 0;
				if (i + 1 < img.rows) dilate(i + 1, j) = 0;
				if (i - 1 > 0)        dilate(i - 1, j) = 0;
				if (j + 1 < img.cols)	 dilate(i, j + 1) = 0;
				if (j - 1 > 0)        dilate(i, j - 1) = 0;
			}
		}
	}
	return dilate;
}

Mat_<Vec3b> computeLabelToRGB(Mat_<uchar> img) {
	Mat_<Vec3b> labelImg = Mat::zeros(img.rows, img.cols, CV_8UC3);
	uchar color = 30;

	// initialize a color for each label possible
	int labels[256] = { 0 };
	for (int i = 0; i < 256; i++) {
		labels[i] = rand() % 256;
	}

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			color = labels[img(i, j)];
			if (img(i, j) != 0) {
				labelImg(i, j)[0] = color;
				labelImg(i, j)[1] = color + 30;
				labelImg(i, j)[2] = color + 50;
			}
			else {
				labelImg(i, j)[0] = 255;
				labelImg(i, j)[1] = 255;
				labelImg(i, j)[2] = 255;
			}
		}
	}

	imshow("coloured label", labelImg);

	return labelImg;
}
pair<int, int>* compute8ConnectivityNeighbours(Mat_<uchar> img, pair<int, int> q) {

	int di[8] = { 0,-1,-1,-1, 0, 1, 1, 1 };
	int dj[8] = { 1, 1, 0,-1,-1,-1, 0, 1 };
	int i = q.first;
	int j = q.second;
	pair<int, int> neighbours[8];
	for (int k = 0; k < 8; k++)
		neighbours[k] = { i + di[k], j + dj[k] };
	return neighbours;
}

map<pair<int, int>, uchar> computeNeighbours(Mat_<uchar> img, pair<int, int> q) {

	int di[4] = { -1,0,1,0 };
	int dj[4] = { 0,-1,0,1 };
	int i = q.first;
	int j = q.second;
	map<pair<int, int>, uchar> neighbours;
	for (int k = 0; k < 4; k++)
		neighbours.insert({ {i + di[k], j + dj[k]}, (uchar)k });
	return neighbours;
}

Mat_<uchar> computeBFSlabeling(Mat_<uchar> origin) {

	Mat_<uchar> img = GrayToBinaryMatrix(160, origin);
	imshow("original", img);
	waitKey();
	Mat_<uchar> labelImg = Mat::zeros(img.rows, img.cols, CV_8UC1);

	int label = 1;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {

			// we start the labeling 
			if (labelImg(i, j) == 0 && img(i, j) == 0) {
				label++;
				queue<pair<int, int>> Q;
				labelImg(i, j) = label;
				Q.push({ i, j });

				while (!Q.empty()) {

					map<pair<int, int>, uchar> neighbours = computeNeighbours(img, Q.front());
					Q.pop();

					for (auto item = neighbours.begin(); item != neighbours.end(); ++item) {
						int ii = item->first.first;
						int jj = item->first.second;

						if (isInside(img,ii,jj) && img(ii, jj) == 0 && labelImg(ii, jj) == 0) {

							labelImg(ii, jj) = label;
							Q.push({ ii,jj });
						}
					}
				}
			}
		}
	}
	return labelImg;
}

int finishBorder(int counter, pair<int, int>* border) {

	if (counter > 1) {
		if (border[1] == border[counter] && border[counter - 1] == border[0])
			return 1;
		else
			return -1;
	}
	else return 0;
}

pair<int,int>* computeBorder(Mat_<uchar> border, Mat_<uchar> img, pair<int, int> pixel, int label) {

	pair<int, int> Border[10000];

	short int dir = 7;
	int counter = 0;
	Border[counter] = pixel;
	bool okey = true;
//	printf("pixel(%d,%d)\n", pixel.first, pixel.second);

	while (okey && finishBorder(counter, Border) != 1) {

		okey = false;
		pair<int, int>* adj = compute8ConnectivityNeighbours(img, pixel);
		dir = (dir % 2 == 0) ? (dir + 7) % 8 : (dir + 6) % 8;

		for (int p = dir; p < dir + 8; p++) {

			int ii = adj[p % 8].first;
			int jj = adj[p % 8].second;
			if (isInside(img, ii, jj) && img(ii, jj) == label) {

				Border[++counter] = { ii,jj };
				dir = (p % 8);

				pixel.first = ii;
				pixel.second = jj;
			//	printf("pixel(%d,%d)\n", ii, jj);
				okey = true;
				break;
			}

		}

	}

	for (int i = 0; i <= counter; i++) {
		border(Border[i].first, Border[i].second) = 255;
	}

	Border[counter+1] = {-1,-1};
	return Border;
}

Mat_<uchar> computeBorderTracing(Mat_<uchar> img) {

	Mat_<uchar> border = Mat::zeros(img.rows, img.cols, CV_8UC1);
	// label - the current label we are at
	int label = 1;

	// traverse image in search of a region-pixel : pixel different from white colour
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {

			// check if current pixel is part of a region( != 255) and also a not explored region (!= label)
			if (img(i, j) == label) {

				//apply the contour algorithm
				pair<int,int>* borderArray = computeBorder(border, img, { i,j }, label);
				label++;
			}
		}
	}

	imshow("border ", border);
	waitKey();
	return border;
}

Mat_<uchar> computeRegionFilling(Mat_<uchar> img) {

	// constant images
	Mat_<uchar> compImg = computeComplement(img);

	Mat_<uchar> tempImg(img.rows, img.cols, 255);
	Mat_<uchar> fillImg(img.rows, img.cols, 255);

	// X_0 
	for (int label = 1; label < 100; label++) {
		pair<int, int> p = computeFirstRegionPixel(img, label);
		if (p.first == -1) continue;
		fillImg(p.first, p.second) = 0;
		printf("first region(%d,%d)\n", p.first, p.second);
		while (p.first != -1 && !isEqual(fillImg, tempImg)) {

			tempImg = fillImg;
			// X_1 
			fillImg = computeDilationForRegion(fillImg);
			fillImg = computeIntersection(fillImg, compImg);
		}

	}
	//imshow("region filling", fillImg);
	//imshow("original ", img);
	//imshow("inverse", compImg);
//	waitKey();
	return fillImg;
}

Mat_<Vec3b> GrayToRGBBorder(Mat_<Vec3b> img, pair<int, int>* boundary) {

	int counter = -1;
	while (boundary[++counter].first != -1) {
		img(boundary[counter].first, boundary[counter].second)[0] = 0;
		img(boundary[counter].first, boundary[counter].second)[1] = 255;
		img(boundary[counter].first, boundary[counter].second)[2] = 0;
	}

	imshow("boundary", img);
	waitKey();
	return img;
}

int getMaxAreaContourId(vector <vector<cv::Point>> contours) {
	double maxArea = 0;
	int maxAreaContourId = -1;
	for (int j = 0; j < contours.size(); j++) {
		double newArea = cv::contourArea(contours.at(j));
		if (newArea > maxArea) {
			maxArea = newArea;
			maxAreaContourId = j;
		} 
	}
	return maxAreaContourId;
} 

Mat_<uchar> computeTextureSegmentation(Mat_<uchar> img) {

	imshow("original", img);
	Mat_<uchar> entropyImg = computeEntropyFilter(img);
	//computeStandardDevFilter(img);
	

	printf("thersholding...\n");
	uchar threshold1 = computeThreshold(img, 0.3);
	printf("binarization...\n");
	Mat_<uchar> binaryImg2 = GrayToBinaryMatrix(threshold1, img);
	printf("area-opening...\n");
	
	Mat kernel(5, 5, 1);
	InputArray newKernel = InputArray(kernel);
	Mat_<uchar> openingImg;
	morphologyEx(binaryImg2, openingImg, MORPH_OPEN, newKernel);
	imshow("opening ", openingImg);
	Mat_<uchar> closingImg;
	printf("area-closing...\n");
	morphologyEx(openingImg, closingImg, MORPH_CLOSE, newKernel);
	imshow("closing ", closingImg);
	
	
	vector<std::vector<cv::Point> > contours;
	Mat contourImg = closingImg.clone();
	printf("finding largest contour...\n");
	findContours(contourImg, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	int index = getMaxAreaContourId(contours);

	Mat RGBcontourImg;
	vector<Vec4i> hierarchy;
	cvtColor(img, RGBcontourImg, COLOR_GRAY2RGB, 0);
	printf("drawing contour...\n");
	drawContours(RGBcontourImg, contours, index, Scalar(0,255,0), FILLED, 8, hierarchy);

	cv::imshow("Contours", RGBcontourImg);
	waitKey();
	return cv::Mat_<uchar>(0, 0);
}


int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 4 - compute texture segmentation\n");
		printf(" 3 - compute entropy filter\n");
		printf(" 2 - compute entropy \n");
		printf(" 1 - compute adjacent matrix \n");
		
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		Mat_<uchar> img = imread("Images/cat.bmp", 0);
		Mat_<uchar> img2 = imread("Images/cameraman.bmp", 0);
		Mat_<uchar> img3 = imread("Images/cell.bmp", 0);
		Mat_<uchar> img4 = imread("Images/rice.bmp", 0);
		int down_width = img.cols * 0.3;
		int down_height = img.rows * 0.3;
		resize(img, img, Size(down_width, down_height), INTER_LINEAR);
		switch (op)
		{

		case 4:
			computeTextureSegmentation(img);
			break;
		case 3:
			computeEntropyFilter(img);
			break;
		case 2:
			computeEntropy(img);
			break;
		case 1:
			computeAdjacentMatrix(img, { 10,10 });
			break;
	
		case 0:
			break;
		}
	} while (op != 0);
	return 0;
}