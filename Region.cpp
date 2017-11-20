/*
* This code is provided as part of "A Practical Introduction to Computer Vision with OpenCV"
* by Kenneth Dawson-Howe © Wiley & Sons Inc. 2014.  All rights reserved.
*/
#include "Utilities.h"

Mat kmeans_clustering(Mat& image, int k, int iterations)
{
	CV_Assert(image.type() == CV_8UC3);
	// Populate an n*3 array of float for each of the n pixels in the image
	Mat samples(image.rows*image.cols, image.channels(), CV_32F);
	float* sample = samples.ptr<float>(0);
	for (int row = 0; row<image.rows; row++)
		for (int col = 0; col<image.cols; col++)
			for (int channel = 0; channel < image.channels(); channel++)
samples.at<float>(row*image.cols + col, channel) =
(uchar)image.at<Vec3b>(row, col)[channel];
// Apply k-means clustering to cluster all the samples so that each sample
// is given a label and each label corresponds to a cluster with a particular
// centre.
Mat labels;
Mat centres;
kmeans(samples, k, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 1, 0.0001),
	iterations, KMEANS_PP_CENTERS, centres);
// Put the relevant cluster centre values into a result image
Mat& result_image = Mat(image.size(), image.type());
for (int row = 0; row < image.rows; row++)
	for (int col = 0; col < image.cols; col++)
		for (int channel = 0; channel < image.channels(); channel++)
			result_image.at<Vec3b>(row, col)[channel] = (uchar)centres.at<float>(*(labels.ptr<int>(row*image.cols + col)), channel);
return result_image;
}


vector<Rect> NewMethod(Mat& image) {
	//Meanshift and convert to greyscale
	Mat greyscaleImage, meanshiftImage;
	pyrMeanShiftFiltering(image, meanshiftImage, 45, 20, 2);
	//GaussianBlur(meanshiftImage, meanshiftImage, Size(5,5), 0, 0);
	cvtColor(meanshiftImage, greyscaleImage, CV_BGR2GRAY);
	Mat greyscaleCopy = greyscaleImage.clone();
	//Edge detection
	Mat edgesResult;
	vector<vector<Point>> contoursFound;
	vector<Vec4i> hierarchy;
	Canny(greyscaleImage, edgesResult, 80, 180, 3);
	edgesResult.convertTo(edgesResult, CV_8U);
	Mat edgesResultCopy = edgesResult.clone();
	
	//Closing (dilate and erode)
	Mat closeRes = edgesResult.clone();
	morphologyEx(closeRes, closeRes, MORPH_CLOSE, Mat(), Point(-1, -1),2);

	//Dilate edges
	//dilate(edgesResult, edgesResult, Mat(), Point(-1, -1), 1, 1, 1);
	//Mat dilateRes = edgesResult.clone();

	 

	//Connected components on the ED
	findContours(closeRes, contoursFound, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

	//Classify edges to get sign
	vector<vector<Point>> test;
	vector<Point> signContour;
	signContour = contoursFound[0];
	int maxContours = 0;
	int curMaxArea = 0;
	for (int contourNum = 0; (contourNum < (int)contoursFound.size()); contourNum++)
	{
		vector<Point> curContour = contoursFound[contourNum];
		//Check is suitable size
		if (contourArea(contoursFound[contourNum]) > 250) {
			//Check if is parent
			int child = hierarchy[contourNum].val[2];
			if (child != -1) {
				int count = 1;
				while (hierarchy[child].val[0] != -1) {
					//Check if valid contour
					int a = contourArea(contoursFound[child]);
					if (a > 50 && a< (250*250)){
						int child2 = hierarchy[child].val[2];
						if (child2 != -1) {
							count++;
							while (hierarchy[child2].val[0] != -1) {
								count++;
								child2 = hierarchy[child2].val[0];
							}
						}
						count++;
					}
					//Go to next contour
					child = hierarchy[child].val[0];
				}
				maxContours = max(maxContours, count);
				if (count >= 20) {
					test.push_back(contoursFound[contourNum]);
					if (contourArea(contoursFound[contourNum]) > curMaxArea) {
						signContour = contoursFound[contourNum];
						curMaxArea = contourArea(contoursFound[contourNum]);
					}
				}
			}
		}
	}
	Mat contours_image2 = Mat::zeros(image.size(), CV_8UC3);
	for (int contour_number = 0; (contour_number<(int)test.size()); contour_number++)
	{
		Scalar colour(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
		drawContours(contours_image2, test, contour_number, colour, 2, 8);
	}


	Rect signRect = boundingRect(signContour);
	Mat cropped = image(signRect);
	Mat clustered_image;
	clustered_image = kmeans_clustering(cropped, 2, 5);
	cvtColor(clustered_image, clustered_image, CV_BGR2GRAY);
	threshold(clustered_image, clustered_image, 150, 255, THRESH_BINARY_INV);
	Mat clusteredCopy = clustered_image.clone();

	vector<vector<Point>> contoursFound2;
	//Connected components on the ED
	findContours(clusteredCopy, contoursFound2, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
	vector<Rect> rectangles;
	Mat croppedContours = Mat::zeros(image.size(), CV_8UC3);
	for (int contour_number = 0; (contour_number<(int)contoursFound2.size()); contour_number++)
	{
		Scalar colour(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
		Rect boundRect = boundingRect(Mat(contoursFound2[contour_number]));
		if (boundRect.width <150 && boundRect.height < 150 && boundRect.height> 5 && boundRect.width>5) {
			rectangles.push_back(boundRect);
		}
		//drawContours(croppedContours, contoursFound2, contour_number, colour, 2, 8);
	}
	///applyBoundingRect(cropped, rectangles, (0, 0, 0));
	bool changes = true;
	vector<Rect> newRects;
	vector<Rect> t = rectangles;
	while (changes) {
		newRects.clear();
		changes = updateRects(t, newRects);
		t = newRects;
	}

	applyBoundingRect(cropped, newRects, (0, 0, 0xFF));

	//Draw biggest contour
	Mat contoursImageED = Mat::zeros(edgesResultCopy.size(), CV_8UC3);
	polylines(contoursImageED, signContour, true, Scalar(255, 255, 255));

	

	//Outputs
	Mat output1, output2, output3;
	output1 = JoinImagesHorizontally(contours_image2, "Original Image", closeRes, "CCA on ED", 6);
	output2 = JoinImagesVertically(edgesResultCopy, "EdgeDetection", contoursImageED, "meanshift greyscale", 6);
	//imshow("cropped", cropped);
	imshow("image", image);
//	imshow("CroppedContours", croppedContours);
	//imshow("ED", edgesResultCopy);
	//imshow("clustered", clustered_image);
	//imshow("Closing Res", closeRes);
	//imshow("Contors- Sign", contoursImageED);
	//simshow("All Contors", contours_image2);
	//imshow("", closeRes);
//	char choice = cvWaitKey();
	//cvDestroyAllWindows();
	//imshow("Grey Scale Morphology & Connected Components", output1);
	return newRects;
}

bool updateRects(vector<Rect> rectangles, vector<Rect> &newRects) {
	//vector <Rect> newRects;
	vector <bool> used (rectangles.size());
	bool changes = false;
	std::fill(used.begin(), used.end(), false);		
		for (int rectNo = 0; rectNo < (int)rectangles.size(); rectNo++) {
			Rect curRect = rectangles[rectNo];
			if (!used.at(rectNo)) {
				for (int rectNo2 = 0; rectNo2 < (int)rectangles.size(); rectNo2++) {
					if (rectNo != rectNo2) {
						Rect rect2 = rectangles[rectNo2];
						if (overlappingRects(curRect, rect2)) {
							//Update to larger rect
							curRect = getNewRect(curRect, rect2);
							used[rectNo2] = true;
							changes = true;
						}
					}
				}
				rectangles[rectNo] = curRect;
				if (curRect.width > 10 && curRect.height > 10) {
					newRects.push_back(curRect);
				}
			}
		}
	return changes;
}

void applyBoundingRect(Mat& image, vector<Rect> rectangles, Scalar colour) {
	//Iterate through all rectangles applying to image
	for (int rectNo = 0; rectNo < (int)rectangles.size(); rectNo++) {
		Rect boundRect = rectangles[rectNo];
		rectangle(image, boundRect.tl(), boundRect.br(), colour, 2, 8, 0);
	}
}
bool areIdenticalRects(Rect r1, Rect r2) {
	return (r1.x == r2.x && r1.y == r2.y && r1.width == r2.width);
}
bool overlappingRects(Rect boundRect, Rect boundRect2) {
	Rect r1 = boundRect;
	r1.width += 20;
	r1.height += 8;

	return ((r1 & boundRect2).area() > 0);
}

Rect getNewRect(Rect boundRect, Rect boundRect2) {
	int newX = min(boundRect.x, boundRect2.x);
	int newY = min(boundRect.y, boundRect2.y);
	int r1x2 = boundRect.x + boundRect.width;
	int r1y2 = boundRect.y + boundRect.height;
	int r2x2 = boundRect2.x + boundRect2.width;
	int r2y2 = boundRect2.y + boundRect2.height;
	int newWidth = max(r1x2, r2x2) - newX;
	int newHeight = max(r1y2, r2y2) - newY;
	return Rect(newX, newY, newWidth, newHeight);
}


void ConnectedComponents(Mat& image)
{
	//Greyscale -> slight dilate -> edge detection -> Classify Sign (Count number of valid inner contours) -> Crop to sign(edRes,edRes) -> Dilate more(edRes) -> components (edRes, edComp) -> BoundingRects(rects, edComp) -> Classify&Merge(rects,rects) -> applyBoundingRects
	//connected components
	Mat greyscaleImage, binaryImage,meanshiftImage;
	pyrMeanShiftFiltering(image, meanshiftImage, 30, 20, 2);
	cvtColor(meanshiftImage, greyscaleImage, CV_BGR2GRAY);
	Mat greyscaleDilate = greyscaleImage.clone();
	dilate(greyscaleDilate, greyscaleDilate, Mat(), Point(-1, -1), 1, 1, 1);
	threshold(greyscaleImage, binaryImage, 150, 255, THRESH_BINARY_INV);
	dilate(binaryImage, binaryImage, Mat(), Point(-1, -1), 2, 1, 1);
	Mat binaryImageCopy = binaryImage.clone();
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(binaryImageCopy, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
	Mat contours_image1 = Mat::zeros(binaryImage.size(), CV_8UC3);
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> rectangles;
	
	for (int contour_number = 0; (contour_number<(int)contours.size()); contour_number++)
	{
		approxPolyDP(Mat(contours[contour_number]), contours_poly[contour_number], 3, true);
		Rect boundRect = boundingRect(Mat(contours_poly[contour_number]));
		if (boundRect.width < 15 || boundRect.height < 10) {
			
			continue;
		}
		//else if (boundRect.height > 180) {
			//continue;
		//}
		else {
			rectangles.push_back(boundRect);
		}
	}
	applyBoundingRect(image, rectangles, (0xFF, 0, 0));
	vector<Rect> newRects;
	for (int rectNo = 0; rectNo < (int)rectangles.size(); rectNo++) {
		Rect boundRect = rectangles[rectNo];
		for (int rectNo2 = rectNo; rectNo2 < (int)rectangles.size(); rectNo2++) {
			if (rectNo != rectNo2) {
				Rect boundRect2 = rectangles[rectNo2];
				if (overlappingRects(boundRect, boundRect2)) {
					//Update to larger rect
					Rect newRect = getNewRect(boundRect, boundRect2);
					boundRect = newRect;
					
					//Remove second rec
					//rectangles.erase(rectangles.begin() + rectNo2);
					//rectNo2=0;
				}
				newRects.push_back(boundRect);
			}
		}
	}
	Mat gray, edgesFound, edgesResult;
	vector<vector<Point>> contoursFound;
	Canny(greyscaleImage, edgesFound, 50, 150, 3);
	edgesFound.convertTo(edgesResult, CV_8U);
	Mat edgesResultCopy = edgesResult.clone();
	findContours(edgesResult, contoursFound, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
	Mat contoursImageED = Mat::zeros(edgesResult.size(), CV_8UC3);
	vector<Point> signContour;
	signContour =contoursFound[0];
	int curMaxArea = contourArea(signContour);
	for (int contourNum = 0; (contourNum<(int)contoursFound.size()); contourNum++)
	{
		int a = contourArea(contoursFound[contourNum]);
		int a2 = (int)contoursFound[contourNum].size();
		if (a > curMaxArea ) {
			curMaxArea = a;
			signContour = contoursFound[contourNum];
		}
	}
	Scalar colour = (0xFF, 0, 0xFF);

	//drawContours(contoursImageED, signContour, -1, colour, CV_FILLED, 8, hierarchy);
	polylines(contoursImageED, signContour, true, Scalar(255, 255, 255));
	Mat binary_display, output1, output2, output3;
	cvtColor(binaryImage, binary_display, CV_GRAY2BGR);
	
	namedWindow("ED", CV_WINDOW_AUTOSIZE);
	imshow("ED", edgesResultCopy);
	output1 = image;
	output1 = JoinImagesHorizontally(image, "Original Image", greyscaleImage, "meanshift greyscale", 6);
	output2 = JoinImagesHorizontally(binary_display, "Thresholded Image", contoursImageED, "CCA on ED", 6);
	output2 = JoinImagesVertically(output1, "", output2, "");
	imshow("Grey Scale Morphology & Connected Components", output2);
}

void edges(Mat& image, string s) {
	Mat gray, edge, draw;
	cvtColor(image, gray, CV_BGR2GRAY);
	Canny(gray, edge, 50, 150, 3);
	edge.convertTo(draw, CV_8U);
	namedWindow(s, CV_WINDOW_AUTOSIZE);
	imshow(s, draw);
}
