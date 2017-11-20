/*
* This code is provided as part of "A Practical Introduction to Computer Vision with OpenCV"
* by Kenneth Dawson-Howe © Wiley & Sons Inc. 2014.  All rights reserved.
*/
#include "Utilities.h"
#include <iostream>
#include <fstream>
using namespace std;
vector<vector<vector<Point>>> groundTruths;
vector<double> dice;
Mat gtImage;
/*=
{
	{ { Point(34, 17) , Point(286, 107) },{ Point(32, 117) ,Point(297, 223) },{ Point(76, 234) , Point(105, 252) } },
	{ { Point(47, 191) ,Point(224, 253) },
	{ { Point(142, 121) , Point(566, 392) } },
	{ { Point(157,72) , Point(378, 134) },{ Point(392, 89) , Point(448, 132) },{ Point(405, 138) , Point(442, 152) },{ Point(80, 157) ,Point(410, 245) },{ Point(82, 258) , Point(372, 322) } },
	{ { Point(112, 73) , Point(598, 170) },{ Point(108, 178) ,Point(549, 256) },{ Point(107, 264) ,Point(522, 352) } },
	{ { Point(91, 54) , Point(446, 227) } },
	{ { Point(64, 64) , Point(476, 268) },{ Point(529, 126) , Point(611, 188) },{ Point(545, 192) , Point(603, 211) },{ Point(210, 305) , Point(595, 384) } },
	{ { Point(158, 90) , Point(768, 161) },{ Point(114, 174) , Point(800, 279) } }
};
*/
int main(int argc, const char** argv)
{
	char* file_location = "Notices/";
	char* image_files[] = {

		"Notice1.jpg", //0
		"Notice2.jpg",//3
		"Notice3.jpg", //1
		"Notice4.jpg",	//4	
		"Notice5.jpg", //2
		"Notice6.jpg",//5
		"Notice7.jpg",//6
		"Notice8.jpg"//7
	};

	// Load images
	int number_of_images = sizeof(image_files) / sizeof(image_files[0]);
	Mat* image = new Mat[number_of_images];
	for (int file_no = 0; (file_no < number_of_images); file_no++)
	{
		string filename(file_location);
		filename.append(image_files[file_no]);
		image[file_no] = imread(filename, -1);
		if (image[file_no].empty())
		{
			cout << "Could not open " << image[file_no] << endl;
			return -1;
		}
	}

	groundTruths.push_back({ { Point(34, 17) , Point(286, 107) },{ Point(32, 117) ,Point(297, 223) },{ Point(76, 234) , Point(105, 252) } });
	groundTruths.push_back({ { Point(47, 191) ,Point(224, 253) } });
	groundTruths.push_back({ { Point(142, 121) , Point(566, 392) } });
	groundTruths.push_back({ { Point(157,72) , Point(378, 134) },{ Point(392, 89) , Point(448, 132) },{ Point(405, 138) , Point(442, 152) },{ Point(80, 157) ,Point(410, 245) },{ Point(82, 258) , Point(372, 322) } });
	groundTruths.push_back({ { Point(112, 73) , Point(598, 170) },{ Point(108, 178) ,Point(549, 256) },{ Point(107, 264) ,Point(522, 352) } });
	groundTruths.push_back({ { Point(91, 54) , Point(446, 227) } });
	groundTruths.push_back({ { Point(64, 64) , Point(476, 268) },{ Point(529, 126) , Point(611, 188) },{ Point(545, 192) , Point(603, 211) },{ Point(210, 305) , Point(595, 384) } });
	groundTruths.push_back({ { Point(158, 90) , Point(768, 161) },{ Point(114, 174) , Point(800, 279) } });
	
	int choice;
	int i = 0;
	do
	{
		choice = cvWaitKey();
		cvDestroyAllWindows();
		//ConnectedComponents(image[6]);
		gtImage = image[i].clone();
		
		vector<Rect> res = NewMethod(image[i]);
		
		calcDICE(groundTruths.at(i), res);
		cout << "Dice for " << i+1 << " : " << dice.at(i) << "\n";
		i++;

	} while (i < number_of_images);
	while (true) {}
}

void calcDICE(vector<vector<Point>> gt, vector<Rect> results) {
	vector<Rect> gtRects;
	double areaGT = 0.0;
	for (int i = 0; i < gt.size(); i++) {
		vector<Point> pts = gt.at(i);
		int x = pts.at(0).x;
		int y = pts.at(0).y;
		int w = pts.at(1).x - x;
		int h = pts.at(1).y - y;
		Rect r = Rect(x, y, w, h);
		areaGT = areaGT + r.area();
		gtRects.push_back(r);
	}
	applyBoundingRect(gtImage, gtRects, (0, 255, 0));
	imshow("GT",gtImage);
	double areaOverlap = 0.0;

	double areaRes = 0.0;
	for (int i = 0; i < results.size(); i++) {
		double maxOverlap = 0.0;
		Rect r1 = results.at(i);
		areaRes = areaRes + r1.area();
		for (int j = 0; j < gtRects.size(); j++) {
			Rect r2 = gtRects.at(j);
			double overlap = (r1 & r2).area();
			maxOverlap = max(maxOverlap, overlap);
		}
		areaOverlap = areaOverlap + maxOverlap;
	}
	double diceRes = (2 * areaOverlap) / (areaGT + areaRes);
	dice.push_back(diceRes);
	//cout << diceRes << "\n";

}

