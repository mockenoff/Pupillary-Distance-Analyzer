#include "cv.h"
#include "cvaux.h"
#include "cxcore.h"
#include "ml.h"
#include "cxmisc.h"
#include "highgui.h"
#include "cxtypes.h"
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;


// Haar Cascade files
const char* cascade_face = "/usr/local/share/opencv/haarcascades/haarcascade_frontalface_alt.xml";
const char* cascade_eyes = "/usr/local/share/opencv/haarcascades/haarcascade_eye.xml";

// The original image
IplImage *img;
// Iris (relative) min x, max x, and width
int ilmax = 0; int ilmin = 0; int ilwidth = 0;
int irmax = 0; int irmin = 0; int irwidth = 0;
// Face rect
int fx = 0; int fy = 0; int fw = 0; int fh = 0;
// Eye rects
int elx = 0; int ely = 0; int elw = 0; int elh = 0; int eli;
int erx = 0; int ery = 0; int erw = 0; int erh = 0; int eri;
// Contour variables for iris detection
CvSeq* con1; CvMemStorage* sto1 = cvCreateMemStorage(0);
CvSeq* con2; CvMemStorage* sto2 = cvCreateMemStorage(0);

// findIris variables
CvPoint* point;
int idx = 0;
int amin = 0; int amax = 0; int absw = 0; int maxp = 0;
int xdir = -2; int ydir = -1; int minx = 0; int maxx = 0; int miny = 0; int maxy = 0; int lastx = 0; int lasty = 0; int ychanges = 0; int pts = 0;

// estimatePD variables
float pd = 0.0;

// Reset findIris in break cases
void resetIris() {
	idx--;
	ychanges = 0;
	if(lastx < point->x) xdir = 1; else if(lastx > point->x) xdir = 0; else xdir = -1;
	if(lasty < point->y) ydir = 1; else if(lasty > point->y) ydir = 0; else ydir = -1;
	if(pts > maxp || (maxx - minx > absw && pts == maxp)) {
		maxp = pts; amin = minx; amax = maxx; absw = maxx - minx;
		cout << " (" << amin << " " << amax << " " << absw << ")";
	}
	pts = 1;
	minx = maxx = lastx;
	miny = maxy = lasty;
}

// Get the iris dimensions
void findIris(int which = 1) {
	// Create and reset variables
	CvSeq* result;
	CvSeq* con = (which == 1) ? con1 : con2;
	CvMemStorage* sto = (which == 1) ? sto1 : sto2;
	amin = 0; amax = 0; absw = 0; maxp = 0;
	while(con) {
		result = cvApproxPoly(con, sizeof(CvContour), sto, CV_POLY_APPROX_DP, cvContourPerimeter(con)*0.01, 0);
		cout << result->total;
		con = con->h_next;
		if(result->total < 3) continue;
		xdir = -2; ydir = -1; minx = 0; maxx = 0; miny = 0; maxy = 0; lastx = 0; lasty = 0; ychanges = 0; pts = 0;
		for(idx = 0; idx < result->total; idx++) {
			pts++;
			point = (CvPoint*)cvGetSeqElem(result, idx);
cout << "\n\t(" << point->x << "," << point->y << "), (" << lastx << "," << lasty << ")";
			// Initial loop
			if(xdir == -2) {
cout << "BREAK1";
				xdir = -1;
				minx = maxx = lastx = point->x;
				miny = maxy = lasty = point->y;
				continue;
			}
			// Break for no vertical or no horizontal movement
			if(lastx == point->x || lasty == point->y) {
				pts = 1;
				ychanges = 0;
				xdir = -1; ydir = -1;
				minx = maxx = lastx = point->x;
				miny = maxy = lasty = point->y;
				continue;
			}
			// Break for backtracking on x direction
			if((xdir == 0 && lastx < point->x) || (xdir == 1 && lastx > point->x)) {
cout << "BREAK2[" << pts << "," << maxp << "," << minx << "," << amin << "," << maxx << "," << amax << "]";
				resetIris();
				continue;
			}
			// Moving right (xdir == 1)
			if(lastx < point->x) {
				if(xdir == 0) {
cout << "BREAK3[" << pts << "," << maxp << "," << minx << "," << amin << "," << maxx << "," << amax << "]";
					resetIris();
					continue;
				}
				else xdir = 1;
			}
			// Moving left (xdir == 0)
			else if(lastx > point->x) {
				if(xdir == 1) {
cout << "BREAK4[" << pts << "," << maxp << "," << minx << "," << amin << "," << maxx << "," << amax << "]";
					resetIris();
					continue;
				}
				else xdir = 0;
			}
			// Moving down (ydir == 1), which it shouldn't since we're only aiming to take the bottom half of the iris for simplicity
			if(lasty < point->y) {
				if(ydir == 0) {
					/*ychanges++;
					if(ychanges > 1) {
cout << "BREAK5[" << pts << "," << maxp << "," << minx << "," << amin << "," << maxx << "," << amax << "]";
						resetIris();
						continue;
					}*/
cout << "BREAK5[" << pts << "," << maxp << "," << minx << "," << amin << "," << maxx << "," << amax << "]";
					resetIris();
					continue;
				}
				ydir = 1;
			}
			// Moving up (ydir == 0)
			else if(lasty > point->y) {
				if(ydir == 1) {
					ychanges++;
					if(ychanges > 1) {
cout << "BREAK6[" << pts << "," << maxp << "," << minx << "," << amin << "," << maxx << "," << amax << "]";
						resetIris();
						continue;
					}
				}
				ydir = 0;
			}
			// Compare the max and min values
			if(point->x < minx) minx = point->x;
			else if(point->x > maxx) maxx = point->x;
			if(point->y < miny) miny = point->y;
			else if(point->y > maxy) maxy = point->y;
			// Keep this point for the next loop
			lastx = point->x;
			lasty = point->y;
cout << " " << ychanges << " " << xdir << " " << ydir << " " << minx << " " << maxx;
		}
		cout << endl;
	}
	// Set appropriate variables after finding iris
	if(which == eli) {
		ilmin = amin; ilmax = amax; ilwidth = absw;
	}
	else {
		irmin = amin; irmax = amax; irwidth = absw;
	}
}

// Function to estimate the PD
void estimatePD() {
	float lpx = (float)fx + (float)elx + (float)ilmin + (((float)ilmax - (float)ilmin)/2);
	float rpx = (float)fx + (float)erx + (float)irmin + (((float)irmax - (float)irmin)/2);
	float lpy = (float)fy + (float)ely + ((float)elh/2);
	float rpy = (float)fy + (float)ery + ((float)erh/2);
	float cside = sqrt(pow(abs(lpx - rpx), 2) + pow(abs(lpy - rpy), 2));
//	pd = cside * (11.8 / ((((float)ilmax - (float)ilmin) + ((float)irmax - (float)irmin)) / 2));
	pd = ((ilmax - ilmin) > (irmax - irmin)) ? cside * (11.8 / ((float)ilmax - (float)ilmin)) : cside * (11.8 / ((float)irmax - (float)irmin));
	cout << "in pixels: " << cside << endl << "in mm: " << pd << endl;
	cvLine(img, cvPoint(lpx, lpy), cvPoint(rpx, rpy), cvScalarAll(255), 1, 8, 0);
}

// Function to detect and isolate the eyes
bool detectEyes(){
	// Create storage
	CvMemStorage* storage = cvCreateMemStorage(0);
	// Load Haar Cascade files
	CvHaarClassifierCascade* cascade_f = (CvHaarClassifierCascade*)cvLoad(cascade_face, 0, 0, 0);
	CvHaarClassifierCascade* cascade_e = (CvHaarClassifierCascade*)cvLoad(cascade_eyes, 0, 0, 0);

	// Create left and right gray scale images
	IplImage* left = cvCreateImage(cvGetSize(img), 8, 1);
	IplImage* right = cvCreateImage(cvGetSize(img), 8, 1);
	cvCvtColor(img, left, CV_BGR2GRAY);
	cvCvtColor(img, right, CV_BGR2GRAY);
	cvThreshold(left, left, 100, 255, CV_THRESH_BINARY);
	cvThreshold(right, right, 100, 255, CV_THRESH_BINARY);

	// Detect the face, return if not found
	CvSeq *faces = cvHaarDetectObjects(img, cascade_f, storage, 1.1, 3, 0, cvSize(40, 40));
	if(faces->total == 0) return false;

	// Draw a red rectangle around the face on the original image
	CvRect *face = (CvRect*)cvGetSeqElem(faces, 0);
	cvRectangle(img, cvPoint(face->x, face->y), cvPoint(face->x + face->width, face->y + face->height), CV_RGB(255, 0, 0), 1, 8, 0);
	cvClearMemStorage(storage);

	// Set face variables and create the ROI for eyes
	fx = face->x; fy = face->y; fw = face->width; fh = face->height;
	cvSetImageROI(img, cvRect(fx, fy + (fh/5.5), fw, fh/3.0));
	// Detect the eyes, return if there are not two of them
	CvSeq *eyes = cvHaarDetectObjects(img, cascade_e, storage, 1.15, 3, 0, cvSize(25, 15));
	if(eyes->total != 2) return false;

	// Determine left from right and store in the eye variables
	CvRect *eye1 = (CvRect*)cvGetSeqElem(eyes, 0);
	CvRect *eye2 = (CvRect*)cvGetSeqElem(eyes, 1);
	if(eye1->x < eye2->x) {
		eli = 1; elx = eye1->x; ely = eye1->y; elw = eye1->width; elh = eye1->height;
		eri = 2; erx = eye2->x; ery = eye2->y; erw = eye2->width; erh = eye2->height;
	}
	else {
		eli = 2; elx = eye2->x; ely = eye2->y; elw = eye2->width; elh = eye2->height;
		eri = 1; erx = eye1->x; ery = eye1->y; erw = eye1->width; erh = eye1->height;
	}

	// Create the eye ROIs on the left and right grayscale images
	cvSetImageROI(left, cvRect(elx+fx, ely+fy+(fh/5.5), elw, elh));
	cvSetImageROI(right, cvRect(erx+fx, ery+fy+(fh/5.5), erw, erh));
	// Draw green rectangles around the eyes on the original image
	cvRectangle(img, cvPoint(elx, ely), cvPoint(elx + elw, ely + elh), CV_RGB(0, 255, 0), 1, 8, 0);
	cvRectangle(img, cvPoint(erx, ery), cvPoint(erx + erw, ery + erh), CV_RGB(0, 255, 0), 1, 8, 0);

	// Find and draw the contours on both of the eyes
	cvFindContours(left, sto1, &con1);
	cvFindContours(right, sto2, &con2);
	cvZero(left);
	cvZero(right);
	cvDrawContours(left, con1, cvScalarAll(255), cvScalarAll(255), 100);
	cvDrawContours(right, con2, cvScalarAll(255), cvScalarAll(255), 100);

	// Show the eye regions with contours
	cvNamedWindow("Countors1",1);
	cvShowImage("Countors1",left);
	cvWaitKey();
	cvDestroyWindow("Countors1");
	cvNamedWindow("Countors2",1);
	cvShowImage("Countors2",right);
	cvWaitKey();
	cvDestroyWindow("Countors2");

	// Find the iris width on the eyes
	findIris(1);
	findIris(2);
	if(ilwidth == 0 || irwidth == 0) return false;
cout << "\n" << ilmin << " " << ilmax << " " << ilwidth << "-" << irmin << " " << irmax << " " << irwidth << endl;

	// Estimate the PD
	estimatePD();

	cvResetImageROI(left);
	cvResetImageROI(right);
	cvClearMemStorage(sto1);
	cvClearMemStorage(sto2);

	cvResetImageROI(img);
	cvClearMemStorage(storage);


	cvNamedWindow("Image:",1);
	cvShowImage("Image:",img);
	cvWaitKey();
	cvDestroyWindow("Image:");

	return true;
}

int main(int argc, const char** argv){
	if(argc > 1) {
		img = cvLoadImage(argv[1]);
		detectEyes();
		cvReleaseImage(&img);
	}
	else {
		cout << "Error: must include image filename as command line argument" << endl;
	}
	return 0;
}

/* g++ -o my_example my_example.cpp `pkg-config opencv --cflags --libs` */
/* 165 * (11.8 / 30) = 64.9 */
