#include <iostream>
#include <iomanip>
#include <map>

#if (defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__) || defined(__WINDOWS__) || (defined(__APPLE__) & defined(__MACH__)))
#include <cv.h>
#include <highgui.h>
#else
#include <opencv/cv.h>
#include <opencv/highgui.h>
#endif

#include <cvblob.h>
using namespace cvb;

int main()
{
  CvTracks tracks;

  cvNamedWindow("line_tracking", CV_WINDOW_AUTOSIZE);

  CvCapture *capture = cvCaptureFromCAM(0);
  cvGrabFrame(capture);
  IplImage *img = cvRetrieveFrame(capture);

  CvSize imgSize = cvGetSize(img);

  IplImage *frame = cvCreateImage(imgSize, img->depth, img->nChannels);

  unsigned int blobNumber = 0;

  bool quit = false;
  while (!quit&&cvGrabFrame(capture))
  {
    IplImage *img = cvRetrieveFrame(capture);

    cvConvertScale(img, frame, 1 , 0);

    IplImage *segmentated = cvCreateImage(imgSize, 8, 1);
    
    // Detecting red pixels:
    // (This is very slow, use direct access better...)
    for (unsigned int j=0; j<imgSize.height; j++)
      for (unsigned int i=0; i<imgSize.width; i++)
      {
	      CvScalar c = cvGet2D(frame, j, i);

	      double b = ((double)c.val[0])/255.;
	      double g = ((double)c.val[1])/255.;
	      double r = ((double)c.val[2])/255.;
	      unsigned char f = 255*((r>0.2+g)&&(r>0.2+b));

	      cvSet2D(segmentated, j, i, CV_RGB(f, f, f));
      }

    IplImage *labelImg = cvCreateImage(cvGetSize(frame), IPL_DEPTH_LABEL, 1);

    CvBlobs blobs;
    unsigned int result = cvLabel(segmentated, labelImg, blobs);

    cvFilterByArea(blobs, 500, 1000000);
    cvRenderBlobs(labelImg, blobs, frame, frame, CV_BLOB_RENDER_BOUNDING_BOX);


    if (blobs.size() == 3) {

      std::map<CvLabel, CvBlob * >::const_iterator itr;
      CvPoint2D64f ab, bc;
      int compt = 0;
      int x1, y1, x2, y2, x3, y3;

      for(itr = blobs.begin(); itr != blobs.end(); ++itr){
        unsigned int x,y;
        x = (*itr).second->centroid.x;
        y = (*itr).second->centroid.y;
        
        if (compt == 0) {
          x1 = x;
          y1 = y;
        } else if (compt == 1) {
          x2 = x;
          y2 = y;
        } else {
          x3 = x;
          y3 = y;
        }
        compt++;
      }

      ab.x = x1 - x2;
      ab.y = y1 - y2;
      bc.x = x2 - x3;
      bc.y = y2 - y3;
      
      //std::cout << "(" << ab.x << "," << ab.y << ") ; (" << bc.x << "," << bc.y << ") =>" << ab.x * bc.y - ab.y * bc.x << "\n";
      
      bool aligned = abs(ab.x * bc.y - ab.y * bc.x) < 2000.;
      
      CvScalar const color = aligned ? CV_RGB(0, 255, 0) : CV_RGB(255, 0, 0);
      
      //std::cout << aligned << "\n";
      
      CvContourPolygon* rectangle;
      rectangle->push_back(cvPoint(x1,y1));
      rectangle->push_back(cvPoint(x2,y2));
      rectangle->push_back(cvPoint(x3,y3));
      rectangle->push_back(cvPoint(x2,y2));
      rectangle->push_back(cvPoint(x1,y1));
      
      cvRenderContourPolygon(rectangle, frame, color);
      rectangle->clear();
    }


    cvShowImage("red_object_tracking", frame);

    cvReleaseImage(&labelImg);
    cvReleaseImage(&segmentated);

    char k = cvWaitKey(10)&0xff;
    switch (k)
    {
      case 27:
      case 'q':
      case 'Q':
        quit = true;
        break;
    }

    cvReleaseBlobs(blobs);
  }

  cvReleaseImage(&frame);

  cvDestroyWindow("line_tracking");

  return 0;
}
