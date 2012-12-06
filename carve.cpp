#include <stdlib.h>
#include <algorithm>
#include <cstdio>
#include <vector>
#include <utility>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "carve_vert.h"
#include "carve_hori.h"

namespace Carve
{

void produceGradient(cv::Mat & inMat, cv::Mat & outMat)
{
  int xSize = inMat.size().width;
  int ySize = inMat.size().height;

  for (int x = 0; x < xSize; x++)
  {
    for (int y = 0; y < ySize; y++)
    {
      char val;
      if (y != ySize-1 && x != xSize-1) {
        val = abs(int(inMat.at<uchar>(y,x)) - int(inMat.at<uchar>(y+1,x))) +
              abs(int(inMat.at<uchar>(y,x)) - int(inMat.at<uchar>(y,x+1)));
        val /= 2;
      } else {
        val = inMat.at<uchar>(y,x);
      }
      outMat.at<uchar>(y,x) = val;
    }
  }
}

void runFastCarve(std::string & fname, int newWidth, int newHeight)
{
  using namespace cv;

  Mat infile = imread(fname);
  Mat infileGray;
  cvtColor(infile, infileGray, CV_BGR2GRAY);
  int xSize = infile.size().width;
  int ySize = infile.size().height;

  if (newWidth > 2 * xSize || newHeight > 2 * ySize)
  {
    std::cout << "Can only expand up to twice the original size" << std::endl;
    return;
  }
  else if (newWidth < 0 || newHeight < 0)
  {
    std::cout << "Dimensions must be positive" << std::endl;
    return;
  }

  Mat gradMat(ySize, xSize, infileGray.type());
  produceGradient(infileGray, gradMat);
  //imwrite("gradient.jpg", gradMat);

  Mat vertMat(ySize, newWidth, infile.type());
  Vert::runFastCarve(infile, vertMat, gradMat, newWidth, newHeight);

  Mat vertGray;
  cvtColor(vertMat, vertGray, CV_BGR2GRAY);
  Mat vertGradMat(ySize, newWidth, vertGray.type());
  produceGradient(vertGray, vertGradMat);

  Mat outMat(newHeight, newWidth, infile.type());
  Hori::runFastCarve(vertMat, outMat, vertGradMat, newWidth, newHeight);
  imwrite("output.jpg", outMat);
}

}
