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

void precomputeVertSeams(cv::Mat & infile)
{
  using namespace cv;

  Mat infileGray;
  cvtColor(infile, infileGray, CV_BGR2GRAY);
  int xSize = infile.size().width;
  int ySize = infile.size().height;

  Mat gradMat(ySize, xSize, infileGray.type());
  produceGradient(infileGray, gradMat);
  //imwrite("gradient.jpg", gradMat);

  Vert::precomputeSeams(infile, gradMat);
}

void precomputeHoriSeams(cv::Mat & infile)
{
  using namespace cv;

  Mat infileGray;
  cvtColor(infile, infileGray, CV_BGR2GRAY);
  int xSize = infile.size().width;
  int ySize = infile.size().height;

  Mat gradMat(ySize, xSize, infileGray.type());
  produceGradient(infileGray, gradMat);
  //imwrite("gradient.jpg", gradMat);

  Hori::precomputeSeams(infile, gradMat);
}

cv::Mat runVertFastCarve(cv::Mat & infile, int newWidth)
{
  int ySize = infile.size().height;

  cv::Mat vertMat(ySize, newWidth, infile.type());
  Vert::runSeamRemove(infile, vertMat, newWidth, 0);
  return vertMat;
}

cv::Mat runHoriFastCarve(cv::Mat & infile, int newHeight)
{
  int xSize = infile.size().width;

  cv::Mat horiMat(newHeight, xSize, infile.type());
  Hori::runSeamRemove(infile, horiMat, 0, newHeight);
  return horiMat;
}

}

