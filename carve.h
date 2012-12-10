#ifndef _CARVE

namespace cv
{
  class Mat;
}

namespace Carve
{
  void precomputeVertSeams(cv::Mat & infile);
  void precomputeHoriSeams(cv::Mat & infile);
  cv::Mat runVertFastCarve(cv::Mat & infile, int newWidth);
  cv::Mat runHoriFastCarve(cv::Mat & infile, int newHeight);
}

#define _CARVE
#endif

