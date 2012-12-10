#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "carve.h"

int main(int argc, char *argv[])
{
  if (argc < 4)
  {
    std::cout << "usage: ./carve <image> <new_width> <new_height> \
                  \nMake the dimension 0 to ignore it." << std::endl;
    return 0;
  }
  std::string fname(argv[1]);
  int newWidth = atoi(argv[2]);
  int newHeight = atoi(argv[3]);
  char buf[512];

  cv::Mat infile = cv::imread(fname);

  int xSize = infile.size().width;
  int ySize = infile.size().height;
  if (newWidth > 2 * xSize || newHeight > 2 * ySize)
  {
    std::cout << "Can only expand up to twice the original size"
              << std::endl;
    return 0;
  }
  else if (newWidth < 0 || newHeight < 0)
  {
    std::cout << "Dimensions must be non-negative" << std::endl;
    return 0;
  }
  else if (newWidth == 0 && newHeight == 0)
  {
    std::cout << "At least one dimension must be non-zero" << std::endl;
    return 0;
  }

  Carve::precomputeVertSeams(infile);
  cv::Mat interMat = Carve::runVertFastCarve(infile, newWidth);

  Carve::precomputeHoriSeams(interMat);
  cv::Mat outMat = Carve::runHoriFastCarve(interMat, newHeight);

  imwrite("output.jpg", outMat);

  //namedWindow( "Display window", CV_WINDOW_AUTOSIZE );
  //imshow( "Display window", b );
  
  //waitKey(0);

  return 0;
}
