#include <iostream>

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

  Carve::runFastCarve(fname, newWidth, newHeight);

  //namedWindow( "Display window", CV_WINDOW_AUTOSIZE );
  //imshow( "Display window", b );
  
  //waitKey(0);

  return 0;
}
