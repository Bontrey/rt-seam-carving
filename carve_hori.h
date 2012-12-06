#ifndef _CARVE_HORI

typedef std::vector<std::vector<int> > Vector2D;

namespace Hori
{

struct Match
{
  Match() {}
  Match(int destY, int edgeWeight)
  {
    nextY = destY;
    weight = edgeWeight;
  }

  int nextY;
  int weight;
};
typedef std::vector<std::vector<Match> > MatchMat;

void runFastCarve(cv::Mat & infile, cv::Mat & outMat, cv::Mat & gradMat,
                  int newWidth, int newHeight);
}

#define _CARVE_HORI

#endif

