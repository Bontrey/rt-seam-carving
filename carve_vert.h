#ifndef _CARVE_VERT

typedef std::vector<std::vector<int> > Vector2D;

namespace Vert
{

struct Match
{
  Match() {}
  Match(int destX, int edgeWeight)
  {
    nextX = destX;
    weight = edgeWeight;
  }

  int nextX;
  int weight;
};
typedef std::vector<std::vector<Match> > MatchMat;

void precomputeSeams(cv::Mat & infile, cv::Mat & gradMat);
void runSeamRemove(cv::Mat & infile, cv::Mat & outMat,
                   int newWidth, int newHeight);
}

#define _CARVE_VERT

#endif

