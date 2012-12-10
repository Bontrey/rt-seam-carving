#include <stdlib.h>
#include <algorithm>
#include <cstdio>
#include <vector>
#include <utility>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "carve_hori.h"

namespace Hori
{

std::vector<std::pair<int,int> > costs;
Vector2D seams;

int calculateWeight(cv::Mat & energyMat, Vector2D & AMat, Vector2D & MMat,
                    int col, int row1, int row2)
{
  if (row1 < 0 || row2 < 0)
    return -9999;

  int a = AMat[row1][col];
  int m = MMat[row2][col+1];
  return a * m;
}

void matchCol(cv::Mat & energyMat, Vector2D & AMat, Vector2D & MMat,
              MatchMat & matchMat, int k)
{
  int ySize = energyMat.size().height;

  std::vector<int> F(ySize);
  std::vector<bool> F_is_case1(ySize);

  std::vector<int> weights1(ySize);
  std::vector<int> weights2(ySize);
  std::vector<int> weights3(ySize);

  for (int i = 0; i < ySize; i++)
  {
    int onePrev = i - 1; int twoPrev = i - 2;
    int F1Prev, F2Prev;
    F1Prev = onePrev >= 0 ? F[onePrev] : 0;
    F2Prev = twoPrev >= 0 ? F[twoPrev] : 0;

    weights1[i] = calculateWeight(energyMat, AMat, MMat, k, i, i);
    weights2[i] = calculateWeight(energyMat, AMat, MMat, k, i, i-1);
    weights3[i] = calculateWeight(energyMat, AMat, MMat, k, i-1, i);

    int case1 = F1Prev + weights1[i];
    int case2 = F2Prev + weights2[i] + weights3[i]; 
    F_is_case1[i] = (case1 >= case2);
    F[i] = std::max(case1, case2);
  }
  for (int i = ySize-1; i >= 0; i--)
  {
    if (F_is_case1[i])
    {
      matchMat[i][k].nextY = i;
      matchMat[i][k].weight = weights1[i];
      AMat[i][k+1] = energyMat.at<uchar>(i,k+1) + AMat[i][k];
    }
    else
    {
      matchMat[i][k].nextY = i-1;
      matchMat[i][k].weight = weights2[i];
      AMat[i-1][k+1] = energyMat.at<uchar>(i-1,k+1) + AMat[i][k];
      matchMat[i-1][k].nextY = i;
      matchMat[i-1][k].weight = weights3[i];
      AMat[i][k+1] = energyMat.at<uchar>(i,k+1) + AMat[i-1][k];
      i -= 1;
    }
  }
}

void calculateSeamCosts(MatchMat & matches,
                        std::vector<std::pair<int, int> > & costs,
                        Vector2D & seams)
{
  int xSize = matches[0].size();
  int ySize = matches.size();

  for (int j = 0; j < ySize; j++)
  {
    int cost = 0;
    int y = j;
    for (int i = 0; i < xSize; i++)
    {
      seams[j][i] = y;
      cost += matches[y][i].weight;
      y = matches[y][i].nextY;
    }
    costs[j].first = cost;
    costs[j].second = j;
  }

  std::sort(costs.begin(), costs.end());
}

void calculateMMat(cv::Mat & gradMat, Vector2D & MMat)
{
  int xSize = gradMat.size().width;
  int ySize = gradMat.size().height;

  for (int j = 0; j < ySize; j++)
  {
    MMat[j][xSize-1] = gradMat.at<uchar>(j,xSize-1);
  }
  for (int i = xSize-2; i >= 0; i--)
  {
    MMat[0][i] = gradMat.at<uchar>(0,i) +
                                  std::min(MMat[0][i+1], MMat[1][i+1]);
    for (int j = 1; j < ySize-1; j++)
    {
      int m1 = MMat[j-1][i+1];
      int m2 = MMat[j][i+1];
      int m3 = MMat[j+1][i+1];
      MMat[j][i] = gradMat.at<uchar>(j,i) + std::min(m1,std::min(m2,m3));
    }
    MMat[ySize-1][i] = gradMat.at<uchar>(ySize-1,i) +
                        std::min(MMat[ySize-1][i+1], MMat[ySize-2][i+1]);
  }
}

void initializeAMat(cv::Mat & gradMat, Vector2D & AMat)
{
  int ySize = gradMat.size().height;

  for (int i = 0; i < ySize; i++)
    AMat[i][0] = gradMat.at<uchar>(i,0);
}

void calculateMatchings(cv::Mat & gradMat, MatchMat & matches)
{
  int xSize = gradMat.size().width;
  int ySize = gradMat.size().height;

  Vector2D MMat(ySize, std::vector<int>(xSize));
  calculateMMat(gradMat, MMat);

  Vector2D AMat(ySize, std::vector<int>(xSize));
  initializeAMat(gradMat, AMat);

  for (int j = 0; j < xSize-1; j++)
  {
    matchCol(gradMat, AMat, MMat, matches, j);
  }
}

void calculateWeightMat(std::vector<std::pair<int,int> > & costs,
                        MatchMat & matches, cv::Mat & weightMat)
{
  int xSize = weightMat.size().width;
  int ySize = weightMat.size().height;

  for (int j = 0; j < costs.size(); j++)
  {
    int y = costs[j].second;
    double alpha = double(j) / double(ySize-1);

    for (int i = 0; i < xSize; i++)
    {
      weightMat.at<cv::Vec3b>(y,i)[0] = 0;
      weightMat.at<cv::Vec3b>(y,i)[1] = 255.0 * (1.0-alpha);
      weightMat.at<cv::Vec3b>(y,i)[2] = 255.0 * alpha;
      y = matches[y][i].nextY;
    }
  }
}

void seamRemove(cv::Mat & infile, cv::Mat & outMat,
                std::vector<std::pair<int,int> > costs,
                Vector2D & seams, int newHeight)
{
  using namespace cv;

  int xSize = infile.size().width;
  int ySize = infile.size().height;
  int numRemoves = ySize - newHeight;
  bool expanding = false;
  if (numRemoves < 0)
  {
    expanding = true;
    numRemoves *= -1;
  }

  std::vector<std::vector<bool> > flagged(ySize,
                          std::vector<bool>(xSize, false));
  for (int j = 0; j < numRemoves; j++)
  {
    int seamIdx = costs[j].second;
    for (int i = 0; i < xSize; i++)
    {
      int y = seams[seamIdx][i];
      flagged[y][i] = true;
    }
  }

  if (!expanding)
  {
    for (int i = 0; i < xSize; i++)
    {
      int y = 0;
      for (int j = 0; j < ySize; j++)
      {
        if (!flagged[j][i])
        {
          outMat.at<Vec3b>(y,i) = infile.at<Vec3b>(j,i);
          y += 1;
        }
      }
    }
  }
  else
  {
    for (int i = 0; i < xSize; i++)
    {
      int y = 0;
      for (int j = 0; j < ySize; j++)
      {
        outMat.at<Vec3b>(y,i) = infile.at<Vec3b>(j,i);
        if (flagged[j][i])
        {
          y += 1;
          outMat.at<Vec3b>(y,i) = infile.at<Vec3b>(j,i);
        }
        y += 1;
      }
    }
  }
}

void precomputeSeams(cv::Mat & infile, cv::Mat & gradMat)
{
  using namespace cv;

  int xSize = infile.size().width;
  int ySize = infile.size().height;

  MatchMat matches(ySize, std::vector<Match>(xSize));
  calculateMatchings(gradMat, matches);

  costs.resize(ySize);
  seams.resize(ySize, std::vector<int>(xSize));
  calculateSeamCosts(matches, costs, seams);

  //Mat weightMat(ySize, xSize, infile.type());
  //calculateWeightMat(costs, matches, weightMat);
  //imwrite("weights_hori.jpg", weightMat);
}

void runSeamRemove(cv::Mat & infile, cv::Mat & outMat,
                   int newWidth, int newHeight)
{
  seamRemove(infile, outMat, costs, seams, newHeight);
}

} // namespace

