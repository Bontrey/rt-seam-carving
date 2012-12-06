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

namespace Vert
{

/*
int calculateWeight(cv::Mat & energyMat, int row, int col1, int col2)
{
  if (col1 < 0 || col2 < 0)
    return -9999;

  return int(energyMat.at<uchar>(row,col1)) *
         int(energyMat.at<uchar>(row+1,col2));
}
*/
int calculateWeight(cv::Mat & energyMat, Vector2D & AMat, Vector2D & MMat,
                    int row, int col1, int col2)
{
  if (col1 < 0 || col2 < 0)
    return -9999;

  int a = AMat[row][col1];
  int m = MMat[row+1][col2];
  return a * m;
}

void matchRow(cv::Mat & energyMat, Vector2D & AMat, Vector2D & MMat,
              MatchMat & matchMat, int k)
{
  int xSize = energyMat.size().width;

  std::vector<int> F(xSize);
  std::vector<bool> F_is_case1(xSize);

  std::vector<int> weights1(xSize);
  std::vector<int> weights2(xSize);
  std::vector<int> weights3(xSize);

  for (int i = 0; i < xSize; i++)
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
  for (int i = xSize-1; i >= 0; i--)
  {
    if (F_is_case1[i])
    {
      matchMat[k][i].nextX = i;
      matchMat[k][i].weight = weights1[i];
      AMat[k+1][i] = energyMat.at<uchar>(k+1,i) + AMat[k][i];
    }
    else
    {
      matchMat[k][i].nextX = i-1;
      matchMat[k][i].weight = weights2[i];
      AMat[k+1][i-1] = energyMat.at<uchar>(k+1,i-1) + AMat[k][i];
      matchMat[k][i-1].nextX = i;
      matchMat[k][i-1].weight = weights3[i];
      AMat[k+1][i] = energyMat.at<uchar>(k+1,i) + AMat[k][i-1];
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

  for (int i = 0; i < xSize; i++)
  {
    int cost = 0;
    int x = i;
    seams[i][0] = x;
    for (int j = 0; j < ySize; j++)
    {
      cost += matches[j][x].weight;
      x = matches[j][x].nextX;
      seams[i][j+1] = x;
    }
    costs[i].first = cost;
    costs[i].second = i;
  }

  std::sort(costs.begin(), costs.end());
}

void calculateMMat(cv::Mat & gradMat, Vector2D & MMat)
{
  int xSize = gradMat.size().width;
  int ySize = gradMat.size().height;

  for (int i = 0; i < xSize; i++)
  {
    MMat[ySize-1][i] = gradMat.at<uchar>(ySize-1,i);
  }
  for (int j = ySize-2; j >= 0; j--)
  {
    MMat[j][0] = gradMat.at<uchar>(j,0) +
                                  std::min(MMat[j+1][0], MMat[j+1][1]);
    for (int i = 1; i < xSize-1; i++)
    {
      int m1 = MMat[j+1][i-1];
      int m2 = MMat[j+1][i];
      int m3 = MMat[j+1][i+1];
      MMat[j][i] = gradMat.at<uchar>(j,i) + std::min(m1,std::min(m2,m3));
    }
    MMat[j][xSize-1] = gradMat.at<uchar>(j,xSize-1) +
                        std::min(MMat[j+1][xSize-1], MMat[j+1][xSize-2]);
  }
}

void initializeAMat(cv::Mat & gradMat, Vector2D & AMat)
{
  int xSize = gradMat.size().width;

  for (int i = 0; i < xSize; i++)
    AMat[0][i] = gradMat.at<uchar>(0,i);
}

void calculateMatchings(cv::Mat & gradMat, MatchMat & matches)
{
  int xSize = gradMat.size().width;
  int ySize = gradMat.size().height;

  Vector2D MMat(ySize, std::vector<int>(xSize));
  calculateMMat(gradMat, MMat);

  Vector2D AMat(ySize, std::vector<int>(xSize));
  initializeAMat(gradMat, AMat);

  for (int i = 0; i < ySize-1; i++)
  {
    matchRow(gradMat, AMat, MMat, matches, i);
  }
}

void calculateWeightMat(std::vector<std::pair<int,int> > & costs,
                        MatchMat & matches, cv::Mat & weightMat)
{
  int xSize = weightMat.size().width;
  int ySize = weightMat.size().height;

  for (int i = 0; i < costs.size(); i++)
  {
    int x = costs[i].second;
    double alpha = double(i) / double(xSize-1);

    for (int j = 0; j < ySize; j++)
    {
      weightMat.at<cv::Vec3b>(j,x)[0] = 0;
      weightMat.at<cv::Vec3b>(j,x)[1] = 255.0 * (1.0-alpha);
      weightMat.at<cv::Vec3b>(j,x)[2] = 255.0 * alpha;
      x = matches[j][x].nextX;
    }
  }
}

void seamRemove(cv::Mat & infile, cv::Mat & outMat,
                std::vector<std::pair<int,int> > costs,
                Vector2D & seams, int newWidth)
{
  using namespace cv;

  int xSize = infile.size().width;
  int ySize = infile.size().height;
  int numRemoves = xSize - newWidth;
  bool expanding = false;
  if (numRemoves < 0)
  {
    expanding = true;
    numRemoves *= -1;
  }

  std::vector<std::vector<bool> > flagged(ySize,
                          std::vector<bool>(xSize, false));
  for (int i = 0; i < numRemoves; i++)
  {
    int seamIdx = costs[i].second;
    for (int j = 0; j < ySize; j++)
    {
      int x = seams[seamIdx][j];
      flagged[j][x] = true;
    }
  }

  if (!expanding)
  {
    for (int j = 0; j < ySize; j++)
    {
      int x = 0;
      for (int i = 0; i < xSize; i++)
      {
        if (!flagged[j][i])
        {
          outMat.at<Vec3b>(j,x) = infile.at<Vec3b>(j,i);
          x += 1;
        }
      }
    }
  }
  else
  {
    for (int j = 0; j < ySize; j++)
    {
      int x = 0;
      for (int i = 0; i < xSize; i++)
      {
        outMat.at<Vec3b>(j,x) = infile.at<Vec3b>(j,i);
        if (flagged[j][i])
        {
          x += 1;
          outMat.at<Vec3b>(j,x) = infile.at<Vec3b>(j,i);
        }
        x += 1;
      }
    }
  }
}

void runFastCarve(cv::Mat & infile, cv::Mat & outMat, cv::Mat & gradMat,
                  int newWidth, int newHeight)
{
  using namespace cv;

  int xSize = infile.size().width;
  int ySize = infile.size().height;

  MatchMat matches(ySize, std::vector<Match>(xSize));
  calculateMatchings(gradMat, matches);

  std::vector<std::pair<int,int> > costs(xSize);
  Vector2D seams(xSize, std::vector<int>(ySize));
  calculateSeamCosts(matches, costs, seams);

  //Mat weightMat(ySize, xSize, infile.type());
  //calculateWeightMat(costs, matches, weightMat);
  //imwrite("weights_vert.jpg", weightMat);

  seamRemove(infile, outMat, costs, seams, newWidth);
}

} // namespace

