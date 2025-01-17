
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    /* float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    } */
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    double medEucDist = 0.0;
    const double distThrsh = 40.0;
    vector <cv::DMatch> matchesInBB;
    vector <float> eucDists;

    // Loop through all the keypoint matches and check if they are in both prev and current frame
    for (const auto matches : kptMatches)
    {
        // Check if the points are in both current and previous keypoint frame
        if (boundingBox.roi.contains(kptsCurr[matches.trainIdx].pt) && 
            boundingBox.roi.contains(kptsPrev[matches.queryIdx].pt))
        {
            matchesInBB.push_back(matches);
            // Calculate euclidian distances
            eucDists.push_back(cv::norm(kptsPrev[matches.queryIdx].pt - kptsCurr[matches.trainIdx].pt));
        }
    }

    // Calculate the median euclidean distance between matched points within ROI
    sort(eucDists.begin(),eucDists.end());
    if (eucDists.size() % 2 == 0)
    {
        medEucDist = (eucDists[(eucDists.size()-1)/2] + eucDists[eucDists.size()/2])/2.0;
    }
    else
    {
        medEucDist = eucDists[eucDists.size()/2];
    }

    // Use the euclidian distance for each point in the bounding box to the median dist to filter outliers
    for (auto matchInBB : matchesInBB)
    {
        if (fabs(cv::norm(kptsPrev[matchInBB.queryIdx].pt - kptsCurr[matchInBB.trainIdx].pt) - medEucDist) <= distThrsh)
        {
            boundingBox.kptMatches.push_back(matchInBB);
        }
    } 
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC)
{
    /* Used this code from the lesson task */
    /* Compute distance ratios between all matched keypoints */
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame

    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer keypoint loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner keypoint loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    // compute camera-based TTC from distance ratios
    double meanDistRatio = std::accumulate(distRatios.begin(), distRatios.end(), 0.0) / distRatios.size();

    // TODO: STUDENT TASK (replacement for meanDistRatio)
    double medianDistRatio = 0.0;

    // 1. To compute median we need to know if the vector has even or odd number of elements
    // 2. We also need to sort the vector
    sort(distRatios.begin(),distRatios.end());
    if (distRatios.size() % 2 == 0)
    {
        medianDistRatio = (distRatios[(distRatios.size()-1)/2] + distRatios[distRatios.size()/2])/2.0;
    }
    else
    {
        medianDistRatio = distRatios[distRatios.size()/2];
    }

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medianDistRatio);
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // Used this code from lesson
    double laneWidth = 3.7;
    double medEucDistPrev = 0.0;
    double medEucDistCurr = 0.0;

    vector <double> prevLidarPointsX,currLidarPointsX;

    // find closest distance to Lidar points within ego lane
    double minXPrev = 1e9, minXCurr = 1e9;

    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        // Find lidar points only in the ego lane
        if (abs(it->y) < laneWidth / 2.0)
        {
            prevLidarPointsX.push_back(it->x);
        }
    }

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        // Find lidar points only in the ego lane
        if (abs(it->y) < laneWidth / 2.0)
        {
            currLidarPointsX.push_back(it->x);
        }
    }

    // Sort the previous and current lidar x points
    sort(prevLidarPointsX.begin(),prevLidarPointsX.end());
    sort(currLidarPointsX.begin(),currLidarPointsX.end());

    // Calculate previous lidar points euclidean distance
    if (prevLidarPointsX.size() % 2 == 0)
    {
        medEucDistPrev = (prevLidarPointsX[(prevLidarPointsX.size()-1)/2] + prevLidarPointsX[prevLidarPointsX.size()/2])/2.0;
    }
    else
    {
        medEucDistPrev = prevLidarPointsX[prevLidarPointsX.size()/2];
    }

    // Calculate current lidar points euclidean distance
    if (currLidarPointsX.size() % 2 == 0)
    {
        medEucDistCurr = (currLidarPointsX[(currLidarPointsX.size()-1)/2] + currLidarPointsX[currLidarPointsX.size()/2])/2.0;
    }
    else
    {
        medEucDistCurr = currLidarPointsX[currLidarPointsX.size()/2];
    }

    // compute TTC from both measurements
    TTC = medEucDistCurr * (1.0 / frameRate) / (medEucDistPrev - medEucDistCurr);
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    multimap <int, int> matchBoxes;

    /* For every keypoint match in matches verify if it's in both current and previous frame bounding box. */
    /* If yes, collect the boxID's for current and previous Frame bounding boxes */
    for (const auto keyPtMatch : matches)
    {
        int prevBoxID = -1;
        int currBoxID = -1;

        /* Loop through current frame bounding box and check for matched keypoints */
        for (const auto currBox : currFrame.boundingBoxes)
        {
            if (currBox.roi.contains(currFrame.keypoints[keyPtMatch.trainIdx].pt))
            {
                currBoxID = currBox.boxID;
            }
        }
        /* Loop through previous frame bounding box and check for matched keypoints */
        for (const auto prevBox : prevFrame.boundingBoxes)
        {
            if (prevBox.roi.contains(prevFrame.keypoints[keyPtMatch.queryIdx].pt))
            {
                prevBoxID = prevBox.boxID;
            }
            
        }
        matchBoxes.insert({currBoxID,prevBoxID});

        /* Find best match bounding box in the previous frame for the current bounding box frame */
        for (size_t i = 0; i < currFrame.boundingBoxes.size(); i++)
        {
            /* make sure the prev frame bounding box size matches */
            auto matchBoxesPair = matchBoxes.equal_range(i);

            vector <int> currBoxCount(prevFrame.boundingBoxes.size(),0);

            for (auto pr = matchBoxesPair.first; pr != matchBoxesPair.second; ++pr)
            {
                if ((*pr).second != -1)
                {
                    currBoxCount[(*pr).second] += 1;
                }  
            }

            // find the position of best prev box which has highest number of keypoint.
            int maxPosition = distance(currBoxCount.begin(),
                                        max_element(currBoxCount.begin(), currBoxCount.end()));
            bbBestMatches.insert({maxPosition, i});
        }
    }
}
