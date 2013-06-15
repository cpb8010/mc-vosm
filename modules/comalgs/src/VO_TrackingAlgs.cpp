/****************************************************************************
*                                                                           *
*   IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.       *
*                                                                           *
*   By downloading, copying, installing or using the software you agree to  *
*   this license. If you do not agree to this license, do not download,     *
*   install, copy or use the software.                                      *
*                                                                           *
*                           License Agreement                               *
*                   For Vision Open Statistical Models                      *
*                                                                           *
*   Copyright (C):      2006~2012 by JIA Pei, all rights reserved.          *
*                                                                           *
*   VOSM is free software under the terms of the GNU Lesser General Public  *
*   License (GNU LGPL) as published by the Free Software Foundation; either *
*   version 3.0 of the License, or (at your option) any later version.      *
*   You can use it, modify it, redistribute it, etc; and redistribution and *
*   use in source and binary forms, with or without modification, are       *
*   permitted provided that the following conditions are met:               *
*                                                                           *
*   a) Redistribution's of source code must retain this whole paragraph of  *
*   copyright notice, including this list of conditions and all the         *
*   following contents in this  copyright paragraph.                        *
*                                                                           *
*   b) Redistribution's in binary form must reproduce this whole paragraph  *
*   of copyright notice, including this list of conditions and all the      *
*   following contents in this copyright paragraph, and/or other materials  *
*   provided with the distribution.                                         *
*                                                                           *
*   c) The name of the copyright holders may not be used to endorse or      *
*   promote products derived from this software without specific prior      *
*   written permission.                                                     *
*                                                                           *
*   Any publications based on this code must cite the following five papers,*
*   technical reports and on-line materials.                                *
*   1) P. JIA, 2D Statistical Models, Technical Report of Vision Open       *
*   Working Group, 2st Edition, October 21, 2010.                           *
*   http://www.visionopen.com/members/jiapei/publications/pei_sm2dreport2010.pdf*
*   2) P. JIA. Audio-visual based HMI for an Intelligent Wheelchair.        *
*   PhD thesis, University of Essex, February, 2011.                        *
*   http://www.visionopen.com/members/jiapei/publications/pei_phdthesis2010.pdf*
*   3) T. Cootes and C. Taylor. Statistical models of appearance for        *
*   computer vision. Technical report, Imaging Science and Biomedical       *
*   Engineering, University of Manchester, March 8, 2004.                   *
*   http://www.isbe.man.ac.uk/~bim/Models/app_models.pdf                    *
*   4) I. Matthews and S. Baker. Active appearance models revisited.        *
*   International Journal of Computer Vision, 60(2):135--164, November 2004.*
*   http://www.ri.cmu.edu/pub_files/pub4/matthews_iain_2004_2/matthews_iain_2004_2.pdf*
*   5) M. B. Stegmann, Active Appearance Models: Theory, Extensions and     *
*   Cases, 2000.                                                            *
*   http://www2.imm.dtu.dk/~aam/main/                                       *
*                                                                           *
* Version:          0.4                                                     *
* Author:           JIA Pei                                                 *
* Contact:          jp4work@gmail.com                                       *
* URL:              http://www.visionopen.com                               *
* Create Date:      2010-11-04                                              *
* Revise Date:      2012-03-22                                              *
*****************************************************************************/


#include "VO_TrackingAlgs.h"


//int CTrackingAlgs::histSize[] = {hbins, sbins};   // histSize
int CTrackingAlgs::histSize[] = {hbins};            // histSize
float CTrackingAlgs::hranges[] = { 0, 180 };        // hue varies from 0 to 179, see cvtColor
float CTrackingAlgs::sranges[] = { 0, 256 };        // saturation varies from 0 (black-gray-white) to 255 (pure spectrum color)
//const float* CTrackingAlgs::ranges[] = { hranges, sranges };  // ranges
const float* CTrackingAlgs::ranges[] = { hranges };             // ranges
int CTrackingAlgs::channels[] = {0};


/**
* @brief    Initialize tracker
*/ 
void CTrackingAlgs::UpdateTracker(const Mat& img, const Rect& obj)
{
    switch(this->m_iTrackingMethod)
    {
    case CAMSHIFT:
        {
            this->m_bTrackerInitialized = 
                CTrackingAlgs::CamshiftUpdateTracker(img, obj, this->m_hist);
        }
        break;
    case KALMANFILTER:
        {
        }
        break;
    case PARTICLEFILTER:
        {
        }
        break;
    case ASMAAM:
        {
        }
        break;
    case NONE:
    default:
        {
            this->m_bTrackerInitialized =  false;
        }
        break;
    }
}


bool CTrackingAlgs::CamshiftUpdateTracker(  const Mat& img,
                                            const Rect& obj,
                                            MatND& hist)
{
    Mat hsv, hue, mask, backproject;
    cv::cvtColor( img, hsv, CV_BGR2HSV );

    int _vmin = CTrackingAlgs::vmin, _vmax = CTrackingAlgs::vmax;

    cv::inRange( hsv, Scalar(0,CTrackingAlgs::smin,MIN(_vmin,_vmax),0),
                Scalar(180,256,MAX(_vmin,_vmax),0), mask );
    vector<Mat> vhsv(3);
    cv::split( hsv, vhsv );
    vhsv[0].copyTo(hue);

    double max_val = 0.f;
    Mat roi         = hue(  Range(obj.y, obj.y+obj.height), 
                            Range(obj.x, obj.x+obj.width) );
    Mat roi_mask    = mask( Range(obj.y, obj.y+obj.height), 
                            Range(obj.x, obj.x+obj.width) );
    cv::calcHist(   &roi, 1, CTrackingAlgs::channels, roi_mask,
        hist, 1, CTrackingAlgs::histSize, CTrackingAlgs::ranges,
        true, // the histogram is uniform
        false );
    cv::minMaxLoc(hist, 0, &max_val, 0, 0);
    hist.convertTo(hist, hist.type(), (max_val ? 255. / max_val : 0.), 0);

    return true;
}


/** 
* @author   JIA Pei
* @version  2009-10-04
* @brief    Object Tracking
* @param    obj         Input - object to be tracked
* @param    img         Input - image to be searched within
* @param    smallSize   Input - smallSize
* @param    bigSize     Input - bigSize
* @return   detection time cost
*/
double CTrackingAlgs::Tracking( Rect& obj,
                                const Mat& img,
                                Size smallSize,
                                Size bigSize)
{
    double res = (double)cvGetTickCount();

    if(this->m_bTrackerInitialized)
    {
        switch(this->m_iTrackingMethod)
        {
        case CAMSHIFT:
            {
                CTrackingAlgs::CamshiftTracking(obj,
                                                img,
                                                this->m_hist,
                                                this->m_bObjectTracked,
                                                smallSize,
                                                bigSize);
                this->m_CVTrackedObjectRect = obj;
            }
            break;
        case KALMANFILTER:
            {
                CTrackingAlgs::KalmanTracking(  obj,
                                                img,
                                                this->m_bObjectTracked,
                                                smallSize,
                                                bigSize);
                this->m_CVTrackedObjectRect = obj;
            }
            break;
        case PARTICLEFILTER:
            {
                CTrackingAlgs::ParticleFilterTracking(
                                                    obj,
                                                    img,
                                                    this->m_bObjectTracked,
                                                    smallSize,
                                                    bigSize);
                this->m_CVTrackedObjectRect = obj;
            }
            break;
        case ASMAAM:
            {
                CTrackingAlgs::ASMAAMTracking(  obj,
                                                img,
                                                this->m_bObjectTracked,
                                                smallSize,
                                                bigSize);
                this->m_CVTrackedObjectRect = obj;
            }
            break;
        case NONE:
        default:
            {
                this->m_bObjectTracked         = false;
                this->m_bTrackerInitialized    = false;
            }
            break;
        }
    }
    else
    {
        this->UpdateTracker(img, obj);
    }

    res = ((double)cvGetTickCount() - res)
        / ((double)cvGetTickFrequency()*1000.);
    return res;
}


/** 
* @author   JIA Pei
* @version  2010-02-02
* @brief    Camshift Tracking
* @param    obj         Input - object to be tracked
* @param    img         Input - image to be searched within
* @param    isTracked   output - is this obj tracked?
* @param    smallSize   Input - the smallest possible object size
* @param    bigSize     Input - the biggest possible object size
* @return   detection time cost
*/
double CTrackingAlgs::CamshiftTracking( Rect& obj,
                                        const Mat& img,
                                        MatND& hist,
                                        bool& isTracked,
                                        Size smallSize,
                                        Size bigSize)
{
    double res = (double)cvGetTickCount();

    if(obj.x <= 0)    obj.x = 0;
    if(obj.y <= 0)    obj.y = 0;
    if(obj.x + obj.width > img.cols) obj.width = img.cols - obj.x;
    if(obj.y + obj.height > img.rows) obj.height = img.rows - obj.y;

    Rect trackwindow = obj;
    Mat hsv, hue, mask, backproject;
    cv::cvtColor( img, hsv, CV_BGR2HSV );

    int _vmin = CTrackingAlgs::vmin, _vmax = CTrackingAlgs::vmax;

    cv::inRange( hsv, Scalar(0,CTrackingAlgs::smin,MIN(_vmin,_vmax),0),
        Scalar(180,256,MAX(_vmin,_vmax),0), mask );
    vector<Mat> vhsv(3);
    cv::split( hsv, vhsv );
    vhsv[0].copyTo(hue);

    cv::calcBackProject( &hue, 1, CTrackingAlgs::channels, hist, backproject, 
                        CTrackingAlgs::ranges);
    cv::bitwise_and( backproject, mask, backproject );
    RotatedRect trackbox = CamShift( backproject, trackwindow, 
                        TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 
                        10, 1) );
    obj = trackwindow;

    //        cv::ellipse(img, trackbox, CV_RGB(255,0,0), 3, CV_AA);

    Point pt1 = Point( (int)(obj.x), (int)(obj.y) );
    Point pt2 = Point( (int)(obj.x + obj.width), 
                        (int)(obj.y + obj.height) );

    // Judge whether it is losing the object or not...
    if(obj.width >= bigSize.width 
        || obj.height >= bigSize.height
        || obj.width <= smallSize.width 
        || obj.height <= smallSize.height
        || pt1.x < FRAMEEDGE 
        || pt1.y < FRAMEEDGE
        || (pt2.x > (img.cols - FRAMEEDGE)) 
        || (pt2.y > (img.rows - FRAMEEDGE)))
    {
        isTracked = false;
        obj.x = obj.y = obj.width = obj.height = -1;
    }
    else
        isTracked = true;

    res = ((double)cvGetTickCount() - res)
        / ((double)cvGetTickFrequency()*1000.);
    return res;
}


double CTrackingAlgs::KalmanTracking(   Rect& obj,
                                        const Mat& img,
                                        bool& isTracked,
                                        Size smallSize,
                                        Size bigSize)
{
    double res = (double)cvGetTickCount();

    res = ((double)cvGetTickCount() - res)
        / ((double)cvGetTickFrequency()*1000.);
    return res;
}


double CTrackingAlgs::ParticleFilterTracking(   Rect& obj,
                                                const Mat& img,
                                                bool& isTracked,
                                                Size smallSize,
                                                Size bigSize)
{
    double res = (double)cvGetTickCount();

    res = ((double)cvGetTickCount() - res)
        / ((double)cvGetTickFrequency()*1000.);
    return res;
}


double CTrackingAlgs::ASMAAMTracking(   Rect& obj,
                                        const Mat& img,
                                        bool& isTracked,
                                        Size smallSize,
                                        Size bigSize)
{
    double res = (double)cvGetTickCount();

    res = ((double)cvGetTickCount() - res)
        / ((double)cvGetTickFrequency()*1000.);
    return res;
}


void CTrackingAlgs::VO_DrawTracking(Mat& ioImg, Scalar color)
{
    Rect curRect;
    Point lefttop, rightbottom;

    if ( this->m_bObjectTracked )
    {
        curRect = this->m_CVTrackedObjectRect;
        lefttop.x = cvRound(curRect.x);
        lefttop.y = cvRound(curRect.y);
        rightbottom.x = cvRound((curRect.x+curRect.width));
        rightbottom.y = cvRound((curRect.y+curRect.height));
        cv::rectangle(ioImg, lefttop, rightbottom, color, 2, 8, 0);
    }
}

