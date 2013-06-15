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
* Create Date:      2008-04-03                                              *
* Revise Date:      2012-03-22                                              *
*****************************************************************************/

#include <fstream>
#include <string>
#include <set>
#include <algorithm>

#include "VO_Point2DDistributionModel.h"
#include "VO_CVCommon.h"



/**
 * @author         JIA Pei
 * @version        2010-02-07
 * @brief          build point model
 * @param          allAlignedShapes            Input - input aligned points.
 * @return        void
*/
void VO_Point2DDistributionModel::VO_BuildPointDistributionModel( const vector<VO_Shape>& allAlignedShapes )
{
    unsigned int NbOfSamples        = allAlignedShapes.size();
    unsigned int NbOfPoints         = allAlignedShapes[0].GetNbOfPoints();
    this->m_VONormalizedEllipses.resize(NbOfPoints);
    Mat_<float> matAlignedPoints    = Mat_<float>::zeros(NbOfSamples, 2);
    Mat_<float> matAlignedMeanPoint = Mat_<float>::zeros(1, 2);
    
    for(unsigned int i = 0; i < NbOfPoints; ++i)
    {
        for(unsigned int j = 0; j < NbOfSamples; ++j)
        {
            matAlignedPoints(j, 0)  = allAlignedShapes[j].GetA2DPoint(i).x;
            matAlignedPoints(j, 1)  = allAlignedShapes[j].GetA2DPoint(i).y;
        }
        PCA pca = PCA(matAlignedPoints, matAlignedMeanPoint, CV_PCA_DATA_AS_ROW, 2);
        float xx = pca.eigenvectors.at<float>(0,0);
        float yy = pca.eigenvectors.at<float>(0,1);
        float theta = 0.0;
        if( fabs(xx) < FLT_MIN )    theta = (float)(CV_PI/2.0);
        else theta = atan(yy/xx)/safeDoubleToFloat(CV_PI*180.0);
        Point2f pt = Point2f(    pca.mean.at<float>(0,0), 
                                pca.mean.at<float>(0,1));
        this->m_VONormalizedEllipses[i] = VO_Ellipse(   pt,
                                                        3.0f*sqrt(pca.eigenvalues.at<float>(0,0)),
                                                        3.0f*sqrt(pca.eigenvalues.at<float>(1,0)),
                                                        0,
                                                        360,
                                                        theta );
    }
}


/**
 * @author     JIA Pei
 * @version    2010-06-07
 * @brief      Constrain single point
 * @param      pt       Input and Output - the input and output point
*/
void VO_Point2DDistributionModel::VO_ConstrainSinglePoint(Point2f& pt, const VO_Ellipse& ellipse)
{
    if( ! const_cast<VO_Ellipse&>(ellipse).IsPointWithinEllipse(pt) )
    {
        Point2f boundpt = const_cast<VO_Ellipse&>(ellipse).FindIntersectPointOnBoundary(pt);
        pt = boundpt;
    }
}


/**
 * @author      JIA Pei
 * @version     2010-06-07
 * @brief       Scale all input ellipses
 * @param       iEllipses    - input
 * @param       oEllipses    - output
*/
void VO_Point2DDistributionModel::VO_ScalePDMEllipses(const vector<VO_Ellipse>& iEllipses, float scale, vector<VO_Ellipse>& oEllipses)
{
    unsigned int NbOfEllipses = iEllipses.size();
    for(unsigned int i = 0; i < NbOfEllipses; i++)
    {
        oEllipses[i] = const_cast< vector<VO_Ellipse>& >(iEllipses)[i]*scale;
    }
}


/**
 * @author     JIA Pei
 * @version    2010-06-07
 * @brief      Constrain all points respetively
 * @param      ioShape      Input and Output - the input and output shape
*/
void VO_Point2DDistributionModel::VO_ConstrainAllPoints(VO_Shape& ioShape)
{
    unsigned int NbOfPoints = ioShape.GetNbOfPoints();
    Point2f pt;

    for(unsigned int i = 0; i < NbOfPoints; i++)
    {
        pt = ioShape.GetA2DPoint(i);
        VO_Point2DDistributionModel::VO_ConstrainSinglePoint( pt, this->m_VONormalizedEllipses[i] );
        ioShape.SetA2DPoint(pt, i);
    }
}


/**
 * @author     JIA Pei
 * @version    2010-02-22
 * @brief      Save ASM to a specified folder
 * @param      fd       Input - the folder that ASM to be saved to
*/
void VO_Point2DDistributionModel::VO_Save(const string& fd)
{
    // create Point2DDistributionModel subfolder for just Point2DDistributionModel data
    string fn = fd+"/Point2DDistributionModel";
    MakeDirectory(fn);

    fstream fp;
    string tempfn;
    
    // Point2DDistributionModel
    tempfn = fn + "/Point2DDistributionModel" + ".txt";
    fp.open(tempfn.c_str (), ios::out);
    fp << "NbOfPoints" << endl << this->m_VONormalizedEllipses.size() << endl;
    fp.close();fp.clear();
    
    // m_VONormalizedEllipses
    tempfn = fn + "/m_VONormalizedEllipses" + ".txt";
    fp.open(tempfn.c_str (), ios::out);
    fp << "m_VONormalizedEllipses" << endl;
    for(unsigned int i = 0; i < this->m_VONormalizedEllipses.size(); i++)
    {
        fp << this->m_VONormalizedEllipses[i] << endl;    
    }
    fp.close();fp.clear();
}


/**
* @author     JIA Pei
* @version    2010-02-22
* @brief      Load all ASM data from a specified folder
* @param      fd        Input - the folder that ASM to be loaded from
*/
void VO_Point2DDistributionModel::VO_Load(const string& fd)
{
    this->VO_LoadParameters4Fitting(fd);
}


/**
* @author     JIA Pei
* @version    2010-02-22
* @brief      Load all ASM data from a specified folder for later fitting
* @param      fd        Input - the folder that ASM to be loaded from
*/
void VO_Point2DDistributionModel::VO_LoadParameters4Fitting(const string& fd)
{
    string fn = fd+"/Point2DDistributionModel";
    if (!MakeDirectory(fn) )
    {
        cout << "Point2DDistributionModel subfolder is not existing. " << endl;
        exit(EXIT_FAILURE);
    }

    ifstream fp;
    string tempfn;
    string temp;
    unsigned int NbOfPoints;
    
    // Point2DDistributionModel
    tempfn = fn + "/Point2DDistributionModel" + ".txt";
    SafeInputFileOpen(fp, tempfn);
    fp >> temp >> NbOfPoints;
    fp.close();fp.clear();
    
    this->m_VONormalizedEllipses.resize(NbOfPoints);

    // m_VONormalizedEllipses
    tempfn = fn + "/m_VONormalizedEllipses" + ".txt";
    SafeInputFileOpen(fp, tempfn);
    fp >> temp;
    this->m_VONormalizedEllipses.resize(NbOfPoints);
    for(unsigned int i = 0; i < NbOfPoints; i++)
    {
        fp >> this->m_VONormalizedEllipses[i];
    }
    fp.close();fp.clear();
}




