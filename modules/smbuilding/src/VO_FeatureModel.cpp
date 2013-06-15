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


#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <set>

#include "VO_FeatureModel.h"
#include "VO_AnnotationDBIO.h"


VO_FeatureModel::VO_FeatureModel()
{
    this->init();
}


/** Initialization */
void VO_FeatureModel::init()
{
    this->m_iNbOfTotalFeatures          = 0;
    this->m_iNbOfEigenFeatureAtMost     = 0;
    this->m_iNbOfFeatureEigens          = 0;
    this->m_fTruncatedPercent_Feature   = 0.95f;
    this->m_vFeatures.clear();
    this->m_vNormalizedFeatures.clear();
}


VO_FeatureModel::~VO_FeatureModel()
{
    this->m_vFeatures.clear();
    this->m_vNormalizedFeatures.clear();
}


/**
 * @author      JIA Pei
 * @version     2010-02-10
 * @brief       Calculate point warping information
 * @param       iShape              Input     - the shape
 * @param       img                 Input     - image
 * @param       templateTriangles   Input     - the composed face template triangles
 * @param       warpInfo            Input     - warping information for all pixels in template face
 * @param       oFeature            Output     - the extracted features
 * @param       trm                 Input     - texture representation method
 * @return      bool                loading succeed or not?
*/
bool VO_FeatureModel::VO_LoadFeatures4OneFace(  const VO_Shape& iShape, 
                                                const Mat& img,
                                                const vector<VO_Triangle2DStructure>& templateTriangles,
                                                const vector<VO_WarpingPoint>& warpInfo,
                                                Mat_<float>& oFeature, 
                                                int trm)
{
    
    return true;
}


/**
 * @author      JIA Pei
 * @version     2010-02-05
 * @brief       Load Training data for texture model
 * @param       allLandmarkFiles4Training   Input - all landmark file names
 * @param       shapeinfoFileName           Input - all 
 * @param       allImgFiles4Training        Input - all input image file names
 * @param       channels                    Input - how to load each image
 * @return      void
*/
bool VO_FeatureModel::VO_LoadFeatureTrainingData(   const vector<string>& allLandmarkFiles4Training,
                                                    const vector<string>& allImgFiles4Training,
                                                    const string& shapeinfoFileName, 
                                                    unsigned int database,
                                                    unsigned int channels)
{
    // load auxiliary shape information
    VO_Shape2DInfo::ReadShape2DInfo(shapeinfoFileName, this->m_vShape2DInfo, this->m_FaceParts);
    CAnnotationDBIO::VO_LoadShapeTrainingData( allLandmarkFiles4Training, database, this->m_vShapes);
    
    this->m_vStringTrainingImageNames = allImgFiles4Training;
    this->m_vFeatures.resize(this->m_iNbOfSamples);
    this->m_vNormalizedFeatures.resize(this->m_iNbOfSamples);
    Mat img;
    
    for(unsigned int i = 0; i < this->m_iNbOfSamples; ++i)
    {
        if(channels == 1)
            img = imread ( allImgFiles4Training[i].c_str (), 0 );
        else if (channels == 3)
            img = imread ( allImgFiles4Training[i].c_str (), 1 );
        else
            cerr << "We can't deal with image channels not equal to 1 or 3!" << endl;

        double start = (double)cvGetTickCount();
        // Explained by JIA Pei -- Feature extraction over the whole image, so many methods.
        if ( !VO_FeatureModel::VO_LoadFeatures4OneFace( this->m_vShapes[i], 
                                                        img, 
                                                        this->m_vTemplateTriangle2D, 
                                                        this->m_vTemplatePointWarpInfo,
                                                        this->m_vFeatures[i], 
                                                        this->m_iTextureRepresentationMethod) )

        {
            cout << "Texture Fail to Load at image " << i << endl;
            return false;
        }

        double end = (double)cvGetTickCount();
        double elapsed = (end - start) / (cvGetTickFrequency()*1000.0);
    }

    return true;
}


/**
 * @author      JIA Pei
 * @version     2010-02-05
 * @brief       build Texture Model
 * @param       allLandmarkFiles4Training   Input - all training landmark files
 * @param       allImgFiles4Training        Input - all training image files
 * @param       shapeinfoFileName            Input - shape info file
 * @param       database                    Input - which database is it?
 * @param       channels                    Input - How many channels are to be used?
 * @param       trm                         Input - texture representation method
 * @param       TPShape                     Input - truncated percentage for shape model
 * @param       TPTexture                   Input - truncated percentage for texture model
 * @param       useKnownTriangles           Input - use known triangle structures??
 * @note        Refer to "Statistical Models of Appearance for Computer Vision" page 31, Cootes
 * @return      void
*/
void VO_FeatureModel::VO_BuildFeatureModel( const vector<string>& allLandmarkFiles4Training,
                                            const vector<string>& allImgFiles4Training,
                                            const string& shapeinfoFileName, 
                                            unsigned int database,
                                            unsigned int channels,
                                            int trm, 
                                            float TPShape, 
                                            float TPTexture, 
                                            bool useKnownTriangles)
{
    
}
