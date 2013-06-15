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

#ifndef __VO_ASMLTCS_H__
#define __VO_ASMLTCS_H__


#include <vector>

#include "opencv/cv.h"
#include "opencv/highgui.h"

#include "VO_DiscreteWavelet.h"
#include "VO_AXM.h"
#include "VO_Features.h"

using namespace std;
using namespace cv;


/**
* @author        JIA Pei
* @brief        Local texture constrained ASM.
 * @note        Current VO_ASMLTCs only deal with 1 channel gray image
 */
class VO_ASMLTCs : public VO_AXM
{
friend class VO_Fitting2DSM;
friend class VO_FittingAAMBasic;
friend class VO_FittingAAMForwardIA;
friend class VO_FittingAAMInverseIA;
friend class VO_FittingASMLTCs;
friend class VO_FittingASMNDProfiles;
friend class VO_FittingAFM;
protected:
    /** */
    VO_Features*                    m_pVOfeatures;

    /** Local texture constrained methods */
    unsigned int                    m_iLTCMethod;

    /** length of LTC vector in format of rr...rrgg...ggbb...bb . */
    unsigned int                    m_iNbOfLTC4PerPoint;

    /** All loaded LTCs in a vector format. m_iNbOfSamples*m_iNbOfPyramidLevels*(m_iNbOfPoints*m_iNbOfLTC4PerPoint) */
    vector< vector< Mat_<float> > > m_vvLTCs;

    /** All normalized LTCs in a vector format. m_iNbOfSamples*m_iNbOfPyramidLevels*(m_iNbOfPoints*m_iNbOfLTC4PerPoint) */
    vector< vector< Mat_<float> > > m_vvNormalizedLTCs;
    
    /** mean LTC for all concerned points. m_iNbOfPyramidLevels*m_iNbOfPoints*(1*m_iNbOfLTC4PerPoint) */
    vector< vector< Mat_<float> > > m_vvLTCMeans;
    
    /** LTC covariance for all concerned points. m_iNbOfPyramidLevels*m_iNbOfPoints*(m_iNbOfLTC4PerPoint*m_iNbOfLTC4PerPoint) */
    vector< vector< Mat_<float> > > m_vvCVMInverseOfLTCCov;

    /** image size for wavelet transform */
    Size                            m_localImageSize;

    /** Initialization */
    void                            init()
    {
        this->m_iMethod                 = VO_AXM::ASM_LTC;
        this->m_iLTCMethod              = VO_Features::DIRECT;
        this->m_iNbOfLTC4PerPoint       = 0;
        this->m_pVOfeatures             = NULL;
    }

public:
    /** Default constructor to create a VO_ASMLTCs object */
    VO_ASMLTCs() {this->init();}

    /** Destructor */
    ~VO_ASMLTCs() {}

    /** Build ASMLTCs model */
    void                            VO_BuildASMLTCs (   const vector<string>& allLandmarkFiles4Training,
                                                        const vector<string>& allImgFiles4Training,
                                                        const string& shapeinfoFileName,
                                                        unsigned int database,
                                                        unsigned int channels = 1,
                                                        unsigned int levels = 4,
                                                        int trm = VO_Features::DIRECT,
                                                        float TPShape = 0.95f,
                                                        bool useKnownTriangles = false,
                                                        unsigned int ltcMtd = VO_Features::DIRECT,
                                                        Size imgSize = Size(16, 16) );

    /** wavelet localization */
    static Rect                     VO_CalcImagePatchRect(const Mat& iImg, const Point2f& pt, Size imgSize);

    /** Load Training data for LTC model */
    void                            VO_LoadLTCTrainingData();

    /** load LTC around 1 single point for 1 single channel */
    static void                     VO_LoadLTC4OneAnnotatedPoint(   const Mat& iImg,
                                                                    const VO_Shape& theShape,
                                                                    unsigned int ptIdx,
                                                                    Size imgSize,
                                                                    VO_Features* vofeatures,
                                                                    int shiftX = 0,
                                                                    int shiftY = 0);

    /** calculate statistics */
    void                            VO_CalcStatistics4AllLTCs();

    /** Output one wavelet transformed image, for the purpose of demonstration */
    void                            VO_HardSaveWaveletSingleChannelImage(   const string& fn, 
                                                                            Size imgSize, 
                                                                            unsigned int displayMtd = STRETCH);

    /** Save ASM LTCs, to a specified folder */
    void                            VO_Save(const string& fd);

    /** Load all parameters */
    void                            VO_Load(const string& fd);

    /** Load parameters for fitting */
    void                            VO_LoadParameters4Fitting(const string& fd);
};

#endif  // __VO_ASMLTCS_H__

