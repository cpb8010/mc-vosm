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


#ifndef __VO_RECOGNITIONALGS_H__
#define __VO_RECOGNITIONALGS_H__

#include <cstring>
#include "opencv/cv.h"
#include "opencv/cvaux.h"
#include "opencv/highgui.h"
#include "VO_CVCommon.h"
#include "VO_AdditiveStrongerClassifier.h"
#include "VO_FaceDetectionAlgs.h"
#include "VO_TrackingAlgs.h"
#include "VO_FaceParts.h"
#include "VO_Shape2DInfo.h"
#include "VO_Shape.h"

#define FOCAL_LENGTH 10000000000

using namespace std;
using namespace cv;


/** 
* @author  JIA Pei
* @brief   Recognition algorithms.
*/
class CRecognitionAlgs
{
protected:
    void            init() {}
public:
    /** Constructor */
    CRecognitionAlgs() {this->init();}

    /** Destructor */
    ~CRecognitionAlgs();

    /** Global Texture Constraint using Probability image, 
    basically an evaluation method for current fitting effect */
    static bool     EvaluateFaceTrackedByProbabilityImage(
                        CTrackingAlgs* trackalg,
                        const Mat& iImg,
                        const VO_Shape& iShape,
                        Size smallSize
                        = Size(FACESMALLESTSIZE, FACESMALLESTSIZE),
                        Size bigSize
                        = Size(FACEBIGGESTSIZE, FACEBIGGESTSIZE) );

    /** An evaluation method for current fitting effect, 
    based on face components detection */
    static bool     EvaluateFaceTrackedByCascadeDetection(
                        const CFaceDetectionAlgs* fd,
                        const Mat& iImg,
                        const VO_Shape& iShape,
                        const vector<VO_Shape2DInfo>& iShapeInfo, 
                        const VO_FaceParts& iFaceParts);

    /** Calculate shape distance */
    static float    ShapeDistance(  const VO_Shape& shape1,
                                    const VO_Shape& shape2);

    /** Calculate fitting effect for static images */
    static bool     CalcFittingEffect4StaticImage(
                        const Mat_<float>& avgSParam,
                        const Mat_<float>& icovSParam,
                        const Mat_<float>& avgTParam,
                        const Mat_<float>& icovTParam,
                        const Mat_<float>& iSParams,
                        const Mat_<float>& iTParams,
                        const Scalar& ShapeDistMean,
                        const Scalar& ShapeDistStddev,
                        const Scalar& TextureDistMean,
                        const Scalar& TextureDistStddev,
                        float& sDist,
                        float& tDist,
                        bool WeakFitting = true );

    /** Calculate fitting effect for dynamic images sequence */
    static bool     CalcFittingEffect4ImageSequence(
                        const Mat_<float>& avgSParam,
                        const Mat_<float>& icovSParam,
                        const Mat_<float>& avgTParam,
                        const Mat_<float>& icovTParam,
                        const Mat_<float>& iSParams,
                        const Mat_<float>& iTParams,
                        const Scalar& ShapeDistMean,
                        const Scalar& ShapeDistStddev,
                        const Scalar& TextureDistMean,
                        const Scalar& TextureDistStddev,
                        bool WeakFitting = true );

    /** Calculate face fitting effect */
	static void		CalcShapeFittingEffect(	const VO_Shape& refShape,
											const VO_Shape& fittedShape,
											float& deviation,
											vector<float>& ptErrorFreq,
											int nb = 20,
											vector<float>* ptDists = 0);

	/** Save shape recognition results */
    static void     			SaveShapeRecogResults(	const string& fd,
														const string& fnIdx,
														float deviation,
														vector<float>& ptErrorFreq);
	/** Save shape data results */
    static void     			SaveShapeResults(		const string& fd,
														const string& fnIdx,
														float deviation,
														vector<float>& ptErrorFreq,
														vector<float>& ptErrorPerPoint,
														const VO_Shape& fittedShape);
	/** Save a myriad of results from the fitting process*/
	static void CRecognitionAlgs::SaveFittingResults(		const string& fd,
												const string& fnIdx,
												float deviation,
												vector<float>& ptDists,
												vector<float>& ptErrorFreq,
												const VO_Shape& fittedShape,
												cv::Point2f* gt_canidatePoints,
												cv::Point2f* t_canidatePoints,
												float fitTime);

    /** Calculate face absolute orientations */
    static vector<float>    CalcAbsoluteOrientations(
                                const VO_Shape& iShape2D,
                                const VO_Shape& iShape3D,
                                VO_Shape& oShape2D);

    /** Recognize face roll angle */
    static float    CalcFaceRoll(const vector<float>& iLine);

    /** Recognize face yaw angle */
    static float    CalcFaceYaw(const vector<float>& iLine,
                                const VO_Shape& iShape,
                                const VO_FaceParts& iFaceParts);

    /** Recognize face pitch angle */
    static float    CalcFacePitch(  const VO_Shape& iShape,
                                    const VO_FaceParts& iFaceParts);

    /** Recognize face angles using 2D model */
    static void     CalcFittedFaceAngle2D(  vector<float>& angles,
                                            const VO_Shape& iShape,
                                            const VO_FaceParts& iFaceParts);

    /** Recognize face angles using 3D model */
    static void     CalcFittedFaceAngle3D(  vector<float>& angles,
                                            const VO_Shape& iShape,
                                            const VO_FaceParts& iFaceParts);

};

#endif    // __VO_RECOGNITIONALGS_H__
