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

#include <sstream>
#include <fstream>


#include "VO_CVCommon.h"
#include "VO_AXM.h"
#include "VO_Fitting2DSM.h"
#include "VO_FittingAAMBasic.h"
#include "VO_FittingAAMForwardIA.h"
#include "VO_FittingAAMInverseIA.h"
#include "VO_FittingAFM.h"
#include "VO_FittingASMLTCs.h"
#include "VO_FittingASMNDProfiles.h"
#include "VO_FaceKeyPoint.h"


float VO_Fitting2DSM::pClose    = 0.90f;

/** Initialization */
void VO_Fitting2DSM::init()
{
    this->m_iNbOfPyramidLevels              = 3;
    this->m_iFittingMethod                  = VO_AXM::AFM;
    this->m_iIteration                      = 0;
    this->m_fFittingTime                    = 0.0f;
    this->m_fScale                          = 1.0f;
    this->m_vRotateAngles.resize(1);
    this->m_MatCenterOfGravity              = Mat_<float>::zeros(2, 1);
    this->m_vTriangle2D.clear();
    this->m_vShape2DInfo.clear();
    this->m_FaceParts.clear();
    this->m_vPointWarpInfo.clear();
}


/** Constructor */
VO_Fitting2DSM::VO_Fitting2DSM()
{
    this->init();
}


/** Destructor */
VO_Fitting2DSM::~VO_Fitting2DSM()
{
    this->m_vTriangle2D.clear();
    this->m_vShape2DInfo.clear();
    this->m_FaceParts.clear();
    this->m_vPointWarpInfo.clear();
}


/**
 *@author     JIA Pei
 *
 *@brief      Obtain the first shape instance
 *
 *@methods  - 1) three points' pairs are used to calculate affine transform
 *          - 2) use my own align transform - COG -> rotation -> scaling...
 *          - 3) SVD:  Y = AX
 *                x'     a00 a01 a02     x
 *               (y') = (a10 a11 a12) * (y) ,
 *                1      a20 a21 a22     1
 *               where  a00 = cos(theta), a01 = -sin(theta)
 *                      a10 = sin(theta), a11 = cons(theta)
 *                      a02 = tx, a12 = ty
 *                      a20 = a21 = 0; a22 = 1
 *            However, the above values are not guaranteed during calculation
**/
Mat_<float> VO_Fitting2DSM::VO_FirstEstimationBySingleWarp(const VO_FaceParts& iFaceParts,
                                                    const VO_Shape& iShape,
                                                    const Point2f& ptLeftEyeCenter,
                                                    const Point2f& ptRightEyeCenter,
                                                    const Point2f& ptMouthCenter )
{
    unsigned int NbOfKeyPoints     = 3;
    Mat fpts(1, NbOfKeyPoints, CV_32FC2);
    Mat tpts(1, NbOfKeyPoints, CV_32FC2);
        
    Point2f pt;
    VO_KeyPoint::CalcFaceKeyPoint(pt,  iShape, iFaceParts, VO_KeyPoint::LEFTEYECENTER);
    fpts.at<Vec2f>(0,0) = pt;
    VO_KeyPoint::CalcFaceKeyPoint(pt,  iShape, iFaceParts, VO_KeyPoint::RIGHTEYECENTER);
    fpts.at<Vec2f>(0,1) = pt;
    VO_KeyPoint::CalcFaceKeyPoint(pt,  iShape, iFaceParts, VO_KeyPoint::MOUTHCENTER);
    fpts.at<Vec2f>(0,2) = pt;

    tpts.at<Vec2f>(0,0) = ptLeftEyeCenter;
    tpts.at<Vec2f>(0,1) = ptRightEyeCenter;
    tpts.at<Vec2f>(0,2) = ptMouthCenter;
    
    // Explained by JIA Pei. For only 3 points, the affine transform can be computed by "getAffineTransform"
//    Mat_<float> matWarping = cv::getAffineTransform( iShapeKeyPoints, detectKeyPoints );
    
    // For more than 3 points, we need "estimateRigidTransform"
    Mat_<float> matWarping = cv::estimateRigidTransform( fpts, tpts, true );
    
    return matWarping;
}


VO_Shape VO_Fitting2DSM::VO_FirstEstimationByScaling(   const VO_Shape& iShape,
                                                        const Rect& rect )
{
    VO_Shape res = iShape;
    Rect_<float> rect0 = iShape.GetShapeRect();
    float fScaleX = static_cast<float>(rect.width)/rect0.width *0.80f;
    float fScaleY = static_cast<float>(rect.height)/rect0.height *0.80f;
    res.ScaleX(fScaleX);
    res.ScaleY(fScaleY);
    rect0 = iShape.GetShapeBoundRect();
    Mat_<float> translation = Mat_<float>::zeros(2, 1);
    float centerX = (float)rect.x + (float)rect.width/2.0f;
    float centerY = (float)rect.y + (float)rect.height/2.0f;
    float center0X = (float)rect0.x + (float)rect0.width/2.0f;
    float center0Y = (float)rect0.x + (float)rect0.height/2.0f;
    translation(0,0) = centerX - center0X;
    translation(1,0) = centerY - center0Y;
    res.Translate( translation );
    return res;
}


/**
 * @author         JIA Pei
 * @version        2010-05-15
 * @brief          Calculate error image by
 * @param        
*/
float VO_Fitting2DSM::VO_CalcErrorImage(const Mat& iImg,
                                        const VO_Shape& iShape,
                                        const VO_Texture& iTexture,
                                        VO_Texture& textureDiff)
{
    float E = FLT_MAX;

    // extract the real texture on the image from the model shape  calculated above
    if ( VO_TextureModel::VO_LoadOneTextureFromShape(iShape, iImg, this->m_vTriangle2D, this->m_vPointWarpInfo, this->m_VOFittingTexture ) )
    {
        this->m_VOFittingTexture.Normalize();
        textureDiff  = this->m_VOFittingTexture - iTexture;
        E = textureDiff.GetTextureNorm();
    }

    return E;
}


/**
 * @author      JIA Pei
 * @version     2010-05-14
 * @brief       Start object tracking
 * @param       iImage              Input -- The current image frame for tracking
 * @param       oImages             Output -- output images - for converging iterations
 * @param       fittingMethod       Input -- fitting method
 * @param       ptLeftEyeCenter     Input -- the detected left eye center
 * @param       ptRightEyeCenter    Input -- the detected right eye center
 * @param       ptMouthCenter       Input -- the detected mouth center
 * @param       epoch               Input -- the iteration epoch
 * @param       pyramidlevel        Input -- pyramid levels, normally 3
*/
float VO_Fitting2DSM::VO_StartFitting(  const Mat& iImage,
										vector<DrawMeshInfo>& oMeshInfo,
                                        const VO_AXM::TYPE fittingMethod,
                                        const Point2f& ptLeftEyeCenter,
                                        const Point2f& ptRightEyeCenter,
                                        const Point2f& ptMouthCenter,
                                        const unsigned int epoch,
                                        const unsigned int pyramidlevel,
										const bool record,
										const vector<VO_Fitting2DSM::MULTICHANNELTECH>& fittingTechs,
										const bool doAffineWarp)
{
    this->m_fFittingTime = 0.0f;
    this->SetInputImage(iImage);
	if(doAffineWarp){
		// The following function is important!!
		// Although it will be done only once during the whole fitting process, it will roughly give out the rough warping (global shape normalization )relationship between m_MatAlignedShapeInstance and m_MatShapeInstance
		// Once this->m_fScale, this->m_vRotateAngles, this->m_MatCenterOfGravity are calculated, they will not change during the whole fitting process
		this->m_VOFittingShape.clone(this->m_VOTemplateAlignedShape);
		this->m_VOFittingShape.Affine2D(    VO_Fitting2DSM::VO_FirstEstimationBySingleWarp(
											this->m_FaceParts,
											this->m_VOFittingShape,
											ptLeftEyeCenter,
											ptRightEyeCenter,
											ptMouthCenter)
										);
		//Rect rect = fd.GetDetectedFaceWindow2SM();    // Explained by JIA Pei, it seems GetDetectedFaceWindow2SM doesn't work fine here
		//Rect rect = fd.GetDetectedFaceWindow();
		//this->m_VOFittingShape = VO_Fitting2DSM::VO_FirstEstimationByScaling(this->m_VOTemplateAlignedShape, fd);
		this->m_VOFittingShape.ConstrainShapeInImage(this->m_ImageInput);

		//Mat tempImg = this->m_ImageInput.clone();
		//for(unsigned int i = 0; i < this->m_VOFittingShape.GetNbOfPoints(); i++)
		//{
		//    Point2f pt = this->m_VOFittingShape.GetA2DPoint(i);
		//    VO_Fitting2DSM::VO_DrawAPoint(pt, tempImg);
		//}
		//imwrite("temp.jpg", tempImg);
	}

    this->m_iFittingMethod = fittingMethod;
    this->m_iNbOfPyramidLevels = pyramidlevel;
    switch(this->m_iFittingMethod)
    {
    case VO_AXM::AAM_BASIC:
        {
			this->m_fFittingTime = dynamic_cast<VO_FittingAAMBasic*>(this)->VO_BasicAAMFitting(iImage, oMeshInfo, epoch, record);
        }
        break;
    case VO_AXM::AAM_DIRECT:
        {
			this->m_fFittingTime = dynamic_cast<VO_FittingAAMBasic*>(this)->VO_DirectAAMFitting(iImage, oMeshInfo, epoch, record);
        }
        break;
    case VO_AXM::CLM:
        {
//            this->m_fFittingTime = 
        }
        break;
    case VO_AXM::AFM:
        {
			this->m_fFittingTime = dynamic_cast<VO_FittingAFM*>(this)->VO_AFMFitting(iImage, oMeshInfo, this->m_iFittingMethod, epoch, record);
        }
        break;
    case VO_AXM::AAM_IAIA:
        {
			this->m_fFittingTime = dynamic_cast<VO_FittingAAMInverseIA*>(this)->VO_IAIAAAMFitting(iImage, oMeshInfo, epoch, record);
        }
        break;
    case VO_AXM::AAM_CMUICIA:
        {
			this->m_fFittingTime = dynamic_cast<VO_FittingAAMInverseIA*>(this)->VO_ICIAAAMFitting(iImage, oMeshInfo, epoch, record);
        }
        break;
    case VO_AXM::AAM_FAIA:
        {
			this->m_fFittingTime = dynamic_cast<VO_FittingAAMForwardIA*>(this)->VO_FAIAAAMFitting(iImage, oMeshInfo, epoch, record);
        }
        break;
    case VO_AXM::ASM_LTC:
        {
			this->m_fFittingTime = dynamic_cast<VO_FittingASMLTCs*>(this)->VO_ASMLTCFitting(iImage, oMeshInfo, VO_Features::DIRECT, epoch, this->m_iNbOfPyramidLevels, record);
        }
        break;
    case VO_AXM::ASM_PROFILEND:
        {
            this->m_fFittingTime = dynamic_cast<VO_FittingASMNDProfiles*>(this)->VO_ASMNDProfileFitting(iImage, oMeshInfo, epoch, this->m_iNbOfPyramidLevels, record, fittingTechs);
        }
        break;
    default:
        {
            this->m_fFittingTime = dynamic_cast<VO_FittingASMNDProfiles*>(this)->VO_ASMNDProfileFitting(iImage, oMeshInfo, epoch, this->m_iNbOfPyramidLevels, record, fittingTechs);
        }
        break;
    }
	if (oMeshInfo.size() > 0 ) VO_DrawMeshOnImage(oMeshInfo.back(),this->m_ImageInput,this->m_ImageOutput);
    else this->m_ImageOutput = this->m_ImageInput;

// vector<float> angles;
// VO_AAMFaceRecognition::CalcFittedFaceAngle2D(angles, this->m_VOShape, this->m_FaceParts);
// imwrite("test.jpg", this->m_ImageOutput);

//    return this->m_ImageOutput;
    return this->m_fFittingTime;
}

/**
 * @author      JIA Pei
 * @version     2010-05-07
 * @brief       draw a point on the image
 * @param       iShape          Input -- the input shape
 * @param       iAAMModel       Input -- the model
 * @param       ioImg           Input and Output -- the image
 * @return      void
 */
void VO_Fitting2DSM::VO_DrawMesh(const VO_Shape& iShape, const VO_AXM* iModel, Mat& ioImg)
{
    Point iorg,idst;
    vector<VO_Edge> edges = iModel->GetEdge();
    unsigned int NbOfEdges = iModel->GetNbOfEdges();

    for (unsigned int i = 0; i < NbOfEdges; i++)
    {
        iorg = cvPointFrom32f( iShape.GetA2DPoint( edges[i].GetIndex1() ) );
        idst = cvPointFrom32f( iShape.GetA2DPoint( edges[i].GetIndex2() ) );
        // Edge
        cv::line( ioImg, iorg, idst, colors[8], 1, 0, 0 );
        // Key points
        cv::circle( ioImg, iorg, 2, colors[0], -1, 8, 0 );
        cv::circle( ioImg, idst, 2, colors[0], -1, 8, 0 );
    }
}

void VO_Fitting2DSM::VO_DrawMeshOnImage(DrawMeshInfo meshStruct,const Mat& iImage, Mat& oImage){
	//i hope this dosen't leak memory.
	oImage = Mat(iImage.size(), iImage.type(), iImage.channels());
	Mat oImageROI = oImage(Range (0, (int)(oImage.rows/meshStruct.pyrScale) ), Range (0, (int)(oImage.cols/meshStruct.pyrScale) ) );
	cv::resize(iImage, oImageROI, oImageROI.size() );
	VO_Fitting2DSM::VO_DrawMesh(meshStruct.drawPts / meshStruct.f2Scale, meshStruct.modelPtr, oImage);

}

/**
 * @author      JIA Pei
 * @version     2010-05-07
 * @brief       draw a point on the image
 * @param       iShape          Input -- the input shape
 * @param       theSubshape     Output -- and input, the image drawn with the point
 * @param       iLine           Input -- the line
 * @param       oImg            Output--  output image
 * @param       dir             Input -- direction
 * @param       ws              Input -- 
 * @param       offset          Input -- add some offset at both ends of the line segment itself
 * @param       ci              Input -- color index
 * @return      void
 */
void VO_Fitting2DSM::VO_DrawAline(  const VO_Shape& iShape,
                                    const VO_Shape& theSubshape,
                                    const vector<float>& iLine,
                                    Mat& oImg,
                                    unsigned int dir,
                                    bool ws,
                                    unsigned int offset,
                                    unsigned int ci)
{
    switch(dir)
    {
    case VERTICAL:
        {
            float A = iLine[0];
            float B = iLine[1];
            float C = iLine[2];
            Point2f ptf1, ptf2;
            if(ws)
            {
                ptf1.y = iShape.MinY() - offset;
                ptf2.y = iShape.MaxY() + offset;
            }
            else
            {
                ptf1.y = theSubshape.MinY() - offset;
                ptf2.y = theSubshape.MaxY() + offset;
            }
            ptf1.x = -(C + B*ptf1.y)/A;
            ptf2.x = -(C + B*ptf2.y)/A;
            Point pt1 = cvPointFrom32f( ptf1 );
            Point pt2 = cvPointFrom32f( ptf2 );
            cv::line( oImg, pt1, pt2, colors[ci], 2, 0, 0);
        }
        break;
    case HORIZONTAL:
    default:
        {
            float A = iLine[0];
            float B = iLine[1];
            float C = iLine[2];
            Point2f ptf1, ptf2;
            if(ws)
            {
                ptf1.x = iShape.MinX() - offset;
                ptf2.x = iShape.MaxX() + offset;
            }
            else
            {
                ptf1.x = theSubshape.MinX() - offset;
                ptf2.x = theSubshape.MaxX() + offset;
            }
            ptf1.y = -(C + A*ptf1.x)/B;
            ptf2.y = -(C + A*ptf2.x)/B;
            Point pt1 = cvPointFrom32f( ptf1 );
            Point pt2 = cvPointFrom32f( ptf2 );
            cv::line( oImg, pt1, pt2, colors[ci], 2, 0, 0);
        }
        break;
    }
}


/**
 * @author      JIA Pei
 * @version     2010-05-07
 * @brief       draw a point on the image
 * @param       pt      Input -- the point
 * @param       oImg    Input and Output -- the image drawn with the point
 * @return      void
 */
void VO_Fitting2DSM::VO_DrawAPoint(const Point2f& pt, Mat& ioImg)
{
    Point tempPt = cvPointFrom32f(pt);
    cv::circle( ioImg, tempPt, 2, colors[5], -1, 8, 0 );
}

