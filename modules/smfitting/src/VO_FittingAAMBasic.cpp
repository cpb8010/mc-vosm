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
#include <string>

#include "VO_FittingAAMBasic.h"


/** For damp, damp coefficients */
vector<float> VO_FittingAAMBasic::k_values;


/** Default Constructor */
VO_FittingAAMBasic::VO_FittingAAMBasic()
{
    this->init();
}

/** Destructor */
VO_FittingAAMBasic::~VO_FittingAAMBasic()
{
    if(this->m_VOAAMBasic)      delete this->m_VOAAMBasic; this->m_VOAAMBasic = NULL;
 }
 
/** Initialization */
void VO_FittingAAMBasic::init()
{
    VO_Fitting2DSM::init();
    this->m_VOAAMBasic          = new VO_AAMBasic();
    this->m_E                   = 0.0f;
    this->m_E_previous          = 0.0f;
    VO_FittingAAMBasic::k_values.push_back(1.0f);
    VO_FittingAAMBasic::k_values.push_back(0.5f);
    VO_FittingAAMBasic::k_values.push_back(0.25f);
    VO_FittingAAMBasic::k_values.push_back(0.125f);
    VO_FittingAAMBasic::k_values.push_back(0.0625f);
    VO_FittingAAMBasic::k_values.push_back(0.0f);
}


/**
 * @author      JIA Pei
 * @version     2010-05-14
 * @brief       obtain the reconstructed real-size shape using c parameters and t parameters
 * @param       c                       Input -- PCA appearance parameters, including both shape non-rigid variance + texture variance
 * @param       t                       Input -- parameters required for global shape normalization, rigid variance, refer to "AAM Revisited equation (42)"
 * @param       modelAlignedShape       Output -- the modeled aligned shape
 * @param       modelNormalizedTexture  Output -- the modeled normalized texture
 * @param       oShape                  Output -- output shape after global shape normalization. Please make sure global shape normalization is not Procrustes Analysis.
 * @param       scale                   Input and Output -- scaling
 * @param       rotateAngles            Input and Output -- rotation
 * @param       matCOG                  Input and Output -- translation
 */
void VO_FittingAAMBasic::VO_CParamTParam2FittingShape(  const Mat_<float>& c,
                                                        const Mat_<float>& t,
                                                        VO_Texture& modelNormalizedTexture,
                                                        VO_Shape& oShape,
                                                        float& scale,
                                                        vector<float>& rotateAngles,
                                                        Mat_<float>& matCOG,
                                                        unsigned int mtd)
{
   // generate shape and texture from C parameters
    this->m_VOAAMBasic->VO_CParamBackProjectToSParamTParam(c, this->m_MatModelAlignedShapeParam, this->m_MatModelNormalizedTextureParam);
    this->m_VOAAMBasic->VO_SParamBackProjectToAlignedShape(this->m_MatModelAlignedShapeParam, this->m_VOModelAlignedShape);
    this->m_VOAAMBasic->VO_TParamBackProjectToNormalizedTexture(this->m_MatModelNormalizedTextureParam,  modelNormalizedTexture, this->m_VOAAMBasic->GetNbOfChannels() );
    //this->VO_CParamBackProjectToAlignedShape(c, this->m_MatModeledAlignedShapeInstance);
    //this->VO_CParamBackProjectToNormalizedTexture(c, this->m_MatModeledNormalizedTextureInstance);

    switch(mtd)
    {
    case VO_Fitting2DSM::USEGLOBALSHAPENORMALIZATION:
        {

        }
        break;
    case VO_Fitting2DSM::USESIMILARITYTRANSFORM:
    default:
        {
            float updatescale = 1.0;
            vector<float> updateangles(1);
            Mat_<float> updatetranslation = Mat_<float>::zeros(2, 1);
            VO_Shape::GlobalShapeNormalization2SimilarityTrans(t, updatescale, updateangles, updatetranslation );
            scale *= updatescale;
            rotateAngles[0] = -rotateAngles[0]+updateangles[0];
            matCOG(0,0) += updatetranslation(0,0);
            matCOG(1,0) += updatetranslation(1,0);

            // shape adjustment by t parameters - pose transform
            VO_Shape::GlobalShapeNormalization2D(this->m_VOModelAlignedShape, oShape, scale, rotateAngles, matCOG);
        }
        break;
    }
}


/**
 * @author         JIA Pei
 * @version        2010-05-18
 * @brief          Load all AAM data from a specified folder for later fitting, to member variable m_VOAAMBasic
 * @param          fd         Input - the folder that AAM to be loaded from
*/
void VO_FittingAAMBasic::VO_LoadParameters4Fitting(const string& fd)
{
    this->m_VOAAMBasic->VO_LoadParameters4Fitting(fd);
    
    // VO_Fitting2DSM
    this->m_VOTemplateAlignedShape          = this->m_VOAAMBasic->m_VOAlignedMeanShape;
    this->m_VOTemplateNormalizedTexture     = this->m_VOAAMBasic->m_VONormalizedMeanTexture;
    this->m_vTriangle2D                     = this->m_VOAAMBasic->m_vNormalizedTriangle2D;
    this->m_vShape2DInfo                    = this->m_VOAAMBasic->m_vShape2DInfo;
    this->m_FaceParts                       = this->m_VOAAMBasic->m_FaceParts;
    this->m_vPointWarpInfo                  = this->m_VOAAMBasic->m_vNormalizedPointWarpInfo;

    // VO_FittingAAMBasic
    this->m_MatDeltaC                       = Mat_<float>::zeros(1, this->m_VOAAMBasic->m_iNbOfAppearanceEigens);
    this->m_MatEstimatedC                   = Mat_<float>::zeros(1, this->m_VOAAMBasic->m_iNbOfAppearanceEigens);
    this->m_MatCurrentC                     = Mat_<float>::zeros(1, this->m_VOAAMBasic->m_iNbOfAppearanceEigens);
    this->m_MatDeltaT                       = Mat_<float>::zeros(1, 4);
    this->m_MatEstimatedT                   = Mat_<float>::zeros(1, 4);
    this->m_MatCurrentT                     = Mat_<float>::zeros(1, 4);
}


/**
 * @author         JIA Pei
 * @version        2010-05-20
 * @brief          Basic AAM Fitting, for static images, so that we record the whole fitting process
 * @param          iImg            Input - image to be fitted
 * @param          oImages         Output - the fitted shape
 * @param          epoch           Input - the iteration epoch
*/
float VO_FittingAAMBasic::VO_BasicAAMFitting(    const Mat& iImg,
												vector<DrawMeshInfo>& oMeshInfo,
                                                unsigned int epoch,
                                                bool record)
{
    double t = (double)cvGetTickCount();

    this->SetProcessingImage(iImg, this->m_VOAAMBasic);
    this->m_iIteration = 0;

    if(record)
    {
		oMeshInfo.push_back(DrawMeshInfo(1,1,this->m_VOFittingShape,this->m_VOAAMBasic));
    }

    // Get m_MatModelAlignedShapeParam and m_fScale, m_vRotateAngles, m_MatCenterOfGravity
    this->m_VOAAMBasic->VO_CalcAllParams4AnyShapeWithConstrain( this->m_VOFittingShape,
                                                                this->m_MatModelAlignedShapeParam,
                                                                this->m_fScale,
                                                                this->m_vRotateAngles,
                                                                this->m_MatCenterOfGravity);
    this->m_VOFittingShape.ConstrainShapeInImage(this->m_ImageProcessing);
    if(record)
    {
		oMeshInfo.push_back(DrawMeshInfo(1,1,this->m_VOFittingShape,this->m_VOAAMBasic));
    }
    
    // Get m_MatModelNormalizedTextureParam
    VO_TextureModel::VO_LoadOneTextureFromShape(this->m_VOFittingShape,
                                                this->m_ImageProcessing,
                                                this->m_vTriangle2D,
                                                this->m_vPointWarpInfo,
                                                this->m_VOFittingTexture );
    // estimate the texture model parameters
    this->m_VOAAMBasic->VO_CalcAllParams4AnyTexture(this->m_VOFittingTexture,
                                                    this->m_MatModelNormalizedTextureParam);

    // Calculate m_MatCurrentC
    this->m_VOAAMBasic->VO_SParamTParamProjectToCParam( this->m_MatModelAlignedShapeParam,
                                                        this->m_MatModelNormalizedTextureParam,
                                                        this->m_MatCurrentC );
    // Set m_MatCurrentT, m_MatDeltaT, m_MatEstimatedT, m_MatDeltaC, m_MatEstimatedC, etc.
    this->m_MatCurrentT     = Mat_<float>::zeros(this->m_MatCurrentT.size());
    this->m_MatDeltaT       = Mat_<float>::zeros(this->m_MatDeltaT.size());
    this->m_MatEstimatedT   = Mat_<float>::zeros(this->m_MatEstimatedT.size());
    this->m_MatDeltaC       = Mat_<float>::zeros(this->m_MatDeltaC.size());
    this->m_MatEstimatedC   = Mat_<float>::zeros(this->m_MatEstimatedC.size());
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // explained by JIA Pei. 2010-05-20
    // For the first round, this->m_VOFittingShape should not change after calling "VO_CParamTParam2FittingShape"
    // But this is not the case. why?
    // Before calling VO_CParamTParam2FittingShape, this->m_VOFittingShape is calculated by 
    // a) assigning m_VOTemplateAlignedShape
    // b) align to the real-size face using detected eyes and mouth
    // c) constrain the shape within the image
    // d) constrain the shape parameters and calculate those rigid transform parameters
    // cout << this->m_VOFittingShape << endl;
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // Estimate m_VOFittingShape and m_VOFittingTexture
    this->VO_CParamTParam2FittingShape( this->m_MatCurrentC,
                                        this->m_MatCurrentT,
                                        this->m_VOModelNormalizedTexture,
                                        this->m_VOFittingShape,
                                        this->m_fScale,
                                        this->m_vRotateAngles,
                                        this->m_MatCenterOfGravity );
    this->m_VOFittingShape.ConstrainShapeInImage(this->m_ImageProcessing);        // Remember to call ConstrainShapeInImage() whenever you update m_VOFittingShape
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // When calling VO_CParamTParam2FittingShape, this->m_VOFittingShape is calculated by
    // a) c parameters to reconstruct shape parameters
    // b) shape parameters to reconstruct shape
    // c) align to the real-size face by global shape normalization
    // cout << this->m_VOFittingShape << endl;
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    if(record)
    {
		oMeshInfo.push_back(DrawMeshInfo(1,1,this->m_VOFittingShape,this->m_VOAAMBasic));
    }
    
    this->m_E_previous = this->m_E = this->VO_CalcErrorImage(this->m_ImageProcessing,
                                                            this->m_VOFittingShape,
                                                            this->m_VOModelNormalizedTexture,
                                                            this->m_VOTextureError);

    do
    {
        ++this->m_iIteration;
        float estScale = this->m_fScale;
        vector<float> estRotateAngles = this->m_vRotateAngles;
        Mat_<float> estCOG = this->m_MatCenterOfGravity.clone();
        bool cBetter     = false;
        bool poseBetter = false;

        /**First shape parameters, c parameters. refer to equation (9.3)
        * Cootes "Statistical Model of Appearance for Computer Vision" */
        cv::gemm(this->m_VOTextureError.GetTheTextureInARow(), this->m_VOAAMBasic->m_MatRc, -1, Mat(), 0, this->m_MatDeltaC, GEMM_2_T);

        // damp -- C
        for(unsigned int i = 0; i < k_values.size(); i++)
        {
            // make damped c prediction
            cv::scaleAdd(this->m_MatDeltaC, k_values[i], this->m_MatCurrentC, this->m_MatEstimatedC);

            // make sure m_MatEstimatedC are constrained
            this->m_VOAAMBasic->VO_AppearanceParameterConstraint(this->m_MatEstimatedC);
            this->VO_CParamTParam2FittingShape( this->m_MatEstimatedC,
                                                this->m_MatCurrentT,
                                                this->m_VOModelNormalizedTexture,
                                                this->m_VOEstimatedShape,
                                                estScale,
                                                estRotateAngles,
                                                estCOG);
            if ( !VO_ShapeModel::VO_IsShapeInsideImage(this->m_VOEstimatedShape, this->m_ImageProcessing) )
                continue;
            else
                this->m_E = this->VO_CalcErrorImage(this->m_ImageProcessing,
                                                    this->m_VOEstimatedShape,
                                                    this->m_VOModelNormalizedTexture,
                                                    this->m_VOEstimatedTextureError);

            if (this->m_E < this->m_E_previous)
            {
                this->m_MatEstimatedC.copyTo(this->m_MatCurrentC);
                this->m_VOFittingShape.clone(this->m_VOEstimatedShape);
                this->m_VOTextureError.clone(this->m_VOEstimatedTextureError);
                this->m_E_previous = this->m_E;
                cBetter = true;
                this->m_fScale = estScale;
                this->m_vRotateAngles = estRotateAngles;
                this->m_MatCenterOfGravity = estCOG.clone();
                break;
            }
        }

        /** Second pose, t parameters. refer to equation (9.3)
        * Cootes "Statistical Model of Appearance for Computer Vision" */
        cv::gemm(this->m_VOTextureError.GetTheTextureInARow(), this->m_VOAAMBasic->m_MatRt, -1, Mat(), 0, this->m_MatDeltaT, GEMM_2_T);

        // damp -- T
        for(unsigned int i = 0; i < k_values.size(); i++)
        {
            // make damped c/pose prediction
            cv::scaleAdd(this->m_MatDeltaT, k_values[i], this->m_MatCurrentT, this->m_MatEstimatedT);
            this->VO_CParamTParam2FittingShape( this->m_MatCurrentC,
                                                this->m_MatEstimatedT,
                                                this->m_VOModelNormalizedTexture,
                                                this->m_VOEstimatedShape,
                                                estScale,
                                                estRotateAngles,
                                                estCOG);
            if ( !VO_ShapeModel::VO_IsShapeInsideImage(this->m_VOEstimatedShape, this->m_ImageProcessing) )
                continue;
            else
                this->m_E = this->VO_CalcErrorImage(this->m_ImageProcessing,
                                                    this->m_VOEstimatedShape,
                                                    this->m_VOModelNormalizedTexture,
                                                    this->m_VOEstimatedTextureError);

            if (this->m_E < this->m_E_previous)
            {
                // Since m_fScale, m_vRotateAngles and m_MatCenterOfGravity have been updated,
                // m_MatCurrentT should be assigned to 0 now!
                this->m_MatCurrentT = Mat_<float>::zeros(this->m_MatCurrentT.size());
//                this->m_MatEstimatedT.copyTo(this->m_MatCurrentT);
                this->m_VOFittingShape.clone(this->m_VOEstimatedShape);
                this->m_VOTextureError.clone(this->m_VOEstimatedTextureError);
                this->m_E_previous = this->m_E;
                poseBetter = true;
                this->m_fScale = estScale;
                this->m_vRotateAngles = estRotateAngles;
                this->m_MatCenterOfGravity = estCOG.clone();
                break;
            }
        }

        if( cBetter || poseBetter)
        {
            if(record)
            {
				oMeshInfo.push_back(DrawMeshInfo(1,1,this->m_VOFittingShape,this->m_VOAAMBasic));
            }
        }
        else
        break;

    }while( ( fabs(this->m_E) > FLT_EPSILON ) && (this->m_iIteration < epoch)/* && (cv::norm(this->m_MatDeltaC) > FLT_EPSILON) */ );
    
t = ((double)cvGetTickCount() -  t )/  (cvGetTickFrequency()*1000.);
cout << "Basic fitting time cost: " << t << " millisec" << endl;

    return safeDoubleToFloat(t);
}


/**
 * @author      JIA Pei
 * @version     2010-05-20
 * @brief       Basic AAM Fitting, for dynamic image sequence
 * @param       iImg            Input - image to be fitted
 * @param       ioShape         Input and Output - the fitted shape
 * @param       oImg            Output - the fitted image
 * @param       epoch           Input - the iteration epoch
*/
float VO_FittingAAMBasic::VO_BasicAAMFitting(const Mat& iImg,
                                            VO_Shape& ioShape,
                                            Mat& oImg,
                                            unsigned int epoch)
{
    this->m_VOFittingShape.clone(ioShape);
double t = (double)cvGetTickCount();

    this->SetProcessingImage(iImg, this->m_VOAAMBasic);
    this->m_iIteration = 0;

    // Get m_MatModelAlignedShapeParam and m_fScale, m_vRotateAngles, m_MatCenterOfGravity
    this->m_VOAAMBasic->VO_CalcAllParams4AnyShapeWithConstrain( this->m_VOFittingShape,
                                                                this->m_MatModelAlignedShapeParam,
                                                                this->m_fScale,
                                                                this->m_vRotateAngles,
                                                                this->m_MatCenterOfGravity);
    this->m_VOFittingShape.ConstrainShapeInImage(this->m_ImageProcessing);

    // Get m_MatModelNormalizedTextureParam
    VO_TextureModel::VO_LoadOneTextureFromShape(this->m_VOFittingShape,
                                                this->m_ImageProcessing,
                                                this->m_vTriangle2D,
                                                this->m_vPointWarpInfo,
                                                this->m_VOFittingTexture );
    // estimate the texture model parameters
    this->m_VOAAMBasic->VO_CalcAllParams4AnyTexture(this->m_VOFittingTexture, this->m_MatModelNormalizedTextureParam);

    // Calculate m_MatCurrentC
    this->m_VOAAMBasic->VO_SParamTParamProjectToCParam( this->m_MatModelAlignedShapeParam,
                                                        this->m_MatModelNormalizedTextureParam,
                                                        this->m_MatCurrentC );
    // Set m_MatCurrentT, m_MatDeltaT, m_MatEstimatedT, m_MatDeltaC, m_MatEstimatedC, etc.
    this->m_MatCurrentT     = Mat_<float>::zeros(this->m_MatCurrentT.size());
    this->m_MatDeltaT       = Mat_<float>::zeros(this->m_MatDeltaT.size());
    this->m_MatEstimatedT   = Mat_<float>::zeros(this->m_MatEstimatedT.size());
    this->m_MatDeltaC       = Mat_<float>::zeros(this->m_MatDeltaC.size());
    this->m_MatEstimatedC   = Mat_<float>::zeros(this->m_MatEstimatedC.size());
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // explained by JIA Pei. 2010-05-20
    // For the first round, this->m_VOFittingShape should not change after calling "VO_CParamTParam2FittingShape"
    // But this is not the case. why?
    // Before calling VO_CParamTParam2FittingShape, this->m_VOFittingShape is calculated by 
    // a) assigning m_VOTemplateAlignedShape
    // b) align to the real-size face using detected eyes and mouth
    // c) constrain the shape within the image
    // d) constrain the shape parameters and calculate those rigid transform parameters
    // cout << this->m_VOFittingShape << endl;
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // Estimate m_VOFittingShape and m_VOFittingTexture
    this->VO_CParamTParam2FittingShape( this->m_MatCurrentC,
                                        this->m_MatCurrentT,
                                        this->m_VOModelNormalizedTexture,
                                        this->m_VOFittingShape,
                                        this->m_fScale,
                                        this->m_vRotateAngles,
                                        this->m_MatCenterOfGravity );
    this->m_VOFittingShape.ConstrainShapeInImage(this->m_ImageProcessing);        // Remember to call ConstrainShapeInImage() whenever you update m_VOFittingShape
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // When calling VO_CParamTParam2FittingShape, this->m_VOFittingShape is calculated by
    // a) c parameters to reconstruct shape parameters
    // b) shape parameters to reconstruct shape
    // c) align to the real-size face by global shape normalization
    // cout << this->m_VOFittingShape << endl;
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    
    this->m_E_previous = this->m_E = this->VO_CalcErrorImage(this->m_ImageProcessing,
                                                            this->m_VOFittingShape,
                                                            this->m_VOModelNormalizedTexture,
                                                            this->m_VOTextureError);

    do
    {
        float estScale = this->m_fScale;
        vector<float> estRotateAngles = this->m_vRotateAngles;
        Mat_<float> estCOG  = this->m_MatCenterOfGravity.clone();
        bool cBetter        = false;
        bool poseBetter     = false;

        /**First shape parameters, c parameters. refer to equation (9.3)
        * Cootes "Statistical Model of Appearance for Computer Vision" */
        cv::gemm(this->m_VOTextureError.GetTheTextureInARow(), this->m_VOAAMBasic->m_MatRc, -1, Mat(), 0.0, this->m_MatDeltaC, GEMM_2_T);

        // damp -- C
        for(unsigned int i = 0; i < k_values.size(); i++)
        {
            // make damped c prediction
            cv::scaleAdd(this->m_MatDeltaC, k_values[i], this->m_MatCurrentC, this->m_MatEstimatedC);

            // make sure m_MatEstimatedC are constrained
            this->m_VOAAMBasic->VO_AppearanceParameterConstraint(this->m_MatEstimatedC);
            this->VO_CParamTParam2FittingShape( this->m_MatEstimatedC,
                                                this->m_MatCurrentT,
                                                this->m_VOModelNormalizedTexture,
                                                this->m_VOEstimatedShape,
                                                estScale,
                                                estRotateAngles,
                                                estCOG);
            if ( !VO_ShapeModel::VO_IsShapeInsideImage(this->m_VOEstimatedShape, this->m_ImageProcessing) )
                continue;
            else
                this->m_E = this->VO_CalcErrorImage(this->m_ImageProcessing,
                                                    this->m_VOEstimatedShape,
                                                    this->m_VOModelNormalizedTexture,
                                                    this->m_VOEstimatedTextureError);

            if (this->m_E < this->m_E_previous)
            {
                this->m_MatEstimatedC.copyTo(this->m_MatCurrentC);
                this->m_VOFittingShape.clone(this->m_VOEstimatedShape);
                this->m_VOTextureError.clone(this->m_VOEstimatedTextureError);
                this->m_E_previous = this->m_E;
                cBetter = true;
                this->m_fScale = estScale;
                this->m_vRotateAngles = estRotateAngles;
                this->m_MatCenterOfGravity = estCOG.clone();
                break;
            }
        }

        /** Second pose, t parameters. refer to equation (9.3)
        * Cootes "Statistical Model of Appearance for Computer Vision" */
        cv::gemm(this->m_VOTextureError.GetTheTextureInARow(), this->m_VOAAMBasic->m_MatRt, -1, Mat(), 0, this->m_MatDeltaT, GEMM_2_T);

        // damp -- T
        for(unsigned int i = 0; i < k_values.size(); i++)
        {
            // make damped c/pose prediction
            cv::scaleAdd(this->m_MatDeltaT, k_values[i], this->m_MatCurrentT, this->m_MatEstimatedT);
            this->VO_CParamTParam2FittingShape( this->m_MatCurrentC,
                                                this->m_MatEstimatedT,
                                                this->m_VOModelNormalizedTexture,
                                                this->m_VOEstimatedShape,
                                                estScale,
                                                estRotateAngles,
                                                estCOG);
            if ( !VO_ShapeModel::VO_IsShapeInsideImage(this->m_VOEstimatedShape, this->m_ImageProcessing) )
                continue;
            else
                this->m_E = this->VO_CalcErrorImage(this->m_ImageProcessing,
                                                    this->m_VOEstimatedShape,
                                                    this->m_VOModelNormalizedTexture,
                                                    this->m_VOEstimatedTextureError);

            if (this->m_E < this->m_E_previous)
            {
                // Since m_fScale, m_vRotateAngles and m_MatCenterOfGravity have been updated,
                // m_MatCurrentT should be assigned to 0 now!
                this->m_MatCurrentT = Mat_<float>::zeros(this->m_MatCurrentT.size());
                //                this->m_MatEstimatedT.copyTo(this->m_MatCurrentT);
                this->m_VOFittingShape.clone(this->m_VOEstimatedShape);
                this->m_VOTextureError.clone(this->m_VOEstimatedTextureError);
                this->m_E_previous = this->m_E;
                poseBetter = true;
                this->m_fScale = estScale;
                this->m_vRotateAngles = estRotateAngles;
                this->m_MatCenterOfGravity = estCOG.clone();
                break;
            }
        }

        if( cBetter || poseBetter)
        {
            ioShape.clone(this->m_VOFittingShape);
        }
        else
        break;

        ++this->m_iIteration;

    }while( ( fabs(this->m_E) > FLT_EPSILON ) && (this->m_iIteration < epoch)/* && (cv::norm(this->m_MatDeltaC) > FLT_EPSILON) */ );

t = ((double)cvGetTickCount() -  t )/  (cvGetTickFrequency()*1000.);
cout << "Basic fitting time cost: " << t << " millisec" << endl;
this->m_fFittingTime = safeDoubleToFloat(t);

    VO_Fitting2DSM::VO_DrawMesh(ioShape, this->m_VOAAMBasic, oImg);

    return safeDoubleToFloat(t);
}


/**
 * @author         JIA Pei
 * @version        2010-05-20
 * @brief          Direct AAM Fitting, for static images, so that we record the whole fitting process
 * @param          iImg            Input - image to be fitted
 * @param          epoch           Input - the iteration epoch
 * @param          oImages         Output - the fitted shape
*/
float VO_FittingAAMBasic::VO_DirectAAMFitting(const Mat& iImg,
												vector<DrawMeshInfo>& oMeshInfo,
                                                unsigned int epoch,
                                                bool record)
{
double t = (double)cvGetTickCount();

t = ((double)cvGetTickCount() -  t )/  (cvGetTickFrequency()*1000.);
cout << "Direct fitting time cost: " << t << " millisec" << endl;
    
    return safeDoubleToFloat(t);
}


/**
 * @author      JIA Pei
 * @version     2010-05-20
 * @brief       Direct AAM Fitting, for dynamic image sequence
 * @param       iImg            Input - image to be fitted
 * @param       ioShape         Input and Output - the fitted shape
 * @param       oImg            Output - the fitted image
 * @param       epoch           Input - the iteration epoch
*/
float VO_FittingAAMBasic::VO_DirectAAMFitting(const Mat& iImg,
                                                VO_Shape& ioShape,
                                                Mat& oImg,
                                                unsigned int epoch)
{
double t = (double)cvGetTickCount();
    
t = ((double)cvGetTickCount() -  t )/  (cvGetTickFrequency()*1000.);
cout << "Direct fitting time cost: " << t << " millisec" << endl;

    return safeDoubleToFloat(t);
}

