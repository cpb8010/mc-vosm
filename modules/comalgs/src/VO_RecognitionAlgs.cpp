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


#include <iostream>
#include <cstdio>
#include <functional>
#include <numeric>
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "VO_FaceKeyPoint.h"
#include "VO_RecognitionAlgs.h"




/**
* @param    trackalg-   input and output    the track algorithm,
                        will record some information for every frame
* @param    iImg    - input     input image
* @param    iShape  - input     the current tracked shape
* @return   bool    whether the tracked shape is acceptable?
*/
bool CRecognitionAlgs::EvaluateFaceTrackedByProbabilityImage(
    CTrackingAlgs* trackalg,
    const Mat& iImg,
    const VO_Shape& iShape,
    Size smallSize,
    Size bigSize)
{
    double t = (double)cvGetTickCount();

    Rect rect = iShape.GetShapeBoundRect();

    trackalg->SetConfiguration( CTrackingAlgs::CAMSHIFT,
                                CTrackingAlgs::PROBABILITYIMAGE);
    trackalg->Tracking( rect,
                        iImg,
                        smallSize,
                        bigSize );

    bool res = false;
    if( !trackalg->IsObjectTracked() )
        res = false;
    else if ( ((double)rect.height/(double)rect.width <= 0.75)
        || ((double)rect.height/(double)rect.width >= 2.5) )
        res = false;
    else
        res = true;

    t = ((double)cvGetTickCount() -  t )
        / (cvGetTickFrequency()*1000.);
    cout << "Camshift Tracking time cost: " << t << "millisec" << endl;

    return res;
}


/**
* @brief    whether the tracked shape is really a face?
*           If we can detect both eyes and mouth
*           according to some prior knowledge due to its shape,
*           we may regard this shape correctly describe a face.
* @param    iImg        - input     input image
* @param    iShape      - input     the current tracked shape
* @param    iShapeInfo  - input     shape info
* @param    iFaceParts  - input     face parts
* @return   bool    whether the tracked shape is acceptable?
*/
bool CRecognitionAlgs::EvaluateFaceTrackedByCascadeDetection(
    const CFaceDetectionAlgs* fd,
    const Mat& iImg,
    const VO_Shape& iShape,
    const vector<VO_Shape2DInfo>& iShapeInfo, 
    const VO_FaceParts& iFaceParts)
{
    double t = (double)cvGetTickCount();

    unsigned int ImgWidth       = iImg.cols;
    unsigned int ImgHeight      = iImg.rows;

    vector<unsigned int> leftEyePoints      = iFaceParts.VO_GetOneFacePart(VO_FacePart::LEFTEYE).GetIndexes();
    vector<unsigned int> rightEyePoints     = iFaceParts.VO_GetOneFacePart(VO_FacePart::RIGHTEYE).GetIndexes();
    vector<unsigned int> lipOuterLinerPoints= iFaceParts.VO_GetOneFacePart(VO_FacePart::LIPOUTERLINE).GetIndexes();

    VO_Shape leftEyeShape       = iShape.GetSubShape(leftEyePoints);
    VO_Shape rightEyeShape      = iShape.GetSubShape(rightEyePoints);
    VO_Shape lipOuterLinerShape = iShape.GetSubShape(lipOuterLinerPoints);

    float dolEye = 12.0f;
    float dolMouth = 12.0f;

    unsigned int possibleLeftEyeMinX    = 0.0f > (leftEyeShape.MinX() - dolEye) ? 0: (int)(leftEyeShape.MinX() - dolEye);
    unsigned int possibleLeftEyeMinY    = 0.0f > (leftEyeShape.MinY() - dolEye) ? 0: (int)(leftEyeShape.MinY() - dolEye);
    unsigned int possibleLeftEyeMaxX    = (leftEyeShape.MaxX() + dolEye) > ImgWidth ? ImgWidth : (int)(leftEyeShape.MaxX() + dolEye);
    unsigned int possibleLeftEyeMaxY    = (leftEyeShape.MaxY() + dolEye) > ImgHeight ? ImgHeight : (int)(leftEyeShape.MaxY() + dolEye);
    unsigned int possibleLeftEyeWidth   = possibleLeftEyeMaxX - possibleLeftEyeMinX;
    unsigned int possibleLeftEyeHeight  = possibleLeftEyeMaxY - possibleLeftEyeMinY;
    unsigned int possibleRightEyeMinX   = 0.0f > (rightEyeShape.MinX() - dolEye) ? 0: (int)(rightEyeShape.MinX() - dolEye);
    unsigned int possibleRightEyeMinY   = 0.0f > (rightEyeShape.MinY() - dolEye) ? 0: (int)(rightEyeShape.MinY() - dolEye);
    unsigned int possibleRightEyeMaxX   = (rightEyeShape.MaxX() + dolEye) > ImgWidth ? ImgWidth : (int)(rightEyeShape.MaxX() + dolEye);
    unsigned int possibleRightEyeMaxY   = (rightEyeShape.MaxY() + dolEye) > ImgHeight ? ImgHeight : (int)(rightEyeShape.MaxY() + dolEye);
    unsigned int possibleRightEyeWidth  = possibleRightEyeMaxX - possibleRightEyeMinX;
    unsigned int possibleRightEyeHeight = possibleRightEyeMaxY - possibleRightEyeMinY;
    unsigned int possibleMouthMinX      = 0.0f > (lipOuterLinerShape.MinX() - dolMouth) ? 0: (int)(lipOuterLinerShape.MinX() - dolMouth);
    unsigned int possibleMouthMinY      = 0.0f > (lipOuterLinerShape.MinY() - dolMouth) ? 0: (int)(lipOuterLinerShape.MinY() - dolMouth);
    unsigned int possibleMouthMaxX      = (lipOuterLinerShape.MaxX() + dolMouth) > ImgWidth ? ImgWidth : (int)(lipOuterLinerShape.MaxX() + dolMouth);
    unsigned int possibleMouthMaxY      = (lipOuterLinerShape.MaxY() + dolMouth) > ImgHeight ? ImgHeight : (int)(lipOuterLinerShape.MaxY() + dolMouth);
    unsigned int possibleMouthWidth     = possibleMouthMaxX - possibleMouthMinX;
    unsigned int possibleMouthHeight    = possibleMouthMaxY - possibleMouthMinY;

    Rect LeftEyePossibleWindow  = Rect( possibleLeftEyeMinX, possibleLeftEyeMinY, possibleLeftEyeWidth, possibleLeftEyeHeight );
    Rect RightEyePossibleWindow = Rect( possibleRightEyeMinX, possibleRightEyeMinY, possibleRightEyeWidth, possibleRightEyeHeight );
    Rect MouthPossibleWindow    = Rect( possibleMouthMinX, possibleMouthMinY, possibleMouthWidth, possibleMouthHeight );
    Rect CurrentWindow          = Rect( 0, 0, iImg.cols, iImg.rows );
    Rect DetectedLeftEyeWindow, DetectedRightEyeWindow, DetectedMouthWindow;

    bool LeftEyeDetected    = const_cast<CFaceDetectionAlgs*>(fd)->VO_FacePartDetection ( iImg, LeftEyePossibleWindow, DetectedLeftEyeWindow, VO_FacePart::LEFTEYE);
    bool RightEyeDetected   = const_cast<CFaceDetectionAlgs*>(fd)->VO_FacePartDetection ( iImg, RightEyePossibleWindow, DetectedRightEyeWindow, VO_FacePart::RIGHTEYE );
    bool MouthDetected      = const_cast<CFaceDetectionAlgs*>(fd)->VO_FacePartDetection ( iImg, MouthPossibleWindow, DetectedMouthWindow, VO_FacePart::LIPOUTERLINE );

    t = ((double)cvGetTickCount() -  t )
        / (cvGetTickFrequency()*1000.0f);
    cout << "Detection Confirmation time cost: " << t << "millisec" << endl;

    if(LeftEyeDetected && RightEyeDetected && MouthDetected)
        return true;
    else
        return false;
}


/**
* @param    shape1  - input     shape1
* @param    shape2  - input     shape2
* @return   the shape distance
*/
float CRecognitionAlgs::ShapeDistance(  const VO_Shape& shape1,
                                        const VO_Shape& shape2)
{
    VO_Shape shapediff = const_cast<VO_Shape&>(shape1) - shape2;
    return shapediff.GetShapeNorm();
}


/**
* @param    avgSParam       - input     mean shape parameters
* @param    icovSParam      - input     covariance matrix of shape parameters
* @param    avgTParam       - input     mean texture parameters
* @param    icovTParam      - input     covariance matrix of texture parameters
* @param    iSParams        - input     the input shape parameter
* @param    iTParams        - input     the input texture parameter
* @param    ShapeDistMean   - input     mean texture parameters
* @param    ShapeDistStddev - input     covariance matrix of texture parameters
* @param    TextureDistMean - input     the input shape parameter
* @param    TextureDistStddev- input    the input texture parameter
* @param    sDist           - output    shape distance
* @param    tDist           - output    texture distance
* @param    WeakFitting     - input     only shape parameter is used?
* @return   whether the fitting is acceptable
*/
bool CRecognitionAlgs::CalcFittingEffect4StaticImage(
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
    bool WeakFitting )
{
	sDist = safeDoubleToFloat(cv::Mahalanobis( iSParams, avgSParam, icovSParam ));
	tDist = safeDoubleToFloat(cv::Mahalanobis( iTParams, avgTParam, icovTParam ));

    if(WeakFitting)
    {
        if( ( fabs( sDist - ShapeDistMean.val[0] )
            < 1.5f * ShapeDistStddev.val[0] ) )
            return true;
        else
            return false;
    }
    else
    {
        if( ( fabs( sDist - ShapeDistMean.val[0] )
            < 1.5f * ShapeDistStddev.val[0] ) && 
            ( fabs( tDist - TextureDistMean.val[0] )
            < 3.0f*TextureDistStddev.val[0] ) )
            return true;
        else
            return false;
    }
}


/**
* @param    avgSParam       - input        mean shape parameters
* @param    icovSParam      - input        covariance matrix of shape parameters
* @param    avgTParam       - input        mean texture parameters
* @param    icovTParam      - input        covariance matrix of texture parameters
* @param    iSParams        - input        the vector of multiple input shape parameters
* @param    iTParams        - input        the vector of multiple input texture parameter
* @param    ShapeDistMean   - input        mean texture parameters
* @param    ShapeDistStddev - input        covariance matrix of texture parameters
* @param    TextureDistMean - input        the input shape parameter
* @param    TextureDistStddev   - input    the input texture parameter
* @param    WeakFitting     - input        only shape parameter is used?
* @return   whether the fitting is acceptable
*/
bool CRecognitionAlgs::CalcFittingEffect4ImageSequence( 
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
    bool WeakFitting )
{
    assert(iSParams.rows == iTParams.rows);
    unsigned int NbOfSamples = iSParams.rows;
    vector<float> sDists, tDists;
    sDists.resize(NbOfSamples);
    tDists.resize(NbOfSamples);

    for(unsigned int i = 0; i < NbOfSamples; ++i)
    {
        CRecognitionAlgs::CalcFittingEffect4StaticImage(
            avgSParam,
            icovSParam,
            avgTParam,
            icovTParam,
            iSParams.row(i),
            iTParams.row(i),
            ShapeDistMean,
            ShapeDistStddev,
            TextureDistMean,
            TextureDistStddev,
            sDists[i],
            tDists[i],
            WeakFitting );
    }

    unsigned int NbOfGood1 = 0;
    unsigned int NbOfGood2 = 0;

    for(unsigned int i = 0; i < NbOfSamples; ++i)
    {
        if( ( fabs( sDists[i] - ShapeDistMean.val[0] )
            < 1.5f * ShapeDistStddev.val[0] ) )
        {
            NbOfGood1++;
            if( ( fabs( tDists[i] - TextureDistMean.val[0] )
                < 3.0f*TextureDistStddev.val[0] ) )
            {
                NbOfGood2++;
            }
        }
    }

    if(WeakFitting)
    {
        if(NbOfGood1 >= (unsigned int )(0.75*NbOfSamples) )
            return true;
        else
            return false;
    }
    else
    {
        if(NbOfGood2 >= (unsigned int )(0.75*NbOfGood1) )
            return true;
        else
            return false;
    }
}


/**
* @brief    Calculate face fitting effect
* @param    refShape    - input     reference shape
* @param    fittedShape - input     fitting result
* @param    deviation   - output    what is the deviation from refShape to fittedShape
* @param    ptErrorFreq - output    point error frequency
* @param    nb          - input     how many evaluation levels that is to be used
* @return   whether the fitting is acceptable
*/
void CRecognitionAlgs::CalcShapeFittingEffect(	const VO_Shape& refShape,
												const VO_Shape& fittedShape,
												float& deviation,
												vector<float>& ptErrorFreq,
												int nb,
												vector<float>* ptErrPerPoint)
{
    assert(refShape.GetNbOfDim() == fittedShape.GetNbOfDim());
	assert(refShape.GetNbOfPoints() == fittedShape.GetNbOfPoints());
    unsigned int NbOfShapeDim   = refShape.GetNbOfDim();
    unsigned int NbOfPoints     = refShape.GetNbOfPoints();
	ptErrorFreq.resize(nb);

	vector<float> ptDists(NbOfPoints, 0.0f);
	
	for(unsigned int i = 0; i < NbOfPoints; i++)
	{
		ptDists[i] = 0.0f;
		for(unsigned int j = 0; j < NbOfShapeDim; j++)
		{
			ptDists[i] += pow(refShape.GetAShape(j*NbOfPoints+i) - fittedShape.GetAShape(j*NbOfPoints+i), 2.0f);
		}
		ptDists[i] = sqrt(ptDists[i]);
	}
	
	ptErrorFreq.resize(nb);
	for(int i = 0; i < nb; i++)
	{
		for (unsigned int j = 0; j < NbOfPoints; j++)
		{
			if (ptDists[j] < i)
			{
				ptErrorFreq[i]++;
			}
		}
		ptErrorFreq[i] /= static_cast<float>(NbOfPoints);
	}
	float sumPtDist = 0.0;
	for(unsigned int i = 0; i<NbOfPoints;++i){
		sumPtDist += ptDists[i];
	}
	printf("Avg ptDists = %f\n",sumPtDist/NbOfPoints);

    deviation = CRecognitionAlgs::ShapeDistance(refShape, fittedShape);
	if(ptErrPerPoint != 0){
		(*ptErrPerPoint) = ptDists;
	}
}


/**
 * @param	fd					- input		folder name
 * @param	fnIdx				- input		fitting result
 * @param	deviation			- input		what is the deviation from refShape to fittedShape
 * @param	ptErrorFreq			- input		for curve to display frequency -- point distance
 * @return	whether the fitting is acceptable
 */
void CRecognitionAlgs::SaveShapeRecogResults(	const string& fd,
												const string& fnIdx,
												float deviation,
												vector<float>& ptErrorFreq)
{
    string fn;
    fn = fd + "/" + fnIdx + ".res";
    
    fstream fp;
    fp.open(fn.c_str (), ios::out);

    fp << "Total Deviation" << endl << deviation << endl;				// deviation
    fp << "Point Error -- Frequency" << endl;
    for(unsigned int i = 0; i < ptErrorFreq.size(); i++)
    {
        fp << ptErrorFreq[i] << " ";
    }
	
    fp << endl;
	
    fp.close();fp.clear();
}

/**
 * @param	fd					- input		folder name
 * @param	fnIdx				- input		fitting result
 * @param	deviation			- input		what is the deviation from refShape to fittedShape
 * @param	ptErrorFreq			- input		for curve to display frequency -- point distance
 * @param	fittedShape			- input		fitting result
 * @return	whether the fitting is acceptable
 */
void CRecognitionAlgs::SaveShapeResults(		const string& fd,
												const string& fnIdx,
												float deviation,
												vector<float>& ptDists,
												vector<float>& ptErrorFreq,
												const VO_Shape& fittedShape)
{
    string fn;
    fn = fd + "/" + fnIdx + ".res";
    
    fstream fp;
    fp.open(fn.c_str (), ios::out);

	fp << "Error per point -- Distance from ground truth" << endl;
	for(unsigned int i = 0; i < ptDists.size(); ++i){
		fp << ptDists[i] << endl;
	}
	fp << endl;

	fp << "Total landmark error" << endl;
	float errSum = std::accumulate(ptDists.begin(),ptDists.end(),0.0f);
	fp << errSum << endl;
	fp <<"Average landmark distance" << endl;
	fp << errSum / ptDists.size() << endl;
	fp << endl;

    fp << "Total Deviation" << endl << deviation << endl;				// deviation
    fp << "Point Error -- Frequency" << endl;
    for(unsigned int i = 0; i < ptErrorFreq.size(); i++)
    {
        fp << ptErrorFreq[i] << " ";
    }
	fp << endl;
	fp << endl;
	fp << "Fitted points" << endl;
	//output actual points along with error frequency
	unsigned int NbOfShapeDim   = fittedShape.GetNbOfDim();
	unsigned int NbOfPoints     = fittedShape.GetNbOfPoints();
	for(unsigned int i = 0; i < NbOfPoints; i++)
	{
		for(unsigned int j = 0; j < NbOfShapeDim; j++)
		{
			fp << fittedShape.GetAShape(j*NbOfPoints+i) << " ";
		}
		fp << endl;
	}
    fp << endl;
	
    fp.close();fp.clear();
}

/**
 * @param	fd					- input		folder name
 * @param	fnIdx				- input		fitting result
 * @param	deviation			- input		what is the deviation from refShape to fittedShape
 * @param	ptErrorFreq			- input		for curve to display frequency -- point distance
 * @param	fittedShape			- input		fitting result
 * @param	gt_cp				- input		ground truth canidate points
 * @param	t_cp				- input		tested canidate points (l eye, r eye, mouth)
 * @return	whether the fitting is acceptable
 */
void CRecognitionAlgs::SaveFittingResults(		const string& fd,
												const string& fnIdx,
												float deviation,
												vector<float>& ptDists,
												vector<float>& ptErrorFreq,
												const VO_Shape& fittedShape,
												cv::Point2f* gt_cP,
												cv::Point2f* t_cP,
												float fitTime)
{
    string fn;
    fn = fd + "/" + fnIdx + ".res";
    
    fstream fp;
    fp.open(fn.c_str (), ios::out);

	fp << "Error per point -- Distance from ground truth" << endl;
	for(unsigned int i = 0; i < ptDists.size(); ++i){
		fp << ptDists[i] << endl;
	}
	fp << endl;

	fp << "Total landmark error" << endl;
	float errSum = std::accumulate(ptDists.begin(),ptDists.end(),0.0f);
	fp << errSum << endl;
	fp << "Average landmark distance" << endl;
	fp << errSum / ptDists.size() << endl;
	fp << "Candidate point error (Left eye, Right eye, Mouth)" << endl;
	//messy distance, too lazy
	float le_dist = sqrt(pow(gt_cP[0].x - t_cP[0].x,2) + pow(gt_cP[0].y - t_cP[0].y,2));
	float re_dist = sqrt(pow(gt_cP[1].x - t_cP[1].x,2) + pow(gt_cP[1].y - t_cP[1].y,2));
	float m_dist = sqrt(pow(gt_cP[2].x - t_cP[2].x,2) + pow(gt_cP[2].y - t_cP[2].y,2));

	fp << le_dist << endl;
	fp << re_dist << endl;
	fp << m_dist << endl;
	fp << endl;
	fp << "Fitting time" << endl;
	fp << fitTime << endl;
	fp << endl;

    fp << "Total deviation" << endl << deviation << endl;				// deviation
    fp << "Point error -- Frequency" << endl;
    for(unsigned int i = 0; i < ptErrorFreq.size(); i++)
    {
        fp << ptErrorFreq[i] << " ";
    }
	fp << endl;
	fp << endl;
	fp << "Canidate points" << endl;
	fp << t_cP[0].x << " " << t_cP[0].y << endl;
	fp << t_cP[1].x << " " << t_cP[1].y << endl;
	fp << t_cP[2].x << " " << t_cP[2].y << endl;
	fp << "Fitted points" << endl;
	//output actual points along with error frequency
	unsigned int NbOfShapeDim   = fittedShape.GetNbOfDim();
	unsigned int NbOfPoints     = fittedShape.GetNbOfPoints();
	for(unsigned int i = 0; i < NbOfPoints; i++)
	{
		for(unsigned int j = 0; j < NbOfShapeDim; j++)
		{
			fp << fittedShape.GetAShape(j*NbOfPoints+i) << " ";
		}
		fp << endl;
	}
    fp << endl;
	
    fp.close();fp.clear();
}

// Estimate face absolute orientations
vector<float> CRecognitionAlgs::CalcAbsoluteOrientations(
    const VO_Shape& iShape2D,
    const VO_Shape& iShape3D,
    VO_Shape& oShape2D)
{
    assert (iShape2D.GetNbOfPoints() == iShape3D.GetNbOfPoints() );
    unsigned int NbOfPoints = iShape3D.GetNbOfPoints();
    Point3f pt3d;
    Point2f pt2d;
    float height1 = iShape2D.GetHeight();
    float height2 = iShape3D.GetHeight();
    VO_Shape tempShape2D = iShape2D;
    tempShape2D.Scale(height2/height1);

    //Create the model points
    std::vector<CvPoint3D32f> modelPoints;
    for(unsigned int i = 0; i < NbOfPoints; ++i)
    {
        pt3d = iShape3D.GetA3DPoint(i);
        modelPoints.push_back(cvPoint3D32f(pt3d.x, pt3d.y, pt3d.z));
    }

    //Create the image points
    std::vector<CvPoint2D32f> srcImagePoints;
    for(unsigned int i = 0; i < NbOfPoints; ++i)
    {
        pt2d = tempShape2D.GetA2DPoint(i);
        srcImagePoints.push_back(cvPoint2D32f(pt2d.x, pt2d.y));
    }

    //Create the POSIT object with the model points
    CvPOSITObject *positObject = cvCreatePOSITObject( &modelPoints[0], NbOfPoints );

    //Estimate the pose
    CvMatr32f rotation_matrix = new float[9];
    CvVect32f translation_vector = new float[3];
    CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 100, 1.0e-4f);
    cvPOSIT( positObject, &srcImagePoints[0], FOCAL_LENGTH, criteria, rotation_matrix, translation_vector );

    //rotation_matrix to Euler angles, refer to VO_Shape::GetRotation
    float sin_beta  = -rotation_matrix[0 * 3 + 2];
    float tan_alpha = rotation_matrix[1 * 3 + 2] / rotation_matrix[2 * 3 + 2];
    float tan_gamma = rotation_matrix[0 * 3 + 1] / rotation_matrix[0 * 3 + 0];

    //Project the model points with the estimated pose
    oShape2D = tempShape2D;
    for ( unsigned int i=0; i < NbOfPoints; ++i )
    {
        pt3d.x = rotation_matrix[0] * modelPoints[i].x +
            rotation_matrix[1] * modelPoints[i].y +
            rotation_matrix[2] * modelPoints[i].z +
            translation_vector[0];
        pt3d.y = rotation_matrix[3] * modelPoints[i].x +
            rotation_matrix[4] * modelPoints[i].y +
            rotation_matrix[5] * modelPoints[i].z +
            translation_vector[1];
        pt3d.z = rotation_matrix[6] * modelPoints[i].x +
            rotation_matrix[7] * modelPoints[i].y +
            rotation_matrix[8] * modelPoints[i].z +
            translation_vector[2];
        if ( pt3d.z != 0 )
        {
            pt2d.x = FOCAL_LENGTH * pt3d.x / pt3d.z;
            pt2d.y = FOCAL_LENGTH * pt3d.y / pt3d.z;
        }
        oShape2D.SetA2DPoint(pt2d, i);
    }

    //return Euler angles
    vector<float> pos(3);
    pos[0] = atan(tan_alpha);    // yaw
    pos[1] = asin(sin_beta);     // pitch
    pos[2] = atan(tan_gamma);    // roll
    return pos;
}


float CRecognitionAlgs::CalcFaceRoll(const vector<float>& iLine)
{
    float roll = 0.0f;

    if(iLine[1] < FLT_MIN) roll = 0.0f;
    else
    {
        float temp = atan ( iLine[0] / iLine[1] ) * safeDoubleToFloat(180.0 / CV_PI);
        roll =  temp >= 0.0f ? - (90.0f - temp ) : (90.0f + temp);
    }
    return roll;
}


float CRecognitionAlgs::CalcFaceYaw(const vector<float>& iLine,
                                    const VO_Shape& iShape,
                                    const VO_FaceParts& iFaceParts)
{
    float yaw = 0.0f;
    int dim = iShape.GetNbOfDim();

    // Theoretically, using eye corner is correct, but it's not stable at all. Therefore, here we use COG_left and COG_right instead.
    ///////////////////////////////////////////////////////////////////////////////
    //     float leftDist = 0.0f, rightDist = 0.0f;    
    //     vector<unsigned int> eyeCornerPoints = iFaceParts.GetEyeCornerPoints().GetIndexes();
    //     Point2f leftmostEyeCorner = Point2f(FLT_MAX, 0.0f);
    //     Point2f rightmostEyeCorner = Point2f(0.0f, 0.0f);
    // 
    //     for(unsigned int i = 0; i < eyeCornerPoints.size(); ++i)
    //     {
    //         if(leftmostEyeCorner.x > iShape.GetAShape(dim*eyeCornerPoints[i]) )
    //         {
    //             leftmostEyeCorner.x = iShape.GetAShape(dim*eyeCornerPoints[i]);
    //             leftmostEyeCorner.y = iShape.GetAShape(dim*eyeCornerPoints[i]+1);
    //         }
    //         if(rightmostEyeCorner.x < iShape.GetAShape(dim*eyeCornerPoints[i]) )
    //         {
    //             rightmostEyeCorner.x = iShape.GetAShape(dim*eyeCornerPoints[i]);
    //             rightmostEyeCorner.y = iShape.GetAShape(dim*eyeCornerPoints[i]+1);
    //         }
    //     }
    //     leftDist = cvDistFromAPoint2ALine2D(leftmostEyeCorner,  iLine);
    //     rightDist = cvDistFromAPoint2ALine2D(rightmostEyeCorner,  iLine);
    //     float r = leftDist/rightDist;
    // Refer to my PhD dissertation. Chapter 4
    //     yaw = atan ( ( 0.65*(r-1) ) / ( 0.24 * (r+1) ) ) * 180.0f / CV_PI;
    ///////////////////////////////////////////////////////////////////////////////

    float leftDist = 0.0f, rightDist = 0.0f;
    vector<unsigned int> leftSidePoints = iFaceParts.VO_GetOneFacePart(VO_FacePart::LEFTSIDEPOINTS).GetIndexes();
    vector<unsigned int> rightSidePoints = iFaceParts.VO_GetOneFacePart(VO_FacePart::RIGHTSIDEPOINTS).GetIndexes();
    for(unsigned int i = 0; i < leftSidePoints.size(); ++i)
    {
        leftDist += cvDistFromAPoint2ALine2D(Point2f(iShape.GetAShape(dim*leftSidePoints[i]), iShape.GetAShape(dim*leftSidePoints[i]+1)),  iLine);
    }
    for(unsigned int i = 0; i < rightSidePoints.size(); ++i)
    {
        rightDist += cvDistFromAPoint2ALine2D(Point2f(iShape.GetAShape(dim*rightSidePoints[i]), iShape.GetAShape(dim*rightSidePoints[i]+1)),  iLine);
    }

    float r = leftDist/rightDist;
    // Refer to my PhD dissertation. Chapter 4
    // yaw = atan ( ( 0.65*(r-1) ) / ( 0.24 * (r+1) ) ) * 180.0f / CV_PI;
    yaw = atan ( ( (r-1) ) / ((r+1) ) ) * safeDoubleToFloat(180.0 / CV_PI);

    return yaw;
}


// Refer to my PhD thesis, chapter 4
float CRecognitionAlgs::CalcFacePitch(  const VO_Shape& iShape,
                                        const VO_FaceParts& iFaceParts)
{
    float pitch = 0.0f;
    int dim = iShape.GetNbOfDim();
    float NNQ, ENQ, EQ, NO;

    // Theoretically, using eye corner is correct, but it's not quite stable at all. It's better we use two nostrils first if nostirl is defined in faceparts
    ///////////////////////////////////////////////////////////////////////////////
    //     unsigned int nosetipBottom = 0;
    //     vector<unsigned int> nosePoints             = iFaceParts.GetNose().GetIndexes();
    //     vector<unsigned int> midlinePoints         = iFaceParts.GetMidlinePoints().GetIndexes();
    //     vector<unsigned int> pitchAxisPoints    = iFaceParts.GetPitchAxisLinePoints().GetIndexes();
    //     VO_Shape nose, midLine, pitchAxis;
    //     nose.SetDim(dim);
    //     midLine.SetDim(dim);
    //     pitchAxis.SetDim(dim);
    //     nose.SetSize( nosePoints.size()*dim );
    //     midLine.SetSize( midlinePoints.size()*dim );
    //     pitchAxis.SetSize(pitchAxisPoints.size()*dim );
    // 
    //     for(unsigned int i = 0; i < nosePoints.size(); ++i)
    //     {
    //         for(unsigned int j = 0; j < midlinePoints.size(); ++j)
    //         {
    //             if(nosePoints[i] == midlinePoints[j])
    //             {
    //                 nosetipBottom = nosePoints[i];
    //                 break;
    //             }
    //         }
    //     }
    // 
    //     Point2f ntPoint  = Point2f(iShape.GetAShape(dim*nosetipBottom), iShape.GetAShape(dim*nosetipBottom+1));
    //     Point2f paPoint1 = Point2f(iShape.GetAShape(dim*pitchAxisPoints[0]), iShape.GetAShape(dim*pitchAxisPoints[0]+1));
    //     Point2f paPoint2 = Point2f(iShape.GetAShape(dim*pitchAxisPoints[1]), iShape.GetAShape(dim*pitchAxisPoints[1]+1));
    // 
    //     float NNQ = ( (ntPoint.y - paPoint1.y) + (ntPoint.y - paPoint2.y) ) / 2.0f;
    //     float ENQ = fabs(ntPoint.x - paPoint1.x) > fabs(paPoint2.x - ntPoint.x) ? fabs(ntPoint.x - paPoint1.x) : fabs(paPoint2.x - ntPoint.x);
    //     float EQ = sqrt(ENQ*ENQ + NNQ*NNQ);
    //     float NO = sqrt(2.0f)/2.0f*EQ;
    ///////////////////////////////////////////////////////////////////////////////

    vector<unsigned int> nostrilPoints          = iFaceParts.VO_GetOneFacePart(VO_FacePart::NOSTRIL).GetIndexes();
    if(nostrilPoints.size() != 0)
    {
        vector<unsigned int> pitchAxisPoints    = iFaceParts.VO_GetOneFacePart(VO_FacePart::PITCHAXISLINEPOINTS).GetIndexes();

        Point2f ntPoint1 = Point2f(iShape.GetAShape(dim*nostrilPoints[0]), iShape.GetAShape(dim*nostrilPoints[0]+1));
        Point2f ntPoint2 = Point2f(iShape.GetAShape(dim*nostrilPoints[1]), iShape.GetAShape(dim*nostrilPoints[1]+1));
        Point2f paPoint1 = Point2f(iShape.GetAShape(dim*pitchAxisPoints[0]), iShape.GetAShape(dim*pitchAxisPoints[0]+1));
        Point2f paPoint2 = Point2f(iShape.GetAShape(dim*pitchAxisPoints[1]), iShape.GetAShape(dim*pitchAxisPoints[1]+1));

        NNQ = ( (ntPoint1.y - paPoint1.y) + (ntPoint2.y - paPoint2.y) ) / 2.0f;
        ENQ = fabs(ntPoint1.x - paPoint1.x) > fabs(paPoint2.x - ntPoint2.x) ? fabs(ntPoint1.x - paPoint1.x + (ntPoint2.x - ntPoint1.x) / 2.0f) : fabs(paPoint2.x - ntPoint2.x + (ntPoint2.x - ntPoint1.x) / 2.0f);
        EQ = sqrt(ENQ*ENQ + NNQ*NNQ);
        NO = sqrt(2.0f)/2.0f*EQ;
    }
    else
    {
        unsigned int nosetipBottom = 0;
        vector<unsigned int> nosePoints         = iFaceParts.VO_GetOneFacePart(VO_FacePart::NOSE).GetIndexes();
        vector<unsigned int> midlinePoints      = iFaceParts.VO_GetOneFacePart(VO_FacePart::MIDLINEPOINTS).GetIndexes();
        vector<unsigned int> pitchAxisPoints    = iFaceParts.VO_GetOneFacePart(VO_FacePart::PITCHAXISLINEPOINTS).GetIndexes();

        for(unsigned int i = 0; i < nosePoints.size(); ++i)
        {
            for(unsigned int j = 0; j < midlinePoints.size(); ++j)
            {
                if(nosePoints[i] == midlinePoints[j])
                {
                    nosetipBottom = nosePoints[i];
                    break;
                }
            }
        }

        Point2f ntPoint  = Point2f(iShape.GetAShape(dim*nosetipBottom), iShape.GetAShape(dim*nosetipBottom+1));
        Point2f paPoint1 = Point2f(iShape.GetAShape(dim*pitchAxisPoints[0]), iShape.GetAShape(dim*pitchAxisPoints[0]+1));
        Point2f paPoint2 = Point2f(iShape.GetAShape(dim*pitchAxisPoints[1]), iShape.GetAShape(dim*pitchAxisPoints[1]+1));

        NNQ = ( (ntPoint.y - paPoint1.y) + (ntPoint.y - paPoint2.y) ) / 2.0f;
        ENQ = fabs(ntPoint.x - paPoint1.x) > fabs(paPoint2.x - ntPoint.x) ? fabs(ntPoint.x - paPoint1.x) : fabs(paPoint2.x - ntPoint.x);
        EQ = sqrt(ENQ*ENQ + NNQ*NNQ);
        NO = sqrt(2.0f)/2.0f*EQ;
    }

    if( fabs(NNQ/NO) < 1.0f)
        pitch = asin ( NNQ / NO ) * safeDoubleToFloat(180.0 / CV_PI);
    else if (NNQ * NO < 0.0f)
        pitch = -90.0f;
    else
        pitch = 90.0f;

    return pitch;
}


void CRecognitionAlgs::CalcFittedFaceAngle2D(   vector<float>& angles,
                                                const VO_Shape& iShape,
                                                const VO_FaceParts& iFaceParts)
{
    angles.resize(3);

    VO_Shape tempShape;

    // float facewidth = iShape.GetWidth();
    // int dim = iShape.GetDim();
    // vector<unsigned int> eyeCornerPoints = iFaceParts.GetEyeCornerPoints().GetIndexes();
    // VO_Shape eyeCorner;
    // eyeCorner.SetDim(dim);
    // eyeCorner.SetSize( eyeCornerPoints.size()*dim );
    // for(unsigned int i = 0; i < eyeCornerPoints.size(); i = ++i)
    // {
    //     for(unsigned int j = 0; j < dim; j++)
    //         eyeCorner.SetAShape( iShape.GetAShape(dim*eyeCornerPoints[i]+j), dim*i+j );
    // }
    // //cout << iShape << endl;
    // cout << iShape.GetAShape(eyeCornerPoints[0]*dim) << " " << iShape.GetAShape(eyeCornerPoints[0]*dim + 1) << endl;
    // cout << iShape.GetAShape(eyeCornerPoints[1]*dim) << " " << iShape.GetAShape(eyeCornerPoints[1]*dim + 1) << endl;
    // cout << iShape.GetAShape(eyeCornerPoints[2]*dim) << " " << iShape.GetAShape(eyeCornerPoints[2]*dim + 1) << endl;
    // cout << iShape.GetAShape(eyeCornerPoints[3]*dim) << " " << iShape.GetAShape(eyeCornerPoints[3]*dim + 1) << endl;
    // cout << eyeCorner << endl;
    // float eyewidth = eyeCorner.GetWidth();
    // float ratio = eyewidth/facewidth;


    vector<float> midline, eyecorner;
    VO_KeyPoint::CalcFaceKeyline(midline, iShape, iFaceParts, tempShape, VO_FacePart::MIDLINEPOINTS);
    VO_KeyPoint::CalcFaceKeyline(eyecorner, iShape, iFaceParts, tempShape, VO_FacePart::EYECORNERPOINTS);
    angles[2] = CRecognitionAlgs::CalcFaceRoll(midline);
    angles[1] = CRecognitionAlgs::CalcFaceYaw(midline, iShape, iFaceParts);
    angles[0] = CRecognitionAlgs::CalcFacePitch(iShape, iFaceParts);

    //     cout << "Pitch = " << angles[0] << endl;
    //     cout << "Yaw = " << angles[1] << endl;
    //     cout << "Roll = " << angles[2] << endl;
}


void CRecognitionAlgs::CalcFittedFaceAngle3D(   vector<float>& angles,
                                                const VO_Shape& iShape,
                                                const VO_FaceParts& iFaceParts)
{
    //     CvSeq* point_seq = cvCreateSeq( CV_32FC3, sizeof(CvSeq), sizeof(CvPoint3D32f), storage );
    //     for(unsigned int i = 0; i < midlinePoints.size(); ++i)
    //     {
    //         cvSeqPush(point_seq, &cvPoint3D32f(iShape.GetAShape(dim*midlinePoints[i]), iShape.GetAShape(dim*midlinePoints[i]+1), iShape.GetAShape(dim*midlinePoints[i] + 2)) );
    //     }
    // //    cvFitPlane( point_seq, CV_DIST_L2, 0, 0.001, 0.001, line );
    //     cvClearSeq(point_seq);
}


//     #include "opencv/cv.h"
//   #include <stdio.h>
//   #include <math.h>
//   float myLinearity(CvSeq *);
//   int main(void)
//   {
//     int i;
//     double fx[] = {0.0, 0.301, 0.477, 0.602, 0.699, 0.778, 0.845, 0.903, 0.954, 1.0};
//     double fy[] = {3.874, 3.202, 2.781, 2.49, 2.274, 2.156, 1.934, 1.74, 1.653, 1.662};
//     float *line = new float[4];
//     float linearity=0.0f;
//     //入れ物の確保
//     CvMemStorage* storage = cvCreateMemStorage(0);
//     //3次元の場合はCV_32FC2がCV_32FC3に、Point2fがCvPoint3D32fになる
//     CvSeq* point_seq = cvCreateSeq( CV_32FC2, sizeof(CvSeq), sizeof(Point2f), storage );
//     for (i=0; i<10; i++){
//       //値の追加はcvSeqPush
//       cvSeqPush(point_seq, &Point2f(fx[i],fy[i]));
//     }
//     linearity = myLinearity(point_seq);
//     cvFitLine(point_seq,CV_DIST_L2,0,0.01,0.01,line);
//     fprintf(stdout,"v=(%f,%f),vy/vx=%f,(x,y)=(%f,%f), Linearity=%f\n",line[0],line[1],line[1]/line[0],line[2],line[3],linearity);
//     cvClearSeq(point_seq);
//     cvReleaseMemStorage(&storage);
//     return 0;
//   }
//   //大津の直線度
//   float myLinearity(CvSeq *seq)
//   {
//     int i;
//     Point2f *p;
//     float *x = new float[seq->total];
//     float *y = new float[seq->total];
//     float x_bar=0.0, y_bar=0.0;
//     float u11=0.0, u20=0.0, u02=0.0;
//     float linearity=0.0;
//     //吸い出し cvGetSeqElemでポインタを得るのでキャストして取得
//     for (i=0; i < seq->total; i++){
//       p=(Point2f*)cvGetSeqElem(seq,i);
//       x[i]=p->x;
//       y[i]=p->y;
//     }
//     //x_bar, y_bar
//     for (i=0; i < seq->total; i++){
//       x_bar+=x[i];
//       y_bar+=y[i];
//     }
//     x_bar/=seq->total;
//     y_bar/=seq->total;
//     //セントラルモーメント
//     for (i=0; i < seq->total; i++){
//       u11+=((x[i]-x_bar)*(y[i]-y_bar));
//       u20+=pow(x[i]-x_bar,2.0f);
//       u02+=pow(y[i]-y_bar,2.0f);
//     }
//     u11/=seq->total;
//     u20/=seq->total;
//     u02/=seq->total;
//     //直線度の算出
//     linearity = sqrt(4*pow(u11,2.0f)+pow(u20-u02,2.0f))/(u20+u02);
//     return linearity;
//   }
