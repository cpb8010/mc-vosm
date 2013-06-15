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

#include <string>
#include <sstream>
#include <fstream>


#include "VO_FittingASMNDProfiles.h"


/** Constructor */
VO_FittingASMNDProfiles::VO_FittingASMNDProfiles()
{
    this->init();
}


/** Destructor */
VO_FittingASMNDProfiles::~VO_FittingASMNDProfiles()
{
    if(this->m_VOASMNDProfile)  delete this->m_VOASMNDProfile; this->m_VOASMNDProfile = NULL;
}


/** Initialization */
void VO_FittingASMNDProfiles::init()
{
    VO_Fitting2DSM::init();
    this->m_VOASMNDProfile      = new VO_ASMNDProfiles();
    this->m_iFittingMethod      = VO_AXM::ASM_PROFILEND;
}


/**
 * @author     JIA Pei
 * @version    2010-05-18
 * @brief      Load all AAM data from a specified folder for later fitting, to member variable m_VOASMNDProfile
 * @param      fd         Input - the folder that AAM to be loaded from
*/
void VO_FittingASMNDProfiles::VO_LoadParameters4Fitting(const string& fd)
{
    this->m_VOASMNDProfile->VO_LoadParameters4Fitting(fd);

    // VO_Fitting2DSM
    this->m_VOTemplateAlignedShape          = this->m_VOASMNDProfile->m_VOAlignedMeanShape;
    //    this->m_VOTemplateNormalizedTexture   = this->m_VOASMNDProfile->m_VONormalizedMeanTexture;
    this->m_vTriangle2D                     = this->m_VOASMNDProfile->m_vNormalizedTriangle2D;
    this->m_vShape2DInfo                    = this->m_VOASMNDProfile->m_vShape2DInfo;
    this->m_FaceParts                       = this->m_VOASMNDProfile->m_FaceParts;
    //    this->m_vPointWarpInfo                = this->m_VOASMNDProfile->m_vNormalizedPointWarpInfo;
}


/**
 * @author      JIA Pei, YAO Wei
 * @version     2010-05-20
 * @brief       Additive ASM ND Profiles Fitting, for static images, so that we record the whole fitting process
 * @param       iImg            Input - image to be fitted
 * @param       oMeshInfo       Output - the fitting process
 * @param       dim             Input - profile dimension, 1, 2, 4 or 8
 * @param       epoch           Input - the iteration epoch
 * @param       pyramidlevel    Input - pyramid level, 1, 2, 3 or 4 at most
 * @note        Refer to "AAM Revisited, page 34, figure 13", particularly, those steps.
*/
float VO_FittingASMNDProfiles::VO_ASMNDProfileFitting(  const Mat& iImg,
														vector<DrawMeshInfo>& oMeshInfo,
                                                        unsigned int epoch,
                                                        unsigned int pyramidlevel,
														bool record,
														const vector<VO_Fitting2DSM::MULTICHANNELTECH>& fitTechs)
{
	double t = (double)cvGetTickCount();
	
	//overwrites pyramid levels
    this->m_iNbOfPyramidLevels = pyramidlevel;
    this->SetProcessingImage(iImg, this->m_VOASMNDProfile);
    this->m_iIteration = 0;

	if(record)
	{
		//Save initial shape placement
		DrawMeshInfo temp(1,1,this->m_VOFittingShape,this->m_VOASMNDProfile);
		oMeshInfo.push_back(temp);
	}

    // Get m_MatModelAlignedShapeParam and m_fScale, m_vRotateAngles, m_MatCenterOfGravity
    this->m_VOASMNDProfile->VO_CalcAllParams4AnyShapeWithConstrain( this->m_VOFittingShape,
                                                                    this->m_MatModelAlignedShapeParam,
                                                                    this->m_fScale,
                                                                    this->m_vRotateAngles,
                                                                    this->m_MatCenterOfGravity);
    this->m_VOFittingShape.ConstrainShapeInImage(this->m_ImageProcessing);
	if(record)
	{
		//Save shape after shape constraints are applied
		oMeshInfo.push_back(DrawMeshInfo(1,1,this->m_VOFittingShape,this->m_VOASMNDProfile));
	}

    // Explained by YAO Wei, 2008-2-9.
    // Scale this->m_VOFittingShape, so face width is a constant StdFaceWidth.
    //this->m_fScale2 = this->m_VOASMNDProfile->m_VOReferenceShape.GetWidth() / this->m_VOFittingShape.GetWidth();
    this->m_fScale2 = this->m_VOASMNDProfile->m_VOReferenceShape.GetCentralizedShapeSize() / this->m_VOFittingShape.GetCentralizedShapeSize();
    this->m_VOFittingShape *= this->m_fScale2;

    int w = (int)(iImg.cols*this->m_fScale2);
    int h = (int)(iImg.rows*this->m_fScale2);
    Mat SearchImage = Mat(Size( w, h ), this->m_ImageProcessing.type(), this->m_ImageProcessing.channels() );

    float PyrScale = pow(2.0f, (float) (this->m_iNbOfPyramidLevels-1.0f) );
    this->m_VOFittingShape /= PyrScale;

    const int nQualifyingDisplacements = (int)(this->m_VOASMNDProfile->m_iNbOfPoints * VO_Fitting2DSM::pClose);

    // for each level in the image pyramid
    for (int iLev = this->m_iNbOfPyramidLevels-1; iLev >= 0; iLev--)
    {
        // Set image roi, instead of cvCreateImage a new image to speed up
        Mat siROI = SearchImage(Rect(0, 0, (int)(w/PyrScale), (int)(h/PyrScale) ) );
        cv::resize(this->m_ImageProcessing, siROI, siROI.size());

		if(record)
		{
			//Save results after scaled for pyramiding (if any)
			oMeshInfo.push_back(DrawMeshInfo(PyrScale,this->m_fScale2,this->m_VOFittingShape,this->m_VOASMNDProfile));
		}

        this->m_VOEstimatedShape = this->m_VOFittingShape;
		//Why does find require non-const iterators??? :(
		//static vector<VO_Fitting2DSM::MULTICHANNELTECH>::iterator anyPyramds = find(fitTechs.begin(),fitTechs.end(),VO_Fitting2DSM::HYBRIDPYRAMID);
		bool usePyramidFit = fitTechs[0] == VO_Fitting2DSM::HYBRIDPYRAMID; //anyPyramds != fitTechs.end();
		if(this->m_VOASMNDProfile->GetNbOfChannels() == 2 && usePyramidFit){
			//The pyramid fits save the shape after each fit+constraint application
			this->StagedPyramidFit(	this->m_VOEstimatedShape,
								SearchImage,
								oMeshInfo,
								iLev,
								VO_Fitting2DSM::pClose,
								epoch,
								record);
		}else
		{
			this->PyramidFit(   this->m_VOEstimatedShape,
                            SearchImage,
							oMeshInfo,
                            iLev,
                            VO_Fitting2DSM::pClose,
                            epoch,
							record,
							fitTechs);
		}
        this->m_VOFittingShape = this->m_VOEstimatedShape;

        if (iLev != 0)
        {
            PyrScale /= 2.0f;
            this->m_VOFittingShape *= 2.0f;
        }
    }

    // Explained by YAO Wei, 2008-02-09.
    // this->m_fScale2 back to original size
    this->m_VOFittingShape /= this->m_fScale2;

t = ((double)cvGetTickCount() -  t )/  (cvGetTickFrequency()*1000.);
printf("MRASM fitting time cost: %.2f millisec\n", t);

    return safeDoubleToFloat(t);
}


/**
 * @author      JIA Pei, YAO Wei
 * @version     2010-05-20
 * @brief       Additive ASM ND Profiles Fitting, for dynamic image sequence
 * @param       iImg            Input - image to be fitted
 * @param       ioShape         Input and output - the shape
 * @param       oImg            Output - the fitted image
 * @param       dim             Input - profile dimension, 1, 2, 4 or 8
 * @param       epoch           Input - the iteration epoch
 * @param       pyramidlevel    Input - pyramid level, 1, 2, 3 or 4 at most
 * @note        Refer to "AAM Revisited, page 34, figure 13", particularly, those steps.
*/
float VO_FittingASMNDProfiles::VO_ASMNDProfileFitting(  const Mat& iImg,
                                                        VO_Shape& ioShape,
                                                        Mat& oImg,
                                                        unsigned int epoch,
                                                        unsigned int pyramidlevel,
                                                        unsigned int dim)
{
    this->m_VOFittingShape.clone(ioShape);
double t = (double)cvGetTickCount();

    this->m_iNbOfPyramidLevels = pyramidlevel;
    this->SetProcessingImage(iImg, this->m_VOASMNDProfile);
    this->m_iIteration = 0;

    // Get m_MatModelAlignedShapeParam and m_fScale, m_vRotateAngles, m_MatCenterOfGravity
    this->m_VOASMNDProfile->VO_CalcAllParams4AnyShapeWithConstrain( this->m_VOFittingShape,
                                                                    this->m_MatModelAlignedShapeParam,
                                                                    this->m_fScale,
                                                                    this->m_vRotateAngles,
                                                                    this->m_MatCenterOfGravity);
    this->m_VOFittingShape.ConstrainShapeInImage(this->m_ImageProcessing);

    // Explained by YAO Wei, 2008-2-9.
    // Scale this->m_VOFittingShape, so face width is a constant StdFaceWidth.
    //this->m_fScale2 = this->m_VOASMNDProfile->m_VOReferenceShape.GetWidth() / this->m_VOFittingShape.GetWidth();
    this->m_fScale2 = this->m_VOASMNDProfile->m_VOReferenceShape.GetCentralizedShapeSize() / this->m_VOFittingShape.GetCentralizedShapeSize();
    this->m_VOFittingShape *= this->m_fScale2;

    int w = (int)(iImg.cols*this->m_fScale2);
    int h = (int)(iImg.rows*this->m_fScale2);
    Mat SearchImage = Mat(Size( w, h ), this->m_ImageProcessing.type(), this->m_ImageProcessing.channels() );

    float PyrScale = pow(2.0f, (float) (this->m_iNbOfPyramidLevels-1.0f) );
    this->m_VOFittingShape /= PyrScale;

    const int nQualifyingDisplacements = (int)(this->m_VOASMNDProfile->m_iNbOfPoints * VO_Fitting2DSM::pClose);

    // for each level in the image pyramid
    for (int iLev = this->m_iNbOfPyramidLevels-1; iLev >= 0; iLev--)
    {
        // Set image roi, instead of cvCreateImage a new image to speed up
        Mat siROI = SearchImage(Rect(0, 0, (int)(w/PyrScale), (int)(h/PyrScale) ) );
        cv::resize(this->m_ImageProcessing, siROI, siROI.size());

        this->m_VOEstimatedShape = this->m_VOFittingShape;
        this->PyramidFit(   this->m_VOEstimatedShape,
                            SearchImage,
                            iLev,
                            VO_Fitting2DSM::pClose,
                            epoch,
                            dim);
        this->m_VOFittingShape = this->m_VOEstimatedShape;

        if (iLev != 0)
        {
            PyrScale /= 2.0f;
            this->m_VOFittingShape *= 2.0f;
        }
    }

    // Explained by YAO Wei, 2008-02-09.
    // this->m_fScale2 back to original size
    this->m_VOFittingShape /= this->m_fScale2;

    ioShape.clone(this->m_VOFittingShape);
    VO_Fitting2DSM::VO_DrawMesh(ioShape, this->m_VOASMNDProfile, oImg);

t = ((double)cvGetTickCount() -  t )/  (cvGetTickFrequency()*1000.);
printf("MRASM fitting time cost: %.2f millisec\n", t);
this->m_fFittingTime = safeDoubleToFloat(t);

    return safeDoubleToFloat(t);
}


/**
 * @author      JIA Pei, YAO Wei
 * @version     2010-05-20
 * @brief       Find the best offset for one point
 * @param       iImg            Input - image to be fitted
 * @param       iShape          Input - the input shape
 * @param       iShapeInfo      Input - the shape information
 * @param       iMean           Input - mean profile
 * @param       iCovInverse     Input - covariance inverse
 * @param       ptIdx           Input - point index
 * @param       ProfileLength   Input - number of profiles per pixel, number of pixel for a single profile * dim of profiles
 * @param       offSetTolerance Input - offset tolerance, which is used to determine whether this point is converged or not
 * @param       DeltaX          Output - update in X direction
 * @param       DeltaY          Output - update in Y direction
 * @param       dim             Input - profile dim
 * @return      int             return the offset of the best fit from the profile center
 * @note        Refer to "AAM Revisited, page 34, figure 13", particularly, those steps.
*/
int VO_FittingASMNDProfiles::VO_FindBestMatchingProfile1D(  const Mat& iImg,
                                                            const Point2f& ThisPoint,
                                                            const Mat_<float>& iMean,
                                                            const Mat_<float>& iCovInverse,
                                                            const unsigned int ProfileLength,
                                                            const unsigned int offSetTolerance,
                                                            const float DeltaX,
                                                            const float DeltaY)
{
    float BestFit = FLT_MAX;
    int nBestOffset = 0;    // might be + or -
    float Fit;

    VO_Profile tempProfile;
    // Do one dim a time.
    VO_Profile::VO_Get1DProfileInMat4OneLandmark (  iImg,
                                                    ThisPoint,
                                                    tempProfile,
                                                    DeltaX,
                                                    DeltaY,
                                                    ProfileLength+2*offSetTolerance);
    VO_Profile tempSubProfile;
	VO_Profile tempSubProfile_depth;

    // Find the best in just one direction
    for (int i = -(int)offSetTolerance; i <= (int)offSetTolerance; ++i)
    {
        tempSubProfile = tempProfile.GetSubProfile(i + offSetTolerance, ProfileLength, 0);
        tempSubProfile.Normalize();

        Fit = (float) cv::Mahalanobis(tempSubProfile.m_MatProf, iMean, iCovInverse );

        // Explained by YAO Wei, 2008-2-9.
        // Test for a new best fit. We test using "<=" instead of just "<"
        // so if there is an exact match then ixBest=0 i.e. no change.
//                if((i <= 0 ) ? Fit <= BestFit:  Fit < BestFit)
        if(Fit < BestFit)
        {
            nBestOffset = i;
            BestFit = Fit;
        }
    }

    // Find the additional best in the 2nd direction

    return nBestOffset;
}

/**
 * @author     	JIA Pei, YAO Wei
 * @author		Colin Bellmore
 * @version    	2012-06-12
 * @brief      	Find the best offset for one point, in two channels
 * @param      	iImg					Input - image to be fitted
 * @param		This Point				Input - the xy location of the point
 * @param		iMean					Input - mean profile
 * @param		iCovInverse				Input - covariance inverse
 * @param		ProfileLength			Input - number of profiles per pixel, number of pixel for a single profile * dim of profiles
 * @param		offSetTolerance			Input - offset tolerance, which is used to determine whether this point is converged or not
 * @param		DeltaX					Output - update in X direction
 * @param		DeltaY					Output - update in Y direction
 * @param		dir						Input - profile direction
 * @return 		int						return the offset of the best fit from the profile center
 * @note		Refer to "AAM Revisited, page 34, figure 13", particularly, those steps.
*/
int VO_FittingASMNDProfiles::VO_FindBestMatchingProfile2D(	const Mat& iImg,
															const Point2f& ThisPoint,
															const VO_Profile iMean,
															const vector< Mat_<float> > iCovInverse,
															const unsigned int ProfileLength,
															const unsigned int offSetTolerance,
															const float DeltaX,
															const float DeltaY,
															const int dir)
{
    float BestFit = FLT_MAX;
    int nBestOffset = 0;    // might be + or -
	float Fit_final;
    float Fit_c1;
	float Fit_c2;

	VO_Profile tempProfile;
	
	// Do one dim a time, returns depth profile in second dim.
	VO_Profile::VO_Get2DProfileInMat4OneLandmark (	iImg,
													ThisPoint,
													tempProfile,
													DeltaX,
													DeltaY,
													ProfileLength+2*offSetTolerance);
    VO_Profile tempSubProfile_c1;
	VO_Profile tempSubProfile_c2;

	// Find the best in just one direction
	for (int i = -(int)offSetTolerance; i <= (int)offSetTolerance; ++i)
	{
		tempSubProfile_c1 = tempProfile.GetSubProfile(i + offSetTolerance, ProfileLength, 0);
		tempSubProfile_c2 = tempProfile.GetSubProfile(i + offSetTolerance, ProfileLength, 1);

		tempSubProfile_c1.Normalize();
		tempSubProfile_c2.Normalize();

		//dir offsets to access mean and cov of normal profiles
		Fit_c1 = (float) cv::Mahalanobis(tempSubProfile_c1.m_MatProf, iMean.Get1DimProfile(dir+0), iCovInverse[dir+0] );
		Fit_c2 = (float) cv::Mahalanobis(tempSubProfile_c2.m_MatProf, iMean.Get1DimProfile(dir+1), iCovInverse[dir+1] );

		//pick the better fit, smaller distances are better.
		if(Fit_c1 < Fit_c2){
			Fit_final = Fit_c1;
		}else{
			Fit_final = Fit_c2;
		}


		// Explained by YAO Wei, 2008-2-9.
		// Test for a new best fit. We test using "<=" instead of just "<"
		// so if there is an exact match then ixBest=0 i.e. no change.
//				if((i <= 0 ) ? Fit <= BestFit:  Fit < BestFit)
		if(Fit_final < BestFit)
		{
			nBestOffset = i;
			BestFit = Fit_final;
		}
	}

	// Find the additional best in the 2nd direction

    return nBestOffset;

}

int VO_FittingASMNDProfiles::FindBestOffset(
				   const Mat& inputMat,
				   const Mat& firstChannelImg,
				   const Mat& secondChannelImg,
				   const Mat& thirdChannelImg,
				   const Point2f& ThisPoint,
				   const vector< VO_Profile >& iMean,
				   const vector< vector< Mat_<float> > >& iCovInverse,
				   const unsigned int offSetTolerance,
				   VO_Fitting2DSM::MULTICHANNELTECH fitTech,
				   const int ptIndex,
				   const int ptDir,
				   const Point2f& dirDistPt )
{

	unsigned int ProfileLength    = iMean[0].GetProfileLength();
	switch(fitTech){
	case VO_Fitting2DSM::FIRSTCHANNELONLY:
		{
			//profiles are stored differently in two channel fitting (0=gray, 2=n.gray)
			return VO_FittingASMNDProfiles::VO_FindBestMatchingProfile1D( 
				firstChannelImg,
				ThisPoint,
				iMean[ptIndex].Get1DimProfile(ptDir),
				iCovInverse[ptIndex][ptDir],
				ProfileLength,
				offSetTolerance,
				dirDistPt.x,
				dirDistPt.y);
		}
	case VO_Fitting2DSM::SECONDCHANNELONLY:
		{
			//assumes iImg was split to become gray only
			//profiles are stored differently in two channel fitting (1=depth, 3=n.depth)
			return VO_FittingASMNDProfiles::VO_FindBestMatchingProfile1D(	
				secondChannelImg,
				ThisPoint,
				iMean[ptIndex].Get1DimProfile(ptDir+1),
				iCovInverse[ptIndex][ptDir],
				ProfileLength,
				offSetTolerance,
				dirDistPt.x,
				dirDistPt.y);
		}
	case VO_Fitting2DSM::THIRDCHANNELONLY:
		{
			//requires 3 channel profile model as well
			return VO_FittingASMNDProfiles::VO_FindBestMatchingProfile1D(	
				thirdChannelImg,
				ThisPoint,
				iMean[ptIndex].Get1DimProfile(ptDir+2),
				iCovInverse[ptIndex][ptDir],
				ProfileLength,
				offSetTolerance,
				dirDistPt.x,
				dirDistPt.y);
		}
	case VO_Fitting2DSM::FULLHYBRID:
		{
			
			//assume 2 channels in 2d profiles (gray, depth, normal gray, normal depth)
			return VO_FittingASMNDProfiles::VO_FindBestMatchingProfile2D( 
				inputMat,
				ThisPoint,
				iMean[ptIndex],
				iCovInverse[ptIndex],
				ProfileLength,
				offSetTolerance,
				dirDistPt.x,
				dirDistPt.y,
				(this->m_VOASMNDProfile->m_iNbOfChannels)*ptDir); 
			//dir is determined by half the profiles.
		}
	default:
		std::cerr << "Unsupported fitting method" << std::endl;
		exit(EXIT_FAILURE);
	}

}
/**
 * @author      YAO Wei, JIA Pei
 * @version     2010-05-20
 * @brief       Find the best offset for one point
 * @param       asmmodel        Input - the ASM model
 * @param       iImg            Input - image to be fitted
 * @param       ioShape         Input and output - the input and output shape
 * @param       iShapeInfo      Input - the shape information
 * @param       iMean           Input - mean profile
 * @param       iCovInverse     Input - covariance inverse
 * @param       Lev             Input - current pyramid level
 * @param       offSetTolerance Input - offset tolerance, which is used to determine whether this point is convergede or not
 *  Sometimes, the trained data is of 4D profiles, but the user may only use 1D to test.
 * @note        Refer to "AAM Revisited, page 34, figure 13", particularly, those steps.
*/
unsigned int VO_FittingASMNDProfiles::UpdateShape(   const VO_ASMNDProfiles* const asmmodel,
                                            const Mat& inputMat,
                                            VO_Shape& ioShape,
                                            const vector<VO_Shape2DInfo>& iShapeInfo,
                                            const vector< VO_Profile >& iMean,
                                            const vector< vector< Mat_<float> > >& iCovInverse,
                                            const unsigned int offSetTolerance,
											const vector<VO_Fitting2DSM::MULTICHANNELTECH>& fitTechs)
{
    unsigned int nGoodLandmarks = 0;
    int bestOffsetIndex[2];
    unsigned int NbOfPoints     = ioShape.GetNbOfPoints();
    unsigned int NbOfShapeDim   = ioShape.GetNbOfDim();
    
	Point2f pt;
	vector<Mat> multiChannelMat;

	unsigned int NbOfChannels = asmmodel->GetNbOfChannels();
	assert(NbOfChannels >= 1);
	if(NbOfChannels > 1){
		cv::split(inputMat,multiChannelMat);
		assert(multiChannelMat.size() >= 3);
	}else if (NbOfChannels == 1){
		multiChannelMat[0] = inputMat;
	}


	Point2f deltaPt;
	Point2f normPt;
	Point2f tangentPt;
    float sqrtsum;
	Point2f bestOffsetPt;

    for (unsigned int i = 0; i < NbOfPoints; i++)
    {
        ///Calculate profile norm direction
        /** Here, this is not compatible with 3D */
        Point2f PrevPoint = ioShape.GetA2DPoint ( iShapeInfo[i].GetFrom() );
        Point2f ThisPoint = ioShape.GetA2DPoint ( i );
        Point2f NextPoint = ioShape.GetA2DPoint ( iShapeInfo[i].GetTo() );

        // left side (connected from side)
		deltaPt = ThisPoint - PrevPoint;
		sqrtsum = sqrt ( (deltaPt.x*deltaPt.x) + (deltaPt.y*deltaPt.y) );
        if ( sqrtsum < FLT_EPSILON ) sqrtsum = 1.0f;
        deltaPt = deltaPt / sqrtsum;        // Normalize
        // Firstly, normX normY record left side norm.
        normPt.x = -deltaPt.y;
        normPt.y = deltaPt.x;

        // right side (connected to side)
		deltaPt =  NextPoint - ThisPoint;
        sqrtsum = sqrt ( (deltaPt.x*deltaPt.x) + (deltaPt.y*deltaPt.y) );
        if ( sqrtsum < FLT_EPSILON ) sqrtsum = 1.0f;
        deltaPt = deltaPt / sqrtsum;        // Normalize
        // Secondly, normX normY will average both left side and right side norm.
        normPt.x += -deltaPt.y;
        normPt.y += deltaPt.x;

        // Average left right side
		sqrtsum = sqrt ( (normPt.x*normPt.x) + (normPt.y*normPt.y) );
        if ( sqrtsum < FLT_EPSILON ) sqrtsum = 1.0f;
        normPt = normPt /  sqrtsum;                      // Final Normalize
        tangentPt.x = -normPt.y;
		tangentPt.y = normPt.x;                        // Final tangent

        /////////////////////////////////////////////////////////////////////////////
		bestOffsetIndex[0] = FindBestOffset(inputMat,multiChannelMat[0],multiChannelMat[1],multiChannelMat[2],
			ThisPoint, iMean, iCovInverse, offSetTolerance, fitTechs[i],i,0,normPt);
		//Update point from norm profiles before doing tangent profiles
		bestOffsetPt = bestOffsetIndex[0] * normPt;
		ThisPoint += bestOffsetPt;
		unsigned int chanCount = this->m_VOASMNDProfile->m_iNbOfChannels;
		unsigned int profCount = this->m_VOASMNDProfile->m_iNbOfProfileDim;
		if(profCount > 1){
			bestOffsetIndex[1] = FindBestOffset(inputMat,multiChannelMat[0],multiChannelMat[1],multiChannelMat[2],
				ThisPoint, iMean, iCovInverse, offSetTolerance, fitTechs[i],i,1,tangentPt);

			// set OutShape(iPoint) to best offset from current position
			// one dimensional profile: must move point along the whisker
			bestOffsetPt = bestOffsetIndex[1] * tangentPt;
		}
		pt = ThisPoint + bestOffsetPt;
        ioShape.SetA2DPoint(pt, i);

        if (abs(bestOffsetIndex[0]) <= 1 && abs(bestOffsetIndex[1]) <= 1)
            nGoodLandmarks++;
    }

    return nGoodLandmarks;
}


//-----------------------------------------------------------------------------
// Pyramid ASM Fitting Algorithm at certain level
//
// An iterative approach to improving the fit of the instance, this->m_VOFittingShape, to an image
// proceeds as follows:
// 1. Examine a region of the image around each point Point-ith to find the best
// nearby match for the point Point'-ith.   ---> UpdateShape
// 2. Update the parameters(s, sigma, tx, ty; b) to best fit the new found points
// X.       ---> ConformShapeToModel
// 3. Repeat until convergence.
//
// For more details, ref to [Cootes & Taylor, 2004].
//-----------------------------------------------------------------------------
/**
 * @author      YAO Wei, JIA Pei
 * @version     2010-05-20
 * @brief       Find the best offset for one point
 * @param       ioShape     Input and output - the input and output shape
 * @param       iImg        Input - image to be fitted
 * @param       oImages     Output - the output images
 * @param       iLev        Input - current pyramid level
 * @param       PClose      Input - percentage of converged points. Say, 0.9 means if 90% of the points
 *                                  are judged as converged, the iteration of this pyramid can stop
 * @param       epoch       Input - the maximum iteration times
 * @note        Refer to "AAM Revisited, page 34, figure 13", particularly, those steps.
*/
void VO_FittingASMNDProfiles::PyramidFit(   VO_Shape& ioShape,
                                            const Mat& iImg,
											vector<DrawMeshInfo>& oMeshInfo,
                                            const unsigned int iLev,
                                            const float PClose,
                                            const unsigned int epoch,
											const bool record,
											const vector<VO_Fitting2DSM::MULTICHANNELTECH>& fitTechs)
{

    float PyrScale = pow(2.0f, (float) (iLev) );

    const int nQualifyingDisplacements = (int)(this->m_VOASMNDProfile->m_iNbOfPoints * PClose);

	Fit(iImg,PyrScale,nQualifyingDisplacements,iLev,epoch,fitTechs,record,ioShape,oMeshInfo);
                                                               
}


/**
 * @author      YAO Wei, JIA Pei
 * @version     2010-05-20
 * @brief       Find the best offset for one point
 * @param       iImg        Input - image to be fitted
 * @param       ioShape     Input and output - the input and output shape
 * @param       iShapeInfo  Input - the shape information
 * @param       iLev        Input - current pyramid level
 * @param       PClose      Input - percentage of converged points. Say, 0.9 means if 90% of the points
 *                                  are judged as converged, the iteration of this pyramid can stop
 * @param       epoch       Input - the maximum iteration times
 * @param       profdim     Input - dimension used during fitting. For example, the trained data could be 4D, but the user may only use 1D
 * @note        Refer to "AAM Revisited, page 34, figure 13", particularly, those steps.
*/
void VO_FittingASMNDProfiles::PyramidFit(   VO_Shape& ioShape,
                                            const Mat& iImg,
                                            const unsigned int iLev,
                                            const float PClose,
                                            const unsigned int epoch,
                                            const unsigned int profdim)
{
	//realistically, this method can never be hit anymore

    VO_Shape tempShape = ioShape;
    int nGoodLandmarks = 0;
    float PyrScale = pow(2.0f, (float) (iLev-1.0f) );

    const int nQualifyingDisplacements = (int)(this->m_VOASMNDProfile->m_iNbOfPoints * PClose);
	vector<VO_Fitting2DSM::MULTICHANNELTECH> allFirst;
	fill(allFirst.begin(),allFirst.end(),VO_Fitting2DSM::FIRSTCHANNELONLY);

    for(unsigned int iter = 0; iter < epoch; iter++)
    {
        this->m_iIteration++;
        // estimate the best ioShape by profile matching the landmarks in this->m_VOFittingShape
        nGoodLandmarks = VO_FittingASMNDProfiles::UpdateShape(  this->m_VOASMNDProfile,
                                                                iImg,
                                                                tempShape,
                                                                this->m_vShape2DInfo,
                                                                this->m_VOASMNDProfile->m_vvMeanNormalizedProfile[iLev],
                                                                this->m_VOASMNDProfile->m_vvvCVMInverseOfSg[iLev],
                                                                3,
																allFirst);//assume single channel?

        // conform ioShape to the shape model
        this->m_VOASMNDProfile->VO_CalcAllParams4AnyShapeWithConstrain( tempShape,
                                                                        this->m_MatModelAlignedShapeParam,
                                                                        this->m_fScale,
                                                                        this->m_vRotateAngles,
                                                                        this->m_MatCenterOfGravity );
        tempShape.ConstrainShapeInImage(iImg);

        // the fitting result is good enough to stop the iteration
        if(nGoodLandmarks > nQualifyingDisplacements)
            break;
    }
    ioShape = tempShape;
}
/**
 * @author		Colin B
 * @brief		Update shape with profiles and apply shape constraints until num epochs or distance
 * @params		Far too many. :(
 */
void VO_FittingASMNDProfiles::Fit(  const Mat& iImage,
                                    const float PyrScale,
									const int nQualifyingDisplacements,
									const unsigned int iLev,
									const unsigned int epoch,
									const vector<VO_Fitting2DSM::MULTICHANNELTECH>& fitTechs,
									const bool record,
									VO_Shape& ioShape,
									vector<DrawMeshInfo>& oMeshInfo)
{
	int nGoodLandmarks = 0;
	for(unsigned int iter = 0; iter < epoch; iter++)
    {
		this->m_iIteration++;
		
        // estimate the best ioShape by profile matching the landmarks in this->m_VOFittingShape
        nGoodLandmarks = VO_FittingASMNDProfiles::UpdateShape(	this->m_VOASMNDProfile,
																iImage,
																ioShape,
																this->m_vShape2DInfo,
																this->m_VOASMNDProfile->m_vvMeanNormalizedProfile[iLev],
																this->m_VOASMNDProfile->m_vvvCVMInverseOfSg[iLev],
																3,
																fitTechs);
		if(record)
		{
			// Record after shape profiles have fit
			oMeshInfo.push_back(DrawMeshInfo(PyrScale,this->m_fScale2,ioShape,this->m_VOASMNDProfile));
		}
        // conform ioShape to the shape model
        this->m_VOASMNDProfile->VO_CalcAllParams4AnyShapeWithConstrain( ioShape,
																		this->m_MatModelAlignedShapeParam,
																		this->m_fScale,
																		this->m_vRotateAngles,
																		this->m_MatCenterOfGravity );
		ioShape.ConstrainShapeInImage(iImage);
		if(record)
		{
			// Record after shape constraint is applied and shape profiles have fit
			oMeshInfo.push_back(DrawMeshInfo(PyrScale,this->m_fScale2,ioShape,this->m_VOASMNDProfile));
		}

		// the fitting result is good enough to stop the iteration
        if(nGoodLandmarks > nQualifyingDisplacements)
            break;
    }
}

/**
 * @author     	YAO Wei, JIA Pei
 * @author		Colin B
 * @version    	2012-05-20
 * @brief      	Find the best offset for one point
 * @param      	iImg					Input - image to be fitted
 * @param      	ioShape         		Input and output - the input and output shape
 * @param		iShapeInfo				Input - the shape information
 * @param		iLev					Input - current pyramid level
 * @param		PClose					Input - percentage of converged points. Say, 0.9 means if 90% of the points
 * 												are judged as converged, the iteration of this pyramid can stop
 * @param		epoch					Input - the maximum iteration times
 * @note		Refer to "AAM Revisited, page 34, figure 13", particularly, those steps.
*/
void VO_FittingASMNDProfiles::StagedPyramidFit(	VO_Shape& ioShape,
											const Mat& iImg,
											vector<DrawMeshInfo>& oMeshInfo,
											unsigned int iLev,
											float PClose,
											unsigned int epoch,
											bool record)
{

    float PyrScale = pow(2.0f, (float) (iLev) );

    const int nQualifyingDisplacements = (int)(this->m_VOASMNDProfile->m_iNbOfPoints * PClose);
	//Add more or rearrange order here.
	vector<VO_Fitting2DSM::MULTICHANNELTECH> allFirstChannel(this->m_VOASMNDProfile->m_iNbOfPoints,VO_Fitting2DSM::FIRSTCHANNELONLY);
	vector<VO_Fitting2DSM::MULTICHANNELTECH> allSecondChannel(this->m_VOASMNDProfile->m_iNbOfPoints,VO_Fitting2DSM::SECONDCHANNELONLY);
	vector<VO_Fitting2DSM::MULTICHANNELTECH> allHybrid(this->m_VOASMNDProfile->m_iNbOfPoints,VO_Fitting2DSM::FULLHYBRID);

	//Depth fitting
	Fit(iImg,PyrScale,nQualifyingDisplacements,iLev,epoch,allSecondChannel,record,ioShape,oMeshInfo);
     
	//Color fitting
	Fit(iImg,PyrScale,nQualifyingDisplacements,iLev,epoch,allFirstChannel,record,ioShape,oMeshInfo);
     
	//2-Channel fitting
	Fit(iImg,PyrScale,nQualifyingDisplacements,iLev,epoch,allHybrid,record,ioShape,oMeshInfo);
     
}

