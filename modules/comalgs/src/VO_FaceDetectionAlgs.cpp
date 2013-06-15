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
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "VO_FacePart.h"
#include "VO_FaceKeyPoint.h"
#include "VO_FaceDetectionAlgs.h"


/** 
* @author   JIA Pei
* @version  2010-02-04
* @brief    Set detection configuration
* @param    strfrontalface  Input - XML for frontal face
* @param    strprofileface  Input - XML for profile face
* @param    strlefteye      Input - XML for left eye
* @param    strrighteye     Input - XML for right eye
* @param    strnose         Input - XML for nose
* @param    strmouth        Input - XML for mouth
* @param    mtd             Input - detection method
* @return   facetype        Input - face type, frontal or profile, etc.
*/
void CFaceDetectionAlgs::SetConfiguration(  const string& strfrontalface, 
                                            const string& strprofileface, 
                                            const string& strlefteye,
                                            const string& strrighteye,
                                            const string& strnose,
                                            const string& strmouth,
                                            unsigned int mtd,
                                            unsigned int facetype)
{
    this->m_iDetectionMethod    = mtd;
    this->m_iFaceType           = facetype;
    switch(this->m_iDetectionMethod)
    {
    case VO_AdditiveStrongerClassifier::BAGGING:
        {
            if(strfrontalface!="")
            {
                this->m_sFile2BLoadFrontalFace  = strfrontalface;
                this->m_rtreeClassifierProfileFace.read( this->m_sFile2BLoadFrontalFace.c_str() );
            }
            if(strprofileface!="")
            {
                this->m_sFile2BLoadProfileFace  = strprofileface;
                this->m_rtreeClassifierProfileFace.read( this->m_sFile2BLoadProfileFace.c_str() );
            }
            if(strlefteye!="")
            {
                this->m_sFile2BLoadLeftEye      = strlefteye;
                this->m_rtreeClassifierLeftEye.read( this->m_sFile2BLoadLeftEye.c_str() );
            }
            if(strrighteye!="")
            {
                this->m_sFile2BLoadRightEye     = strrighteye;
                this->m_rtreeClassifierRightEye.read( this->m_sFile2BLoadRightEye.c_str() );
            }
            if(strnose!="")
            {
                this->m_sFile2BLoadNose         = strnose;
                this->m_rtreeClassifierNose.read( this->m_sFile2BLoadNose.c_str() );
            }
            if(strmouth!="")
            {
                this->m_sFile2BLoadMouth        = strmouth;
                this->m_rtreeClassifierMouth.read( this->m_sFile2BLoadMouth.c_str() );
            }
        }
        break;
    case VO_AdditiveStrongerClassifier::BOOSTING:
    default:
        {
            if(strfrontalface!="")
            {
                this->m_sFile2BLoadFrontalFace  = strfrontalface;
                this->m_cascadeClassifierFrontalFace.load( this->m_sFile2BLoadFrontalFace );
            }
            if(strprofileface!="")
            {
                this->m_sFile2BLoadProfileFace  = strprofileface;
                this->m_cascadeClassifierProfileFace.load( this->m_sFile2BLoadProfileFace );
            }
            if(strlefteye!="")
            {
                this->m_sFile2BLoadLeftEye      = strlefteye;
                this->m_cascadeClassifierLeftEye.load( this->m_sFile2BLoadLeftEye );
            }
            if(strrighteye!="")
            {
                this->m_sFile2BLoadRightEye     = strrighteye;
                this->m_cascadeClassifierRightEye.load( this->m_sFile2BLoadRightEye );
            }
            if(strnose!="")
            {
                this->m_sFile2BLoadNose         = strnose;
                this->m_cascadeClassifierNose.load( this->m_sFile2BLoadNose );
            }
            if(strmouth!="")
            {
                this->m_sFile2BLoadMouth        = strmouth;
                this->m_cascadeClassifierMouth.load( this->m_sFile2BLoadMouth );
            }
        }
        break;
    }
}


/** 
* @author   JIA Pei
* @version  2010-02-04
* @brief    Face Detection  -- refer to CDetectionAlgs::Detection
* @param    iImg            Input - image to be searched within
* @param    confinedArea    Input - only detecting the object within this confined area
* @param    scale   Input - scalar for img scaling
* @param    sSize   Input - detected object should be bigger than sSize
* @param    bSize   Input - detected object should be smaller than bSize
* @return   double  Return - the entire detection time
*/
double CFaceDetectionAlgs::FaceDetection(   const Mat& iImg,
                                            const Rect* confinedArea,
                                            const double scale,
                                            Size sSize,
                                            Size bSize)
{
    double res = (double)getTickCount();

    switch(this->m_iDetectionMethod)
    {
    case VO_AdditiveStrongerClassifier::BAGGING:
        CDetectionAlgs::BaggingDetection(
            this->m_vDetectedFaceRects,
            this->m_rtreeClassifierFrontalFace,
            iImg,
            confinedArea,
            scale,
            sSize,
            bSize);
        break;
    case VO_AdditiveStrongerClassifier::BOOSTING:
        CDetectionAlgs::BoostingDetection(
            this->m_vDetectedFaceRects,
            this->m_cascadeClassifierFrontalFace,
            iImg,
            confinedArea,
            scale,
            sSize,
            bSize);
        break;
    }

    if ( this->m_vDetectedFaceRects.size() > 0 )
    {
        this->m_bFaceDetected = true;
        this->m_VOFaceComponents.SetObjectRect( this->m_vDetectedFaceRects[0] );
        this->m_VOFaceComponents0.SetObjectRect( this->m_vDetectedFaceRects[0] );
        this->m_CVDetectedFaceWindow2SM = 
            CFaceDetectionAlgs::VO_FaceRectFromDetection2SM(
            iImg.size(),
            this->m_VOFaceComponents0.m_rectObject );
        this->m_VOFaceComponents0.m_rectObject =
            Rect(   0,
                    0,
                    this->m_CVDetectedFaceWindow2SM.width,
                    this->m_CVDetectedFaceWindow2SM.height);
        this->m_CVDetectedFaceImagePatch2SM = iImg(this->m_CVDetectedFaceWindow2SM );
    }
    else
        this->m_bFaceDetected = false;

    res = ((double)getTickCount() - res)
        / ((double)cv::getTickFrequency()*1000.);
    return res;
}


/**
* @author   JIA Pei
* @version  2008-03-04
* @brief    Face Detection
* @param    iImg        Input - the input image, in which the face detection will be carried out
* @param    confinedArea    Input - only detecting the object within this confined area
* @param    face        Input - whether to detect face? always true!
* @param    lefteye     Input - whether to detect lefteye?
* @param    righteye    Input - whether to detect righteye?
* @param    nose        Input - whether to detect nose?
* @param    mouth       Input - whether to detect mouth?
* @param    scale       Input - always 1.0
* @param    sSize       Input - smallest face 
* @param    bSize       Input - biggest face
* @return   double      Return - the entire detection time
* @note     This function defaultly detect face
*/
double CFaceDetectionAlgs::FullFaceDetection (  const Mat& iImg,
                                                const Rect* confinedArea,
                                                bool lefteye,
                                                bool righteye,
                                                bool nose,
                                                bool mouth,
                                                const double scale,
                                                Size sSize,
                                                Size bSize )
{
    double res = (double)cvGetTickCount();

    // this is to detect the frontal face
    this->FaceDetection(iImg, confinedArea, scale, sSize, bSize);
    if ( this->m_vDetectedFaceRects.size() > 0 )
        this->m_bFaceDetected = true;
    else
        this->m_bFaceDetected = false;

    // if detected
    if(this->m_bFaceDetected)
    {
        this->VO_FaceComponentsDetection(
            this->m_CVDetectedFaceImagePatch2SM,
            this->m_iFaceType,
            lefteye,
            righteye,
            nose,
            mouth);
    }
    else
    {
        this->m_bLeftEyeDetected = false;
        this->m_bRightEyeDetected = false;
        this->m_bNoseDetected = false;
        this->m_bMouthDetected = false;
    }

    res = ((double)cvGetTickCount() - res)
        / ((double)cvGetTickFrequency()*1000.);
    return res;
}


/** 
* @author   JIA Pei
* @version  2010-02-04
* @brief    Set detection configuration
* @param    iImg        Input - input image, a face
* @param    faceOrient  Input - face orientation
* @param    lefteye     Input - whether to detect left eye
* @param    righteye    Input - whether to detect right eye
* @param    nose        Input - whether to detect nose
* @param    mouth       Input - whether to detect mouth
*/
double CFaceDetectionAlgs::VO_FaceComponentsDetection(  const Mat& iImg,
                                                        int faceOrient,
                                                        bool lefteye,
                                                        bool righteye,
                                                        bool nose,
                                                        bool mouth)
{
    double res = (double)cvGetTickCount();

    // Emphasized by JIA Pei, make sure you update m_CVDetectedFaceWindow finally!
    // set the top left half to look for left eye
    if ( lefteye )
    {
        this->m_CVLeftEyePossibleWindow = CFaceDetectionAlgs::VO_SetDetectedFacePartsPossibleWindow(iImg.cols, iImg.rows, VO_FacePart::LEFTEYE, faceOrient);
        this->m_bLeftEyeDetected = this->VO_FacePartDetection ( iImg, this->m_CVLeftEyePossibleWindow, this->m_VOFaceComponents0.m_rectLeftEye, VO_FacePart::LEFTEYE);

        if ( this->m_bLeftEyeDetected )
        {            
            this->m_VOFaceComponents.m_rectLeftEye.x = ( int ) ( this->m_VOFaceComponents0.m_rectLeftEye.x + this->m_CVLeftEyePossibleWindow.x + this->m_CVDetectedFaceWindow2SM.x );
            this->m_VOFaceComponents.m_rectLeftEye.y = ( int ) ( this->m_VOFaceComponents0.m_rectLeftEye.y + this->m_CVLeftEyePossibleWindow.y + this->m_CVDetectedFaceWindow2SM.y );
            this->m_VOFaceComponents.m_rectLeftEye.width = ( int ) ( this->m_VOFaceComponents0.m_rectLeftEye.width );
            this->m_VOFaceComponents.m_rectLeftEye.height = ( int ) ( this->m_VOFaceComponents0.m_rectLeftEye.height );
            this->m_VOFaceComponents0.m_rectLeftEye.x = ( int ) ( this->m_VOFaceComponents0.m_rectLeftEye.x + this->m_CVLeftEyePossibleWindow.x );
            this->m_VOFaceComponents0.m_rectLeftEye.y = ( int ) ( this->m_VOFaceComponents0.m_rectLeftEye.y + this->m_CVLeftEyePossibleWindow.y );

        }
        else
        {
            this->m_VOFaceComponents.m_rectLeftEye.x = ( int ) ( this->m_CVLeftEyePossibleWindow.x + this->m_CVDetectedFaceWindow2SM.x );
            this->m_VOFaceComponents.m_rectLeftEye.y = ( int ) ( this->m_CVLeftEyePossibleWindow.y + this->m_CVDetectedFaceWindow2SM.y );
            this->m_VOFaceComponents.m_rectLeftEye.width = ( int ) ( this->m_CVLeftEyePossibleWindow.width );
            this->m_VOFaceComponents.m_rectLeftEye.height = ( int ) ( this->m_CVLeftEyePossibleWindow.height);
            this->m_VOFaceComponents0.m_rectLeftEye.x = ( int ) ( this->m_CVLeftEyePossibleWindow.x );
            this->m_VOFaceComponents0.m_rectLeftEye.y = ( int ) ( this->m_CVLeftEyePossibleWindow.y );

        }
    }

    // set the top right half to look for right eye
    if ( righteye )
    {
        this->m_CVRightEyePossibleWindow = CFaceDetectionAlgs::VO_SetDetectedFacePartsPossibleWindow(iImg.cols, iImg.rows, VO_FacePart::RIGHTEYE, faceOrient);
        this->m_bRightEyeDetected = this->VO_FacePartDetection ( iImg, this->m_CVRightEyePossibleWindow, this->m_VOFaceComponents0.m_rectRightEye, VO_FacePart::RIGHTEYE );

        if ( this->m_bRightEyeDetected )
        {
            this->m_VOFaceComponents.m_rectRightEye.x = ( int ) ( this->m_VOFaceComponents0.m_rectRightEye.x + this->m_CVRightEyePossibleWindow.x + this->m_CVDetectedFaceWindow2SM.x );
            this->m_VOFaceComponents.m_rectRightEye.y = ( int ) ( this->m_VOFaceComponents0.m_rectRightEye.y + this->m_CVRightEyePossibleWindow.y + this->m_CVDetectedFaceWindow2SM.y );
            this->m_VOFaceComponents.m_rectRightEye.width = ( int ) ( this->m_VOFaceComponents0.m_rectRightEye.width );
            this->m_VOFaceComponents.m_rectRightEye.height = ( int ) ( this->m_VOFaceComponents0.m_rectRightEye.height);
            this->m_VOFaceComponents0.m_rectRightEye.x = ( int ) ( this->m_VOFaceComponents0.m_rectRightEye.x + this->m_CVRightEyePossibleWindow.x );
            this->m_VOFaceComponents0.m_rectRightEye.y = ( int ) ( this->m_VOFaceComponents0.m_rectRightEye.y + this->m_CVRightEyePossibleWindow.y );

        }
        else
        {
            this->m_VOFaceComponents.m_rectRightEye.x = ( int ) ( this->m_CVRightEyePossibleWindow.x + this->m_CVDetectedFaceWindow2SM.x );
            this->m_VOFaceComponents.m_rectRightEye.y = ( int ) ( this->m_CVRightEyePossibleWindow.y + this->m_CVDetectedFaceWindow2SM.y );
            this->m_VOFaceComponents.m_rectRightEye.width = ( int ) ( this->m_CVRightEyePossibleWindow.width );
            this->m_VOFaceComponents.m_rectRightEye.height = ( int ) ( this->m_CVRightEyePossibleWindow.height );
            this->m_VOFaceComponents0.m_rectRightEye.x = ( int ) ( this->m_CVRightEyePossibleWindow.x );
            this->m_VOFaceComponents0.m_rectRightEye.y = ( int ) ( this->m_CVRightEyePossibleWindow.y );

        }
    }

    if ( nose )
    {
        this->m_CVNosePossibleWindow = CFaceDetectionAlgs::VO_SetDetectedFacePartsPossibleWindow(iImg.cols, iImg.rows, VO_FacePart::NOSE, faceOrient);
        this->m_bNoseDetected = this->VO_FacePartDetection ( iImg, this->m_CVNosePossibleWindow, this->m_VOFaceComponents0.m_rectNose, VO_FacePart::NOSE );

        if ( this->m_bNoseDetected )
        {
            this->m_VOFaceComponents.m_rectNose.x = ( int ) ( this->m_VOFaceComponents0.m_rectNose.x + this->m_CVNosePossibleWindow.x + this->m_CVDetectedFaceWindow2SM.x );
            this->m_VOFaceComponents.m_rectNose.y = ( int ) ( this->m_VOFaceComponents0.m_rectNose.y + this->m_CVNosePossibleWindow.y + this->m_CVDetectedFaceWindow2SM.y );
            this->m_VOFaceComponents.m_rectNose.width = ( int ) ( this->m_VOFaceComponents0.m_rectNose.width );
            this->m_VOFaceComponents.m_rectNose.height = ( int ) ( this->m_VOFaceComponents0.m_rectNose.height );
            this->m_VOFaceComponents0.m_rectNose.x = ( int ) ( this->m_VOFaceComponents0.m_rectNose.x + this->m_CVNosePossibleWindow.x );
            this->m_VOFaceComponents0.m_rectNose.y = ( int ) ( this->m_VOFaceComponents0.m_rectNose.y + this->m_CVNosePossibleWindow.y );

        }
        else
        {
            this->m_VOFaceComponents.m_rectNose.x = ( int ) ( this->m_CVNosePossibleWindow.x + this->m_CVDetectedFaceWindow2SM.x );
            this->m_VOFaceComponents.m_rectNose.y = ( int ) ( this->m_CVNosePossibleWindow.y + this->m_CVDetectedFaceWindow2SM.y );
            this->m_VOFaceComponents.m_rectNose.width = ( int ) ( this->m_CVNosePossibleWindow.width );
            this->m_VOFaceComponents.m_rectNose.height = ( int ) ( this->m_CVNosePossibleWindow.height );
            this->m_VOFaceComponents0.m_rectNose.x = ( int ) ( this->m_CVNosePossibleWindow.x );
            this->m_VOFaceComponents0.m_rectNose.y = ( int ) ( this->m_CVNosePossibleWindow.y );

        }
    }

    if ( mouth )
    {
        this->m_CVMouthPossibleWindow = CFaceDetectionAlgs::VO_SetDetectedFacePartsPossibleWindow(iImg.cols, iImg.rows, VO_FacePart::LIPOUTERLINE, faceOrient);
        this->m_bMouthDetected = this->VO_FacePartDetection ( iImg, this->m_CVMouthPossibleWindow, this->m_VOFaceComponents0.m_rectMouth, VO_FacePart::LIPOUTERLINE );

        if ( this->m_bMouthDetected )
        {
            this->m_VOFaceComponents0.m_rectMouth.x += static_cast<int>(this->m_VOFaceComponents0.m_rectMouth.width*0.1);
            // ensure this->m_VOFaceComponents.m_rectMouth.x + this->m_VOFaceComponents.m_rectMouth.width < iImg.cols-1
            while (this->m_VOFaceComponents0.m_rectMouth.x + this->m_VOFaceComponents0.m_rectMouth.width >= iImg.cols-1)
            {
                this->m_VOFaceComponents0.m_rectMouth.x--;
            }

            this->m_VOFaceComponents.m_rectMouth.x = ( int ) ( this->m_VOFaceComponents0.m_rectMouth.x + this->m_CVMouthPossibleWindow.x + this->m_CVDetectedFaceWindow2SM.x );
            this->m_VOFaceComponents.m_rectMouth.y = ( int ) ( this->m_VOFaceComponents0.m_rectMouth.y + this->m_CVMouthPossibleWindow.y + this->m_CVDetectedFaceWindow2SM.y );
            this->m_VOFaceComponents.m_rectMouth.width = ( int ) ( this->m_VOFaceComponents0.m_rectMouth.width );
            this->m_VOFaceComponents.m_rectMouth.height = ( int ) ( this->m_VOFaceComponents0.m_rectMouth.height);
            this->m_VOFaceComponents0.m_rectMouth.x = ( int ) ( this->m_VOFaceComponents0.m_rectMouth.x + this->m_CVMouthPossibleWindow.x );
            this->m_VOFaceComponents0.m_rectMouth.y = ( int ) ( this->m_VOFaceComponents0.m_rectMouth.y + this->m_CVMouthPossibleWindow.y );

            // For mouth, we need a modification due to the reason that the mouth is not able to be well detected.
            // We need small adjustment.
            this->m_VOFaceComponents0.m_rectMouth.y -= static_cast<int>(this->m_VOFaceComponents0.m_rectMouth.height*0.2);
            this->m_VOFaceComponents.m_rectMouth.y -= static_cast<int>(this->m_VOFaceComponents.m_rectMouth.height*0.2);
            // ensure this->m_VOFaceComponents.m_rectMouth.y > 0
            while (this->m_VOFaceComponents0.m_rectMouth.y <= 0)
            {
                this->m_VOFaceComponents0.m_rectMouth.y++;
            }
            while (this->m_VOFaceComponents.m_rectMouth.y <= 0)
            {
                this->m_VOFaceComponents.m_rectMouth.y++;
            }
        }
        else
        {
            this->m_VOFaceComponents.m_rectMouth.x = ( int ) ( this->m_CVMouthPossibleWindow.x + this->m_CVDetectedFaceWindow2SM.x );
            this->m_VOFaceComponents.m_rectMouth.y = ( int ) ( this->m_CVMouthPossibleWindow.y + this->m_CVDetectedFaceWindow2SM.y );
            this->m_VOFaceComponents.m_rectMouth.width = ( int ) ( this->m_CVMouthPossibleWindow.width );
            this->m_VOFaceComponents.m_rectMouth.height = ( int ) ( this->m_CVMouthPossibleWindow.height );
            this->m_VOFaceComponents0.m_rectMouth.x = ( int ) ( this->m_CVMouthPossibleWindow.x );
            this->m_VOFaceComponents0.m_rectMouth.y = ( int ) ( this->m_CVMouthPossibleWindow.y );

        }
    }

    // Default logical relationship among all the above 4 face parts
    // According to my observation, the mouth some times goes wrong.
    // the nose x direction goes wrong some times, but y direction keeps correct
    if ( this->m_bNoseDetected && this->m_bMouthDetected ) // mouth must be lower than nose
    {
        if ( this->m_VOFaceComponents.m_rectMouth.y+1.0f*this->m_VOFaceComponents.m_rectMouth.height/2.0f <= ( this->m_VOFaceComponents.m_rectNose.y + this->m_VOFaceComponents.m_rectNose.height ) )
            this->m_VOFaceComponents.m_rectMouth.y += this->m_VOFaceComponents.m_rectMouth.height;
        if ( this->m_VOFaceComponents0.m_rectMouth.y+1.0f*this->m_VOFaceComponents0.m_rectMouth.height/2.0f <= ( this->m_VOFaceComponents0.m_rectNose.y + this->m_VOFaceComponents0.m_rectNose.height ) )
            this->m_VOFaceComponents0.m_rectMouth.y += this->m_VOFaceComponents0.m_rectMouth.height;
    }

    res = ((double)cvGetTickCount() - res)
        / ((double)cvGetTickFrequency()*1000.);
    return res;
}


/**
* @author   JIA Pei
* @version  2010-02-04
* @brief    Face Component Detection
* @param    iImg        Input   - the input image, in which the face detection will be carried out
*                       - Here, make sure iImg is gray level image
* @param    iWindow     Input - Possible input face part window, in which this face part is in
* @param    oWindow     Output - Output detected face part window
* @param    facepart    Input - Which face part is it to be detected in. 
                        VO_FacePart::LEFTEYE,
                        VO_FacePart::RIGHTEYE,
                        VO_FacePart::NOSE, MOUTH
* @return   bool        Return - Whether the face component has been detected or not
*/
bool CFaceDetectionAlgs::VO_FacePartDetection ( const Mat& iImg,
                                                const Rect& iWindow,
                                                Rect& oWindow,
                                                unsigned int facepart)
{
    bool result = false;
    Mat smallImgROI = iImg(iWindow);

    vector<Rect> detectedfp;
    switch ( facepart )
    {
    case VO_FacePart::LEFTEYE:
        {
            switch(this->m_iDetectionMethod)
            {
            case VO_AdditiveStrongerClassifier::BAGGING:
                CDetectionAlgs::BaggingDetection(
                    detectedfp,
                    this->m_rtreeClassifierLeftEye,
                    smallImgROI,
                    0,
                    1.0,
                    Size(iImg.cols/4, iImg.cols/8),
                    Size(iImg.cols, iImg.cols*2/3) );
                //  Size(18, 12),
                //  Size(54, 36) );
                break;
            case VO_AdditiveStrongerClassifier::BOOSTING:
                CDetectionAlgs::BoostingDetection(
                    detectedfp,
                    this->m_cascadeClassifierLeftEye,
                    smallImgROI,
                    0,
                    1.0,
                    Size(iImg.cols/4, iImg.cols/8),
                    Size(iImg.cols, iImg.cols*2/3) );
                //  Size(18, 12),
                //  Size(54, 36) );
                break;
            }
        }
        break;
    case VO_FacePart::RIGHTEYE:
        {
            switch(this->m_iDetectionMethod)
            {
            case VO_AdditiveStrongerClassifier::BAGGING:
                CDetectionAlgs::BaggingDetection(
                    detectedfp,
                    this->m_rtreeClassifierRightEye,
                    smallImgROI,
                    0,
                    1.0,
                    Size(iImg.cols/4, iImg.cols/8),
                    Size(iImg.cols, iImg.cols*2/3) );
                //  Size(18, 12),
                //  Size(54, 36) );
                break;
            case VO_AdditiveStrongerClassifier::BOOSTING:
                CDetectionAlgs::BoostingDetection(
                    detectedfp,
                    this->m_cascadeClassifierRightEye,
                    smallImgROI,
                    0,
                    1.0,
                    Size(iImg.cols/4, iImg.cols/8),
                    Size(iImg.cols, iImg.cols*2/3) );
                //  Size(18, 12),
                //  Size(54, 36) );
                break;
            }
        }
        break;
    case VO_FacePart::NOSE:
        {
            switch(this->m_iDetectionMethod)
            {
            case VO_AdditiveStrongerClassifier::BAGGING:
                CDetectionAlgs::BaggingDetection(
                    detectedfp,
                    this->m_rtreeClassifierNose,
                    smallImgROI,
                    0,
                    1.0,
                    Size(iImg.cols/6, iImg.rows/6),
                    Size(iImg.cols, iImg.rows) );
                //  Size(18, 15),
                //  Size(54, 45) );
                break;
            case VO_AdditiveStrongerClassifier::BOOSTING:
                CDetectionAlgs::BoostingDetection(
                    detectedfp,
                    this->m_cascadeClassifierNose,
                    smallImgROI,
                    0,
                    1.0,
                    Size(iImg.cols/6, iImg.rows/6),
                    Size(iImg.cols, iImg.rows) );
                //  Size(18, 15),
                //  Size(54, 45) );
                break;
            }
        }
        break;
    case VO_FacePart::LIPOUTERLINE:
        {
            switch(this->m_iDetectionMethod)
            {
            case VO_AdditiveStrongerClassifier::BAGGING:
                CDetectionAlgs::BaggingDetection(
                    detectedfp,
                    this->m_rtreeClassifierMouth,
                    smallImgROI,
                    0,
                    1.0,
                    Size(iImg.cols/6, iImg.rows/6),
                    Size(iImg.cols, iImg.rows) );
                //  Size(25, 15),
                //  Size(75, 45) );
                break;
            case VO_AdditiveStrongerClassifier::BOOSTING:
                CDetectionAlgs::BoostingDetection(
                    detectedfp,
                    this->m_cascadeClassifierMouth,
                    smallImgROI,
                    0,
                    1.0,
                    Size(iImg.cols/6, iImg.rows/6),
                    Size(iImg.cols, iImg.rows) );
                //  Size(25, 15),
                //  Size(75, 45) );
                break;
            }
        }
        break;
    }

    // Within one specific face, there is only 1 face component.
    // For instance, 1 lefteye, 1 righteye, 1 mouth, 1 nose.
    if ( detectedfp.size() >= 1 )
    {
        oWindow = detectedfp[0];
        result  = true;
    }
    else
    {
        oWindow = iWindow;
        result  = false;
    }

    return result;
}


/**
* @brief    detect face directions
* @param    iImg        -- must be detected face image patch
* @param    faceOrient  -- frontal
*/
int CFaceDetectionAlgs::VO_DetectFaceDirection(int faceOrient)
{
    if( this->m_bFaceDetected &&
        this->m_bLeftEyeDetected &&
        this->m_bRightEyeDetected &&
        this->m_bNoseDetected)
    {
        this->m_CVNoseCentralArea =
            CFaceDetectionAlgs::VO_SetDetectedFacePartsPossibleWindow(
            this->m_CVDetectedFaceImagePatch2SM.cols,
            this->m_CVDetectedFaceImagePatch2SM.rows,
            VO_FacePart::NOSECENTRALAREA,
            faceOrient);
        return CFaceDetectionAlgs::VO_DetectFaceDirection(
            this->m_VOFaceComponents0,
            this->m_CVNosePossibleWindow,
            this->m_CVNoseCentralArea);
    }
    else
        return CFaceDetectionAlgs::UNDETECTED;
}


int CFaceDetectionAlgs::VO_DetectFaceDirection(
                            const VO_FaceCompPos& facecomponents,
                            const Rect& possiblenose,
                            const Rect& nosecentralarea)
{
    int dir = CFaceDetectionAlgs::DIR_FRONTAL;

    int leftright = 0;
    int updown = 0;

    vector<Point2f> centers = facecomponents.VO_CalcFaceComponentsCenters();
    float middleX = (centers[1].x + centers[2].x)/2.0f;
    Point2f possiblenoseCenter;
    possiblenoseCenter.x = possiblenose.x + possiblenose.width/2.0f;
    possiblenoseCenter.y = possiblenose.y + possiblenose.height/2.0f;

    // centers[3] is the nose center
    if(centers[3].x < nosecentralarea.x && centers[3].x < middleX)
        leftright = -1; // from the user to the laptop screen, left; but actually right
    if(centers[3].x > nosecentralarea.x+nosecentralarea.width && centers[3].x > middleX)
        leftright = 1;  // from the user to the laptop screen, right; but actually left
    if(centers[3].y < nosecentralarea.y && centers[3].y < possiblenoseCenter.y)
        updown = -1;    // up
    if(centers[3].y > nosecentralarea.y+nosecentralarea.height && centers[3].y > possiblenoseCenter.y)
        updown = 1;     // down

    if( leftright == 0 && updown == 1)      dir = DIR_DOWNFRONTAL;
    if( leftright == 1 && updown == 1)      dir = DIR_DOWNLEFT;
    if( leftright == -1 && updown == 1)     dir = DIR_DOWNRIGHT;
    if( leftright == 0 && updown == -1)     dir = DIR_UPFRONTAL;
    if( leftright == 1 && updown == -1)     dir = DIR_UPLEFT;
    if( leftright == -1 && updown == -1)    dir = DIR_UPRIGHT;
    if( leftright == 1 && updown == 0)      dir = DIR_LEFT;
    if( leftright == -1 && updown == 0)     dir = DIR_RIGHT;
    if( leftright == 0 && updown == 0)      dir = DIR_FRONTAL;

    return dir;
}


/**
* @author   JIA Pei
* @version  2008-03-04
* @brief    Draw face detection parts
* @param    ioImg       Input & Output - the input image,
*                       in which the face detection will be carried out
* @param    face        Input - whether to detect face? always true!
* @param    lefteye     Input - whether to detect lefteye?
* @param    righteye    Input - whether to detect righteye?
* @param    nose        Input - whether to detect nose?
* @param    mouth       Input - whether to detect mouth?
* @param    color       Input - line color
*/
void CFaceDetectionAlgs::VO_DrawDetection ( Mat& ioImg, 
                                            bool face,  
                                            bool lefteye, 
                                            bool righteye, 
                                            bool nose, 
                                            bool mouth,
                                            Scalar color)
{
    Point pt1, pt2, pt;

    // Face
    if ( face )
    {
        //  pt1.x = this->m_VOFaceComponents.m_rectObject.x;
        //  pt1.y = this->m_VOFaceComponents.m_rectObject.y;
        //  pt2.x = this->m_VOFaceComponents.m_rectObject.x + this->m_VOFaceComponents.m_rectObject.width;
        //  pt2.y = this->m_VOFaceComponents.m_rectObject.y + this->m_VOFaceComponents.m_rectObject.height;
        //  cv::rectangle( ioImg, pt1, pt2, color, 2, 8, 0 );

        pt1.x = this->m_CVDetectedFaceWindow2SM.x;
        pt1.y = this->m_CVDetectedFaceWindow2SM.y;
        pt2.x = this->m_CVDetectedFaceWindow2SM.x + this->m_CVDetectedFaceWindow2SM.width;
        pt2.y = this->m_CVDetectedFaceWindow2SM.y + this->m_CVDetectedFaceWindow2SM.height;
        cv::rectangle( ioImg, pt1, pt2, color, 2, 8, 0 );
    }

    // Left eye
    if ( lefteye )
    {
        pt1.x = this->m_VOFaceComponents.m_rectLeftEye.x;
        pt1.y = this->m_VOFaceComponents.m_rectLeftEye.y;
        pt2.x = this->m_VOFaceComponents.m_rectLeftEye.x + this->m_VOFaceComponents.m_rectLeftEye.width;
        pt2.y = this->m_VOFaceComponents.m_rectLeftEye.y + this->m_VOFaceComponents.m_rectLeftEye.height;
        cv::rectangle ( ioImg, pt1, pt2, colors[1], 1, 8, 0 );
    }

    // Right eye
    if ( righteye )
    {
        pt1.x = this->m_VOFaceComponents.m_rectRightEye.x;
        pt1.y = this->m_VOFaceComponents.m_rectRightEye.y;
        pt2.x = this->m_VOFaceComponents.m_rectRightEye.x + this->m_VOFaceComponents.m_rectRightEye.width;
        pt2.y = this->m_VOFaceComponents.m_rectRightEye.y + this->m_VOFaceComponents.m_rectRightEye.height;
        cv::rectangle ( ioImg, pt1, pt2, colors[2], 1, 8, 0 );
    }

    // Nose
    if ( nose )
    {
        pt1.x = this->m_VOFaceComponents.m_rectNose.x;
        pt1.y = this->m_VOFaceComponents.m_rectNose.y;
        pt2.x = this->m_VOFaceComponents.m_rectNose.x + this->m_VOFaceComponents.m_rectNose.width;
        pt2.y = this->m_VOFaceComponents.m_rectNose.y + this->m_VOFaceComponents.m_rectNose.height;
        cv::rectangle ( ioImg, pt1, pt2, colors[3], 1, 8, 0 );
    }

    // Mouth
    if ( mouth )
    {
        pt1.x = this->m_VOFaceComponents.m_rectMouth.x;
        pt1.y = this->m_VOFaceComponents.m_rectMouth.y;
        pt2.x = this->m_VOFaceComponents.m_rectMouth.x + this->m_VOFaceComponents.m_rectMouth.width;
        pt2.y = this->m_VOFaceComponents.m_rectMouth.y + this->m_VOFaceComponents.m_rectMouth.height;
        cv::rectangle ( ioImg, pt1, pt2, colors[4], 1, 8, 0 );
    }
}


/**
* @param    ptType    point type, which point is it?
* @return   Point2f    the point coordinates
*/
void CFaceDetectionAlgs::CalcFaceKeyPoints()
{
    if(this->m_bFaceDetected)
    {
        this->m_CVDetectedCenterOfGravity.x = this->m_VOFaceComponents.m_rectObject.x + this->m_VOFaceComponents.m_rectObject.width/2.0f;
        this->m_CVDetectedCenterOfGravity.y = this->m_VOFaceComponents.m_rectObject.y + this->m_VOFaceComponents.m_rectObject.height/2.0f;

        //  No matter the face components have been detected or not, m_VOFaceComponents always have some values !
        //  if(this->m_bLeftEyeDetected)
        {
            this->m_CVDetectedLeftEyeCenter.x = this->m_VOFaceComponents.m_rectLeftEye.x + this->m_VOFaceComponents.m_rectLeftEye.width/2.0f;
            this->m_CVDetectedLeftEyeCenter.y = this->m_VOFaceComponents.m_rectLeftEye.y + this->m_VOFaceComponents.m_rectLeftEye.height/2.0f;

            //this->m_CVDetectedLeftEyeLeftCorner
            //this->m_CVDetectedLeftEyeRightCorner
        }

        //  if(this->m_bRightEyeDetected)
        {
            this->m_CVDetectedRightEyeCenter.x = this->m_VOFaceComponents.m_rectRightEye.x + this->m_VOFaceComponents.m_rectRightEye.width/2.0f;
            this->m_CVDetectedRightEyeCenter.y = this->m_VOFaceComponents.m_rectRightEye.y + this->m_VOFaceComponents.m_rectRightEye.height/2.0f;

            //this->m_CVDetectedLeftEyeLeftCorner
            //this->m_CVDetectedLeftEyeRightCorner
        }

        //  if(this->m_bNoseDetected)
        {
            this->m_CVDetectedNoseCenter.x = this->m_VOFaceComponents.m_rectNose.x + this->m_VOFaceComponents.m_rectNose.width/2.0f;
            this->m_CVDetectedNoseCenter.y = this->m_VOFaceComponents.m_rectNose.y + this->m_VOFaceComponents.m_rectNose.height/2.0f;

            //this->m_CVDetectedLeftEyeLeftCorner
            //this->m_CVDetectedLeftEyeRightCorner
        }

        //  if(this->m_bMouthDetected)
        {
            this->m_CVDetectedMouthCenter.x = this->m_VOFaceComponents.m_rectMouth.x + this->m_VOFaceComponents.m_rectMouth.width/2.0f;
            this->m_CVDetectedMouthCenter.y = this->m_VOFaceComponents.m_rectMouth.y + this->m_VOFaceComponents.m_rectMouth.height/2.0f;

            //this->m_CVDetectedLeftEyeLeftCorner
            //this->m_CVDetectedLeftEyeRightCorner
        }
    }
}


/**
* @param    ptType    point type, which point is it?
* @return   Point2f    the point coordinates
*/
Point2f CFaceDetectionAlgs::GetDetectedFaceKeyPoint(unsigned int ptType) const
{
    switch(ptType)
    {
    case VO_KeyPoint::CENTEROFGRAVITY:
        return    this->m_CVDetectedCenterOfGravity;
    case VO_KeyPoint::LEFTEYELEFTCORNER:
        return    this->m_CVDetectedLeftEyeLeftCorner;
    case VO_KeyPoint::LEFTEYERIGHTCORNER:
        return    this->m_CVDetectedLeftEyeRightCorner;
    case VO_KeyPoint::LEFTEYECENTER:
        return     this->m_CVDetectedLeftEyeCenter;
    case VO_KeyPoint::RIGHTEYELEFTCORNER:
        return     this->m_CVDetectedRightEyeLeftCorner;
    case VO_KeyPoint::RIGHTEYERIGHTCORNER:
        return     this->m_CVDetectedRightEyeRightCorner;
    case VO_KeyPoint::RIGHTEYECENTER:
        return     this->m_CVDetectedRightEyeCenter;
    case VO_KeyPoint::NOSETIPKEY:
        return    this->m_CVDetectedNoseTip;
    case VO_KeyPoint::NOSTRILLEFT:
        return     this->m_CVDetectedNostrilLeft;
    case VO_KeyPoint::NOSTRILRIGHT:
        return     this->m_CVDetectedNostrilRight;
    case VO_KeyPoint::NOSECENTER:
        return     this->m_CVDetectedNoseCenter;
    case VO_KeyPoint::MOUTHLEFTCORNER:
        return     this->m_CVDetectedMouthLeftCorner;
    case VO_KeyPoint::MOUTHRIGHTCORNER:
        return     this->m_CVDetectedMouthRightCorner;
    case VO_KeyPoint::MOUTHCENTER:
        return     this->m_CVDetectedMouthCenter;
    }
	throw; //unreachable code.
	return Point2f();
}


/**
* @author   Yao Wei
* @brief    Basically, this function turns the detected face square to a
*           rectangle of size 1*1.1
* @param    imgSize     image size
* @param    iFaceRect   the detected face rectangle
* @return   Rect        the adjusted face rectangle
*/
Rect CFaceDetectionAlgs::VO_FaceRectFromDetection2SM (const Size& imgSize,
                                                    const Rect& iFaceRect )
{
    Rect res;

    // Note: copied from aamlibrary and asmlibrary
    static const float CONF_VjHeightShift = 0.15f;      // shift height down by 15%
    static const float CONF_VjShrink      = 0.80f;      // shrink size of VJ box by 20%

    float xMin      = static_cast<float>(iFaceRect.x);
    float yMin      = static_cast<float>(iFaceRect.y);
    float xMax      = static_cast<float>(iFaceRect.x + iFaceRect.width);
    float yMax      = static_cast<float>(iFaceRect.y + iFaceRect.height);

    float NewWidth  = CONF_VjShrink * static_cast<float>(iFaceRect.width);
    float NewHeight = CONF_VjShrink * static_cast<float>(iFaceRect.height);

    float yMean     = ( yMin + yMax ) / 2;
    yMean           += CONF_VjHeightShift * NewHeight;  // move face down
    float xMean     = ( xMin + xMax ) / 2;

    res.x           = xMean - 0.5f * NewWidth > 0 ?
                        (int)(xMean - 0.5f * NewWidth) : 0;
    res.y           = yMean - 0.5f * NewHeight > 0 ?
                        (int)(yMean - 0.5f * NewHeight) : 0;
    if(NewWidth + res.x < imgSize.width)
        res.width   = static_cast<int>(NewWidth);
    else
        res.width   = imgSize.width - res.x;
    if(NewHeight*1.1f + res.y < imgSize.height)
        res.height  = static_cast<int>(NewHeight*1.1f);
    else
        res.height  = imgSize.height - res.y;
    //res.width = NewWidth < imgSize.width ? (int)NewWidth : imgSize.width;
    //res.height = (NewHeight*1.1f) < imgSize.height ? (int)(NewHeight*1.1f) : imgSize.height;

    return res;
}


/**
* @author   Pei JIA
* @param    imgWidth    image size
* @param    imgHeight   the detected face rectangle
* @param    facepart    which face part is it?
* @param    dir         face directions
* @return   Rect        the possible rectangle for this specific face rectangle
*/
Rect CFaceDetectionAlgs::VO_SetDetectedFacePartsPossibleWindow(
                                                    int imgWidth,
                                                    int imgHeight,
                                                    unsigned int facepart,
                                                    unsigned int dir)
{
    Rect rect;
    switch(dir)
    {
    case LEFTPROFILE:
        break;
    case RIGHTPROFILE:
        break;
    case FRONTAL:
    default:
        {
            switch(facepart)
            {
            case VO_FacePart::LEFTEYE:
                rect    = Rect ( ( int ) ( imgWidth/20.0f ), ( int ) ( imgHeight/10.0f ), ( int ) ( 9.0f*imgWidth/20.0f ), ( int ) ( 3*imgHeight/10.0f ) );
                break;
            case VO_FacePart::RIGHTEYE:
                rect    = Rect ( ( int ) ( 1.0f*imgWidth/2.0f ), ( int ) ( imgHeight/10.0f ), ( int ) ( 9.0f*imgWidth/20.0f ), ( int ) ( 3*imgHeight/10.0f ) );
                break;
            case VO_FacePart::NOSE:
                rect    = Rect ( ( int ) ( imgWidth/4.0f ), ( int ) ( imgHeight/5.0f ), ( int ) ( imgWidth/2.0f ), ( int ) ( 3.0*imgHeight/5.0f ) );
                break;
            case VO_FacePart::LIPOUTERLINE:
                rect    = Rect ( ( int ) ( imgWidth/5.0f ), ( int ) ( 1.0f*imgHeight/2.0f ), ( int ) ( 3.0f*imgWidth/5.0f ), ( int ) ( 1.0f*imgHeight/2.0f ) );
                break;
            case VO_FacePart::NOSECENTRALAREA:
                rect    = Rect ( ( int ) ( 0.48*imgWidth ), ( int ) ( 0.45*imgHeight ), ( int ) ( 0.05*imgWidth ), ( int ) ( 0.05*imgHeight ) );
                break;
            default:
                break;
            }
        }
        break;
    }

    return rect;
}

