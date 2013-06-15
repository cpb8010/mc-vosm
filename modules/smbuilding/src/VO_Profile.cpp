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


#include "VO_Profile.h"
#include "VO_CVCommon.h"


ostream& operator<<(ostream& os, const VO_Profile& profile)
{
    os << profile.m_MatProf;
    return os;
}


istream& operator>>(istream &is, VO_Profile& profile)
{
    is >> profile.m_MatProf;
    return is;
}


/**
 * @author     JIA Pei
 * @version    2010-02-07
 * @brief      operator= overloading, similar to copy constructor
 * @param      iProfile         The input profile
 * @return     VO_Profile&
*/
VO_Profile& VO_Profile::operator=(const VO_Profile& iProfile)
{
    this->CopyData (iProfile);
    return *this;
}


/**
 * @author     JIA Pei
 * @version    2010-02-07
 * @brief      operator= overloading, similar to copy constructor
 * @param      iProfile         The input profile
 * @return     VO_Profile&
*/
VO_Profile& VO_Profile::operator=(const Mat_<float>& iProfile)
{
    iProfile.copyTo(m_MatProf);
    return *this;
}


/**
* @author     JIA Pei
* @version    2010-02-07
* @brief      operator= overloading, similar to copy constructor
* @param      value         assign all values in VO_Profile to value
* @return     VO_Profile&
*/
VO_Profile& VO_Profile::operator=(float value)
{
    m_MatProf = value;
    return *this;
}


/**
  * @author     JIA Pei
  * @version    2010-02-07
  * @brief      operator+ overloading, shift one AAM shape by value
  * @param      value
  * @return     VO_Profile
*/
VO_Profile VO_Profile::operator+(float value)
{
    VO_Profile res(*this);
    res.m_MatProf += value;

    return res;
}


/**
 * @author      JIA Pei
 * @version     2010-02-07
 * @brief       operator+= overloading, add value to this profile
 * @param       value
 * @return      VO_Profile&
*/
VO_Profile& VO_Profile::operator+=(float value)
{
    m_MatProf += value;

    return *this;
}


/**
* @author     JIA Pei
* @version    2010-02-07
* @brief      operator+ overloading, add two profiles to one
* @param      iProfile      the added profile
* @return     VO_Profile
*/
VO_Profile VO_Profile::operator+(const VO_Profile& iProfile)
{
    VO_Profile res(*this);
    res.m_MatProf += iProfile.m_MatProf;

    return res;
}


/**
* @author     JIA Pei
* @version    2010-02-07
* @brief      operator+= overloading, add the input profile to this profile
* @param      iProfile      the added profile
* @return     VO_Profile&
*/
VO_Profile& VO_Profile::operator+=(const VO_Profile& iProfile)
{
    this->m_MatProf += iProfile.m_MatProf;

    return *this;
}


/**
* @author     JIA Pei
* @version    2010-02-07
* @brief      operator- overloading, shift one profile by -value
* @param      value
* @return     VO_Profile
*/
VO_Profile VO_Profile::operator-(float value)
{
    VO_Profile res(*this);
    res.m_MatProf -= value;
    
    return res;
}


/**
* @author     JIA Pei
* @version    2010-02-07
* @brief      operator-= overloading, subtract value from this profile
* @param      value
* @return     VO_Profile&
*/
VO_Profile& VO_Profile::operator-=(float value)
{
    this->m_MatProf -= value;

    return *this;
}


/**
  * @author     JIA Pei
  * @version    2010-02-07
  * @brief      operator- overloading, subtract one profile from another
  * @param      iProfile      the subtracted profile
  * @return     VO_Profile
*/
VO_Profile VO_Profile::operator-(const VO_Profile& iProfile)
{
    VO_Profile res(*this);
    res.m_MatProf -= iProfile.m_MatProf;

    return res;
}


/**
* @author     JIA Pei
* @version    2010-02-07
* @brief      operator-= overloading, subtract the input profile from this profile
* @param      iProfile      the subtracted profile
* @return     VO_Profile&
*/
VO_Profile& VO_Profile::operator-=(const VO_Profile& iProfile)
{
    this->m_MatProf -= iProfile.m_MatProf;

    return *this;
}


/**
* @author     JIA Pei
* @version    2010-02-07
* @brief      operator* overloading, scale a profile with input float value
* @param      value      scale size
* @return     VO_Profile
*/
VO_Profile VO_Profile::operator*(float value)
{
    VO_Profile res(*this);
    res.m_MatProf *= value;

    return res;
}


/**
* @author     JIA Pei
* @version    2010-02-07
* @brief      operator*= overloading, scale this profile with input float value
* @param      value
* @return     VO_Profile&
*/
VO_Profile& VO_Profile::operator*=(float value)
{
    this->m_MatProf *= value;

    return *this;
}


/**
* @author     JIA Pei
* @version    2010-02-07
* @brief      operator* overloading, element-wise product of two profiles
* @param      iProfile    profile to be dot producted
* @return     float       dot product
*/
VO_Profile VO_Profile::operator*(const VO_Profile& iProfile)
{
    VO_Profile res(*this);
    res.m_MatProf = res.m_MatProf.mul(iProfile.m_MatProf);

    return res;
}


/**
* @author     JIA Pei
* @version    2010-02-07
* @brief      operator* overloading, element-wise product of two profiles
* @param      iProfile    profile to be dot producted
* @return     float       dot product
*/
VO_Profile& VO_Profile::operator*=(const VO_Profile& iProfile)
{
    this->m_MatProf = this->m_MatProf.mul(iProfile.m_MatProf);

    return *this;
}


/**
* @author     JIA Pei
* @version    2010-02-07
* @brief      operator/ overloading, scale a profile
* @param      value      1.0/value = scale size
* @return     VO_Profile
*/
VO_Profile VO_Profile::operator/(float value)
{
    if( fabs(value) <= FLT_MIN )
    {
        cerr << "Divided by zero!" << endl;
        exit(EXIT_FAILURE);
    }

    VO_Profile res(*this);
    res.m_MatProf /= value;

    return res;
}


/**
* @author     JIA Pei
* @version    2010-02-07
* @brief      operator/= overloading, scale this profile with input float value
* @param      value      1.0/value = the scaled value
* @return     VO_Profile&
*/
VO_Profile& VO_Profile::operator/=(float value)
{
    if( fabs(value) <= FLT_MIN )
    {
        cerr << "Divided by zero!" << endl;
        exit(EXIT_FAILURE);
    }

    this->m_MatProf /= value;

    return *this;
}


/**
* @author     JIA Pei
* @version    2010-02-07
* @brief      operator/ overloading, scale a profile
* @param      iProfile      for element-wise division
* @return     VO_Profile
*/
VO_Profile VO_Profile::operator/(const VO_Profile& iProfile)
{
    for(int i = 0; i < iProfile.m_MatProf.rows; i++)
    {
        for(int j = 0; j < iProfile.m_MatProf.cols; j++)
        {
            if( fabs(iProfile.m_MatProf(i,j)) <= FLT_MIN )
            {
                cerr << "Divided by zero!" << endl;
                exit(EXIT_FAILURE);
            }
        }
    }

    VO_Profile res(*this);
    res.m_MatProf /= iProfile.m_MatProf;

    return res;
}


/**
* @author     JIA Pei
* @version    2010-02-07
* @brief      operator/= overloading, scale this profile with input float value
* @param      iProfile      for element-wise division
* @return     VO_Profile&
*/
VO_Profile& VO_Profile::operator/=(const VO_Profile& iProfile)
{
    for(int i = 0; i < iProfile.m_MatProf.rows; i++)
    {
        for(int j = 0; j < iProfile.m_MatProf.cols; j++)
        {
            if( fabs(iProfile.m_MatProf(i,j)) <= FLT_MIN )
            {
                cerr << "Divided by zero!" << endl;
                exit(EXIT_FAILURE);
            }
        }
    }

    this->m_MatProf /= iProfile.m_MatProf;

    return *this;
}



/**
* @author     JIA Pei
* @version    2010-02-07
* @brief      operator() overloading, obtain the profile value at row and col
* @param      row      input    - row index
* @param      col      input    - col index
* @return     float&
*/
float&  VO_Profile::operator() (unsigned row, unsigned col)
{
    if ( row >= m_MatProf.rows || col >= m_MatProf.cols )
        cerr << "Matrix subscript out of bounds" << endl;
    return m_MatProf.at<float>(row, col);
}


/**
* @author     JIA Pei
* @version    2010-02-07
* @brief      operator() overloading, obtain the profile value at row and col
* @param      row      input    - row index
* @param      col      input    - col index
* @return     float&
*/
float  VO_Profile::operator() (unsigned row, unsigned col) const
{
    if ( row >= m_MatProf.rows || col >= m_MatProf.cols )
        cerr << "Matrix subscript out of bounds" << endl;
    return m_MatProf.at<float>(row, col);
}
    

/**
* @author     JIA Pei
* @version    2010-02-07
* @brief      operator* overloading, dot product of two profiles
* @param      iProfile    profile to be dot producted
* @return     float       dot product
*/
float VO_Profile::dot(const VO_Profile& iProfile)
{
    return safeDoubleToFloat(this->m_MatProf.dot(iProfile.m_MatProf));
}


/**
 * @author     JIA Pei
 * @version    2010-02-07
 * @brief      resize the matrix
 * @param      length        Input, profile length
 * @param      dim            Input, profile dim
 * @return     void
*/
void VO_Profile::Resize(unsigned int length, unsigned int dim)
{
    this->m_MatProf.release();
    this->m_MatProf = Mat_<float>::zeros(length, dim);
}

/**
 * @author		Colin B
 * @version		2012-05-28
 * @brief		Range check for before texture calculation
 * @param		inPt	Input/Output	point to bound
 * @param		rows	Input	height of image
 * @param		cols	Input	width of image
 * @return		void	
 */
void	VO_Profile::BoundPoint(Point2f& toBound,const int height,const int width){

	if(static_cast<int>(toBound.x) < 0)
		toBound.x = 0.0f;
	else if (static_cast<int>(toBound.x+1) >= width-1)
		toBound.x -= 2*(toBound.x - (width-1)) +2;

	assert(toBound.x+1 < width);

	if(static_cast<int>(toBound.y) < 0)
		toBound.y = 0.0f;
	else if(static_cast<int>(toBound.y+1) >= height-1)
		toBound.y -= 2*(toBound.y - (height-1)) +2;

	assert(toBound.y+1 < height);
}

/**
 * @author      JIA Pei
 * @version     2010-02-07
 * @brief       Get a subprofile in 1 dimension, length*1
 * @param       start           Input       start from which index
 * @param       length          Input       how long will the subprofile be
 * @return      Mat_<float>     the subprofile
 */ 
Mat_<float> VO_Profile::GetSubProfile(int start, unsigned int length, unsigned int dimIdx) const
{
    if(start+length > this->m_MatProf.rows)
        cerr << "VO_Profile  start+length is bigger than m_MatProf.rows" << endl;

    Mat_<float> oProfile = Mat_<float>::zeros(length, 1);
    for(unsigned int i = 0; i < length; i++)
    {
        oProfile(i, 0) = this->m_MatProf(i+start, dimIdx);
    }

    return oProfile;
}


/**
 * @author      JIA Pei
 * @version     2010-02-07
 * @brief       Set 1 dimension Profile
 * @param       iOneDimProf     Input       1D column vector, length*1
 * @return      void
 */ 
void VO_Profile::Set1DimProfile(const Mat_<float>& iOneDimProf, unsigned int idx)
{
    unsigned int NbOfDim = this->m_MatProf.cols;
    unsigned int NbOfLength = this->m_MatProf.rows;
    if(idx >= NbOfDim)
    {
        cerr << "idx shouldn't be bigger than the dim of m_MatProf!" << endl;
        exit(1);
    }
    if(iOneDimProf.rows != NbOfLength)
    {
        cerr << "input profile length should be equal to the length of m_MatProf!" << endl;
        exit(1);
    }
    for(unsigned int i = 0; i < NbOfLength; i++)
    {
        this->m_MatProf(i, idx) = iOneDimProf(i, 0);
    }
}

void VO_Profile::Set1DimProfile(const VO_Profile& iOneDimProf, unsigned int idx)
{
    this->Set1DimProfile(iOneDimProf.m_MatProf, idx);
}

/**
 * @author		Colin B
 * @author     	JIA Pei
 * @version    	2012-05-17
 * @brief		Set 2dimension Profile
 * @param		iTwoDimProf 	Input		2D column vector, length*1
 * @return		void
 */ 
void VO_Profile::Set2DimProfile(const Mat_<float>& iTwoDimProf, unsigned int idx)
{
	unsigned int NbOfDim = this->m_MatProf.cols;
	unsigned int NbOfLength = this->m_MatProf.rows;
	if(idx >= NbOfDim)
	{
		cerr << "idx shouldn't be bigger than the dim of m_MatProf!" << endl;
		exit(1);
	}
	if(iTwoDimProf.rows != NbOfLength)
	{
		cerr << "input profile length should be equal to the length of m_MatProf!" << endl;
		exit(1);
	}
	for(unsigned int i = 0; i < NbOfLength; i++)
	{
		this->m_MatProf(i, idx) = iTwoDimProf(i, 0);
		this->m_MatProf(i, idx+1) = iTwoDimProf(i, 1);
	}
}

void VO_Profile::Set2DimProfile(const VO_Profile& iTwoDimProf, unsigned int idx)
{
	this->Set2DimProfile(iTwoDimProf.m_MatProf, idx);
}


/**
 * @author      JIA Pei
 * @version     2010-02-22
 * @brief       Get ND profiles for a single landmark
 * @param       iImg            Input   -- the input image
 * @param       ThisPoint       Input   -- the concerned point
 * @param       oProf           Output  -- output profile in column format
 * @param       deltaX          Input   -- deltaX in some direction
 * @param       deltaY          Input   -- deltaY in some direction
 * @param       ProfileLength   Input   -- how many elements for a single profile
 * @return      void
 * @note        Using "float* oProf" is much much faster than using "VO_Profile& oProf" or "vector<float>"
 */
void VO_Profile::VO_Get1DProfileInMat4OneLandmark ( const Mat& iImg, 
                                                    const Point2f& ThisPoint,
                                                    VO_Profile& oProf,
                                                    const float deltaX, 
                                                    const float deltaY,
                                                    const unsigned int ProfileLength)
{
//static Mat oImg(iImg);
//Point pt1, pt2;
//int rgb = 255;
//Scalar profileColor = Scalar(rgb,rgb,rgb);
//unsigned int TheColor = 0;
    unsigned int width         = iImg.cols;
    unsigned int height     = iImg.rows;
    unsigned int channels    = iImg.channels();
    
	//each image channel get a seperate profile
	oProf = Mat_<float>::zeros(ProfileLength, channels);
	
    // Emphasized by JIA Pei. k shouldn't be unsigned int in this function
    int k                     = (ProfileLength-1)/2;
    Point2f normalPoint;
    
    normalPoint.x             = ThisPoint.x + ( -k-1 ) * deltaX;
    normalPoint.y             = ThisPoint.y + ( -k-1 ) * deltaY;

//pt1 = normalPoint;
//cv::line( tempImg, pt1, pt1, colors[TheColor%8], 5, 0, 0 );
//if(ptIdx == 84 || ptIdx == 85 || ptIdx == 86 )
//{
//cv::line( oImg, pt1, pt1, colors[TheColor%8], 5, 0, 0 );
//}

    // make sure the point is within the image, otherwise, you can't extract the pixel RGB texture
    BoundPoint(normalPoint,height,width);

    if(channels == 1)
    {
        float gray_prev     = 0.0f;
        float gray_curr     = 0.0f;

        VO_TextureModel::VO_CalcSubPixelTexture ( normalPoint.x, normalPoint.y, iImg, &gray_prev );

        for (int i = -k; i <= k; ++i)
        {
//pt1 = normalPoint;
            normalPoint.x = ThisPoint.x + i * deltaX;
            normalPoint.y = ThisPoint.y + i * deltaY;
//pt2 = normalPoint;

//{
//rgb = int ( 255.0/(float)ProfileLength*(float)TheColor );
//profileColor = Scalar(rgb,rgb,rgb);
//cv::line( oImg, pt1, pt2, profileColor, 2, 0, 0 );
//++TheColor;
//}

            // make sure the point is within the image, otherwise, you can't extract the pixel RGB texture
            BoundPoint(normalPoint,height,width);

            VO_TextureModel::VO_CalcSubPixelTexture ( normalPoint.x, normalPoint.y, iImg, &gray_curr );

            oProf.m_MatProf(i+k, 0) = gray_curr - gray_prev;
            gray_prev = gray_curr;
        }
    }
	else if(channels == 2){
		
		float gray_prev 	= 0.0f;
		float gray_curr 	= 0.0f;
		float depth_prev	= 0.0f;
		float depth_curr	= 0.0f;

		//seperate pixel channels calculated
		VO_TextureModel::VO_CalcSubPixelTexture ( normalPoint.x, normalPoint.y, iImg, &gray_prev, &depth_prev );
		
		//silly loop for dealing with high profile dimensions
		for (int i = -k; i <= k; ++i)
		{
			normalPoint.x = ThisPoint.x + i * deltaX;
			normalPoint.y = ThisPoint.y + i * deltaY;

			// range check
			BoundPoint(normalPoint,height,width);

			VO_TextureModel::VO_CalcSubPixelTexture ( normalPoint.x, normalPoint.y, iImg, &gray_curr, &depth_curr );

			oProf.m_MatProf(i+k, 0) = gray_curr - gray_prev;
			oProf.m_MatProf(i+k, 1) = depth_curr - depth_prev;

			gray_prev = gray_curr;
			depth_prev = depth_curr;
		}
	}
    else if (channels == 3)
    {
        Mat grayImg;
        cv::cvtColor(iImg, grayImg, CV_BGR2GRAY);
        float gray_prev     = 0.0f;
        float gray_curr     = 0.0f;

        VO_TextureModel::VO_CalcSubPixelTexture ( normalPoint.x, normalPoint.y, grayImg, &gray_prev );

        for (int i = -k; i <= k; ++i)
        {
//pt1 = normalPoint;
            normalPoint.x = ThisPoint.x + i * deltaX;
            normalPoint.y = ThisPoint.y + i * deltaY;
//pt2 = normalPoint;

//{
//rgb = int ( 255.0/(float)ProfileLength*(float)TheColor );
//profileColor = Scalar(rgb,rgb,rgb);
//cv::line( oImg, pt1, pt2, profileColor, 2, 0, 0 );
//++TheColor;
//}

            // make sure the point is within the image, otherwise, you can't extract the pixel RGB texture
            BoundPoint(normalPoint,height,width);

            VO_TextureModel::VO_CalcSubPixelTexture ( normalPoint.x, normalPoint.y, grayImg, &gray_curr );

            oProf.m_MatProf(i+k, 0) = gray_curr - gray_prev;
            gray_prev = gray_curr;
        }
////////////////////////////////////////////////////////////////////////////////////////////
// The following is dealing with 3 channels
//        float b_prev = 0.0f, g_prev = 0.0f, r_prev = 0.0f;
//        float b_curr = 0.0f, g_curr = 0.0f, r_curr = 0.0f;
//
//        VO_TextureModel::VO_CalcSubPixelTexture ( normalPoint.x, normalPoint.y, iImg, &b_prev, & g_prev, &r_prev );
//
//        for (int i = -k; i <= k; ++i)
//        {
////pt1 = normalPoint;
//            normalPoint.x = ThisPoint.x + i * deltaX;
//            normalPoint.y = ThisPoint.y + i * deltaY;
////pt2 = normalPoint;
//
////{
////rgb = int ( 255.0/(float)ProfileLength*(float)TheColor );
////profileColor = Scalar(rgb,rgb,rgb);
////cv::line( oImg, pt1, pt2, profileColor, 2, 0, 0 );
////++TheColor;
////}
//
        //// make sure the point is within the image, otherwise, you can't extract the pixel RGB texture
        //BoundPoint(normalPoint,height,width);
//            VO_TextureModel::VO_CalcSubPixelTexture ( normalPoint.x, normalPoint.y, iImg, &b_curr, &g_curr, &r_curr );
//
//            oProf(3*(i+k)+0, 0) = b_curr - b_prev;
//            oProf(3*(i+k)+1, 0) = g_curr - g_prev;
//            oProf(3*(i+k)+2, 0) = r_curr - r_prev;
//            b_prev = b_curr;
//            g_prev = g_curr;
//            r_prev = r_curr;
//        }
////////////////////////////////////////////////////////////////////////////////////////////
    }
    else
    {
        cerr << "VO_Profile: image channels error!" << endl;
        exit(1);
    }

//imwrite("test.jpg", oImg);
}

/**
 * @author     	JIA Pei
 * @author		Colin B
 * @version    	2010-02-22
 * @brief		Get ND profiles for a single landmark
 * @param		iImg 					Input	-- the input image
 * @param		ThisPoint				Input	-- the concerned point
 * @param		oProf 					Output	-- output profile in column format
 * @param		deltaX					Input	-- deltaX in some direction
 * @param		deltaY					Input	-- deltaY in some direction
 * @param		ProfileLength	Input	-- how many elements for a single profile
 * @return		void
 * @note		Using "float* oProf" is much much faster than using "VO_Profile& oProf" or "vector<float>"
 */
void VO_Profile::VO_Get2DProfileInMat4OneLandmark (	const Mat& iImg, 
													const Point2f& ThisPoint,
													VO_Profile& oProf,
													const float deltaX, 
													const float deltaY,
													const unsigned int ProfileLength)
{
//static Mat oImg(iImg);
//Point pt1, pt2;
//int rgb = 255;
//Scalar profileColor = Scalar(rgb,rgb,rgb);
//unsigned int TheColor = 0;
	unsigned int width 		= iImg.cols;
	unsigned int height 	= iImg.rows;

	//each image channel get a seperate profile
	oProf = Mat_<float>::zeros(ProfileLength, 2);
	
	// Emphasized by JIA Pei. k shouldn't be unsigned int in this function
	int k 					= (ProfileLength-1)/2;
	Point2f normalPoint;
	
	normalPoint.x 			= ThisPoint.x + ( -k-1 ) * deltaX;
	normalPoint.y 			= ThisPoint.y + ( -k-1 ) * deltaY;

//pt1 = normalPoint;
//cv::line( tempImg, pt1, pt1, colors[TheColor%8], 5, 0, 0 );
//if(ptIdx == 84 || ptIdx == 85 || ptIdx == 86 )
//{
//cv::line( oImg, pt1, pt1, colors[TheColor%8], 5, 0, 0 );
//}

	// make sure the point is within the image, otherwise, you can't extract the pixel RGB texture
    BoundPoint(normalPoint,height,width);


	float gray_prev 	= 0.0f;
	float gray_curr 	= 0.0f;
	float depth_prev	= 0.0f;
	float depth_curr	= 0.0f;

	//seperate pixel channels calculated
	VO_TextureModel::VO_CalcSubPixelTexture ( normalPoint.x, normalPoint.y, iImg, &gray_prev, &depth_prev );
		
	//silly loop for dealing with high profile dimensions
	for (int i = -k; i <= k; ++i)
	{
		normalPoint.x = ThisPoint.x + i * deltaX;
		normalPoint.y = ThisPoint.y + i * deltaY;

		// range check and reflection padding
		BoundPoint(normalPoint,height,width);

		VO_TextureModel::VO_CalcSubPixelTexture ( normalPoint.x, normalPoint.y, iImg, &gray_curr, &depth_curr );

		oProf.m_MatProf(i+k, 0) = gray_curr - gray_prev;
		oProf.m_MatProf(i+k, 1) = depth_curr - depth_prev;

		gray_prev = gray_curr;
		depth_prev = depth_curr;
	}

//imwrite("test.jpg", oImg);
}

/**
 * @author      JIA Pei
 * @version     2010-02-22
 * @brief       Get ND profiles for a single landmark
 * @param       iImg            Input   -- the input image
 * @param       iShape          Input   -- the training shape
 * @param       iShapeInfo      Input   -- shape info
 * @param       ptIdx           Input   -- the landmark index
 * @param       oProf           Output  -- output profile
 * @param       dim             Input   -- 1D, 2D, 4D?
 * @param       ProfileLength   Input   -- how many elements for a single profile, already multiply by 3 if iImg is of 3 channels
 * @param       pDeltaX         Output  -- deltaX in normal direction
 * @param       pDeltaY         Output  -- deltaY in normal direction
 * @return      void
 * @note        Using "float* oProf" is much much faster than using "VO_Profile& oProf" or "vector<float>"
 */
void VO_Profile::VO_GetNDProfiles4OneLandmark ( const Mat& iImg,
                                                const VO_Shape& iShape,
                                                const vector<VO_Shape2DInfo>& iShapeInfo,
                                                unsigned int ptIdx,
                                                VO_Profile& oProf,
                                                unsigned int dim,
												unsigned int channels,
                                                unsigned int ProfileLength,
                                                float* pDeltaX,
                                                float* pDeltaY)
{
	//number of profile dimension might mean channels
    oProf = Mat_<float>::zeros( ProfileLength, dim );

    /** Here, this is not compatible with 3D */
    Point2f PrevPoint = iShape.GetA2DPoint ( iShapeInfo[ptIdx].GetFrom() );
    Point2f ThisPoint = iShape.GetA2DPoint ( ptIdx );
    Point2f NextPoint = iShape.GetA2DPoint ( iShapeInfo[ptIdx].GetTo() );

    float deltaX, deltaY;
    float normX, normY;
    float sqrtsum;

    // left side (connected from side)
    deltaX = ThisPoint.x - PrevPoint.x;
    deltaY = ThisPoint.y - PrevPoint.y;
    sqrtsum = sqrt ( deltaX*deltaX + deltaY*deltaY );
    if ( sqrtsum < FLT_EPSILON ) sqrtsum = 1.0f;
    deltaX /= sqrtsum; deltaY /= sqrtsum;         // Normalize
    // Firstly, normX normY record left side norm.
    normX = -deltaY;
    normY = deltaX;

    // right side (connected to side)
    deltaX = NextPoint.x - ThisPoint.x;
    deltaY = NextPoint.y - ThisPoint.y;
    sqrtsum = sqrt ( deltaX*deltaX + deltaY*deltaY );
    if ( sqrtsum < FLT_EPSILON ) sqrtsum = 1.0f;
    deltaX /= sqrtsum; deltaY /= sqrtsum;         // Normalize
    // Secondly, normX normY will average both left side and right side norm.
    normX += -deltaY;
    normY += deltaX;

    // Average left right side
    sqrtsum = sqrt ( normX*normX + normY*normY );
    if ( sqrtsum < FLT_EPSILON ) sqrtsum = 1.0f;
    normX /= sqrtsum;
    normY /= sqrtsum;                             // Final Normalize

    //////////////////////////////////////////////////////////////////////////////
    // For the 1st dimension -- ASM_PROFILE1D
    // terrific - speed up always. Explained by JIA Pei, coded by Yao Wei.
    VO_Profile tmpCol;
    switch(dim)
    {
        case 2:
        {
            float tangentX     =     -normY;
            float tangentY    =    normX;

            VO_Profile::VO_Get1DProfileInMat4OneLandmark (  iImg, 
                                                            ThisPoint,
                                                            tmpCol, 
                                                            normX, 
                                                            normY, 
                                                            ProfileLength);
			if(channels == 2){
				//sets dim 0,1
				oProf.Set2DimProfile(tmpCol, 0 );
			}else{
            oProf.Set1DimProfile(tmpCol, 0 );
			}
            VO_Profile::VO_Get1DProfileInMat4OneLandmark (  iImg, 
                                                            ThisPoint,
                                                            tmpCol, 
                                                            tangentX, 
                                                            tangentY, 
                                                            ProfileLength);
			if(channels == 2){
				//set dim 2,3
				oProf.Set2DimProfile(tmpCol, 2 );
			}else{
            oProf.Set1DimProfile(tmpCol, 1 );
        }
		}
        break;
        case 4:
        {
            float tangentX  =   -normY;
            float tangentY  =   normX;
			if(channels == 2){
				//special case
				VO_Profile::VO_Get2DProfileInMat4OneLandmark (iImg, ThisPoint, tmpCol, normX, normY, ProfileLength);
				//sets dim 0,1
				oProf.Set2DimProfile(tmpCol, 0 );
				VO_Profile::VO_Get2DProfileInMat4OneLandmark (iImg, ThisPoint, tmpCol, tangentX, tangentY, ProfileLength);
				//set dim 2,3
				oProf.Set2DimProfile(tmpCol, 2 );
			}
			else{
            float tmp45X    =   0.707106781f*normX-0.707106781f*normY;
            float tmp45Y    =   0.707106781f*normX+0.707106781f*normY;
            float tmp135X   =   -0.707106781f*normX-0.707106781f*normY;
            float tmp135Y   =   0.707106781f*normX-0.707106781f*normY;
            
            VO_Profile::VO_Get1DProfileInMat4OneLandmark (  iImg,
                                                            ThisPoint,
                                                            tmpCol,
                                                            normX,
                                                            normY,
                                                            ProfileLength);
            oProf.Set1DimProfile(tmpCol, 0 );
            VO_Profile::VO_Get1DProfileInMat4OneLandmark (  iImg,
                                                            ThisPoint,
                                                            tmpCol,
                                                            tangentX,
                                                            tangentY,
                                                            ProfileLength);
            oProf.Set1DimProfile(tmpCol, 1 );
            VO_Profile::VO_Get1DProfileInMat4OneLandmark (  iImg,
                                                            ThisPoint,
                                                            tmpCol,
                                                            tmp45X,
                                                            tmp45Y,
                                                            ProfileLength);
            oProf.Set1DimProfile(tmpCol, 2 );
            VO_Profile::VO_Get1DProfileInMat4OneLandmark (  iImg,
                                                            ThisPoint,
                                                            tmpCol,
                                                            tmp135X,
                                                            tmp135Y,
                                                            ProfileLength);
            oProf.Set1DimProfile(tmpCol, 3 );
        }
		}
        break;
        case 1:
        default:
        {
            VO_Profile::VO_Get1DProfileInMat4OneLandmark (  iImg,
                                                            ThisPoint,
                                                            tmpCol,
                                                            normX,
                                                            normY,
                                                            ProfileLength);
            oProf.Set1DimProfile(tmpCol, 0 );
        }
        break;
    }
    //////////////////////////////////////////////////////////////////////////////

    if(pDeltaX)     *pDeltaX = normX;
    if(pDeltaY)     *pDeltaY = normY;
}


/**
 * @brief   Normalization for every dim
 *          1D normalization - refer to Cootes "Statistical Models of Appearance for Computer Vision" page 38, (7.1)
 * @note    It's not a direct normalization over all elements in the matrix, it's basically column-wise normalization
*/
void VO_Profile::Normalize()
{
    for(int i = 0; i < this->m_MatProf.cols; i++)
    {
        Mat oneCol = this->m_MatProf.col(i);
        cv::normalize( oneCol, oneCol);
    }
}

