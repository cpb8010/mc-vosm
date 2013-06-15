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
* Create Date:      2010-03-05                                              *
* Revise Date:      2012-03-22                                              *
*****************************************************************************/

#ifndef __VO_WINDOWFUNC_H__
#define __VO_WINDOWFUNC_H__

#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>


using namespace std;
using namespace cv;


/** 
 * @author        JIA Pei
 * @brief        This is a class to generate 2D discrete window functions.
 *                 If it only contains 1 single row or col, it's a 1D window function.
 */
class VO_WindowFunc
{
friend class VO_STFT;
friend class VO_Gabor;
friend class VO_Gauss;
protected:
    /**    This is used to record window function size in 2D, could be 1D. */
    Size                m_iWindowSize;

    /**    This is used to record window function type */
    unsigned int        m_iWindowFunc;

    /** The Window Function kernel */
    Mat_<float>         m_MatWindowedKernel;

    /** Initialization */
    void                init();

public:
    enum {
        RECTANGULAR = 0,
        HAMMING = 1,
        HANN = 2,
        TUKEY = 3,
        COSINE = 4,
        LANCZOS = 5,
        BARTLETT = 6,
        TRIANGULAR = 7,
        GAUSS = 8,
        BARTLETTHANN = 9,
        BLACKMAN = 10,
        KAISER = 11,
        NUTTALL = 12,
        BLACKMANHARRIS = 13,
        BLACKMANNUTTALL = 14,
        FLATTOP = 15,
        BESSEL = 16,
        DOLPHCHEBYSHEV = 17,
        EXPONENTIAL = 18,
        RIFEVINCENT = 19,
        GABOR = 20
    };

    /** Constructor */
    VO_WindowFunc(Size windowSize, unsigned int windowFunc = RECTANGULAR);

    /** Destructor */
    ~VO_WindowFunc();

    /** Generate the Window Function */
    Mat_<float>         VO_GenerateWindowFunc(Size windowSize, unsigned int windowFunc);

    /** Display the Window Function */
    void                VO_DisplayWindowFuncKernel(const string& fn);

    /** Gets and Sets */
    Size                GetWindowSize() const { return this->m_iWindowSize;}
    Mat_<float>         GetWindowKernel() const { return this->m_MatWindowedKernel; }
};

#endif // __VO_WINDOWFUNC_H__
