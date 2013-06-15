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
* Create Date:      2010-03-08                                              *
* Revise Date:      2012-03-22                                              *
*****************************************************************************/


#ifndef _VO_HAARFEATURES_H_
#define _VO_HAARFEATURES_H_

#include "VO_Features.h"

#define CC_HAAR                    "HAAR"
#define CC_MODE                    "mode"
#define CC_MODE_BASIC              "BASIC"
#define CC_MODE_CORE               "CORE"
#define CC_MODE_ALL                "ALL"
//#define CC_RECTS                   "rects"
#define CC_TILTED                  "tilted"
#define CV_HAAR_FEATURE_MAX         3



/** 
 * @author        JIA Pei
 * @brief        Haar wavelet features.
 */
class VO_HaarFeatures : public VO_Features
{
protected:
    /** which Haar mode is to be used? BASIC, CORE or ALL? */
    unsigned int                m_iMode;

    class Feature
    {
    public:
        Feature();
        Feature( int offset, bool _tilted,
                int x0, int y0, int w0, int h0, float wt0,
                int x1, int y1, int w1, int h1, float wt1,
                int x2 = 0, int y2 = 0, int w2 = 0, int h2 = 0, float wt2 = 0.0F ); 
        float   calc( const Mat &sum, const Mat &tilted) const;
        void    write( FileStorage &fs ) const;

        bool    tilted;
        struct
        {
            Rect r;
            float weight;
        } rect[CV_HAAR_FEATURE_MAX];

        struct                      
        {
            int p0, p1, p2, p3;
        } fastRect[CV_HAAR_FEATURE_MAX];
    }; 

    vector<Feature>             m_vAllFeatures;

    /** Initialization */
    void                        init();

public:
    /* 0 - BASIC = Viola
    *  1 - CORE  = All upright
    *  2 - ALL   = All features */
    enum { BASIC = 0, CORE = 1, ALL = 2 };

    /** default constructor */
    VO_HaarFeatures ()          {this->m_iFeatureType = HAAR;}

    /** destructor */
    virtual ~VO_HaarFeatures () {this->m_vAllFeatures.clear();}

    /** Generate all features with a specific mode */
    virtual void                VO_GenerateAllFeatureInfo(const Size& size, unsigned int generatingMode = 0);
    virtual void                VO_GenerateAllFeatures(const Mat& iImg, Point pt = Point(0,0));

    /** Read and write */
    virtual void                ReadFeatures( const FileStorage& fs, Mat_<float>& featureMap );
    virtual void                WriteFeatures( FileStorage& fs, const Mat_<float>& featureMap ) const;
};


inline float VO_HaarFeatures::Feature::calc( const Mat &_sum, const Mat &_tilted) const
{
    const int* img = tilted ? _tilted.ptr<int>(0) : _sum.ptr<int>(0);
    float ret = rect[0].weight * (img[fastRect[0].p0] - img[fastRect[0].p1] - img[fastRect[0].p2] + img[fastRect[0].p3] ) +
                rect[1].weight * (img[fastRect[1].p0] - img[fastRect[1].p1] - img[fastRect[1].p2] + img[fastRect[1].p3] );
    if( rect[2].weight != 0.0f )
        ret += rect[2].weight * (img[fastRect[2].p0] - img[fastRect[2].p1] - img[fastRect[2].p2] + img[fastRect[2].p3] );
    return ret;
}

#endif        // _VO_HAARFEATURES_H_
