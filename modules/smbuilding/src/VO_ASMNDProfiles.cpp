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

#include <fstream>
#include <sstream>
#include <string>

#include "VO_ASMNDProfiles.h"
#include "VO_CVCommon.h"


/**
 * @author      JIA Pei
 * @version     2010-02-05
 * @brief       Calculate how many profiles for each level
 * @param       nbOfProfilesPerPixelAtLevels    Output - number of profiles per pixel at different levels
 * @param       NbOfLevels              Input - profile k pixels refer to Cootes "Statistical Models of Appearance for Computer Vision" page 38
 * @param       NbOfProfileInLevel0     Input - an odd natural number, number of samples for a single landmark
 * @param       ProduceMethod           Input - method
 * @return      void
*/
void VO_ASMNDProfiles::VO_ProduceLevelProfileNumbers(   vector<unsigned int>& nbOfProfilesPerPixelAtLevels,
                                                        unsigned int NbOfLevels, 
                                                        unsigned int NbOfProfileInLevel0, 
                                                        unsigned int ProduceMethod)
{
    /////////////////////////////////////////////////////////////////////////////////
    if(NbOfProfileInLevel0%2 == 0)
    {
        cerr << "Number of Profiles in level 0 must be an odd " << endl;
        exit(1);
    }
    /////////////////////////////////////////////////////////////////////////////////

    nbOfProfilesPerPixelAtLevels.resize(NbOfLevels);
    nbOfProfilesPerPixelAtLevels[0] = NbOfProfileInLevel0;
    switch(ProduceMethod)
    {
        case DESCENDING:
        {
            if( (NbOfProfileInLevel0 - (NbOfLevels-1)*2) <= 0 )
            {
                cerr << "Too many multi scale levels. DESCENDING. VO_ASMNDProfiles. " << endl;
                exit(1);
            }
            for(unsigned int i = 1; i < NbOfLevels; i++)
            {
                nbOfProfilesPerPixelAtLevels[i] = nbOfProfilesPerPixelAtLevels[0] - i*2;
            }
        }
        break;
        case PYRAMID:
        {
            if( ( (float)(NbOfProfileInLevel0 - 1) / ( pow( 2.0f, (float)(NbOfLevels) ) ) ) < 2.0 )
            {
                cerr << "Too many multi scale levels. PYRAMID. VO_ASMNDProfiles. " << endl;
                exit(1);
            }
            if( ( (NbOfProfileInLevel0 - 1) / (unsigned int) ( pow( 2.0f, (float)(NbOfLevels) ) ) ) % 2 != 0 )
            {
                cerr << "Multi scale levels are not suitable for PYRAMID. " << endl;
                exit(1);
            }
            for(unsigned int i = 1; i < NbOfLevels; i++)
            {
                unsigned int PyrScale = (unsigned int) ( pow(2.0f, (float)(i) ) );
                nbOfProfilesPerPixelAtLevels[i] = nbOfProfilesPerPixelAtLevels[0] / PyrScale + 1;
            }
        }
        break;
        case SAME:
        default:
        {
            for(unsigned int i = 1; i < NbOfLevels; i++)
            {
                nbOfProfilesPerPixelAtLevels[i] = nbOfProfilesPerPixelAtLevels[0];
            }
        }
        break;
    }
}


/**
 * @author      JIA Pei
 * @version     2010-02-05
 * @brief       Calculate all profiles
 * @return      void
*/
void VO_ASMNDProfiles::VO_LoadProfileTrainingData (int loadFlag)
{
    Mat img, resizedImg;
    VO_Shape resizedShape;
    VO_Profile tempProfile;
    float PyrScale, scale, sampleScale;
    float refScale = this->m_VOReferenceShape.GetCentralizedShapeSize();

    for( unsigned int i = 0; i < this->m_iNbOfSamples; ++i )
    {
		img = imread ( this->m_vStringTrainingImageNames[i].c_str (), loadFlag );

        sampleScale = this->m_vShapes[i].GetCentralizedShapeSize();

        for( unsigned int j = 0; j < this->m_iNbOfPyramidLevels; ++j)
        {
            PyrScale = pow(2.0f, (float)j);
            scale = refScale/sampleScale/PyrScale;

            resizedShape = this->m_vShapes[i]*scale;
            cv::resize(img, resizedImg, Size( (int)(img.cols*scale), (int)(img.rows*scale) ) );

// static stringstream ss;
// static string ssi;
// ss << i;
// ss >> ssi;
// static string fn = ssi+".jpg";
// imwrite(fn.c_str(), resizedImg);

            // Explained by JIA Pei -- wavelet feature extraction
            for( unsigned int k = 0; k < this->m_iNbOfPoints; ++k )
            {
                VO_Profile::VO_GetNDProfiles4OneLandmark (  resizedImg,
                                                            resizedShape,
                                                            this->m_vShape2DInfo,
                                                            k,
                                                            tempProfile,
                                                            this->m_iNbOfProfileDim,
                                                            this->m_iNbOfProfilesPerPixelAtLevels[j] );
                this->m_vvvNormalizedProfiles[i][j][k].SetProfile(tempProfile );
                this->m_vvvNormalizedProfiles[i][j][k].Normalize();
            }
        }
    }
}


/**
 * @author      JIA Pei
 * @version     2010-02-05
 * @brief       Calculate statistics for all profiles; Computer all landmarks' mean prof and covariance matrix
 * @return      void
*/
void VO_ASMNDProfiles::VO_CalcStatistics4AllProfiles()
{
    // Calcuate Inverse of Sg
    for(unsigned int i = 0; i < this->m_iNbOfPyramidLevels; i++)
    {
        Mat_<float> allProfiles = Mat_<float>::zeros( this->m_iNbOfSamples, this->m_vvvNormalizedProfiles[0][i][0].GetProfileLength() );
        Mat_<float> Covar = Mat_<float>::zeros(this->m_vvvNormalizedProfiles[0][i][0].GetProfileLength(), 
                                                this->m_vvvNormalizedProfiles[0][i][0].GetProfileLength() );
        Mat_<float> Mean = Mat_<float>::zeros(1, this->m_vvvNormalizedProfiles[0][i][0].GetProfileLength() );
        for(unsigned int j = 0; j < this->m_iNbOfPoints; j++)
        {
            for(unsigned int k = 0; k < this->m_iNbOfProfileDim; k++)
            {
                for(unsigned int l = 0; l < this->m_iNbOfSamples; l++)
                {
                    Mat_<float> tmpRow = allProfiles.row(l);
                    Mat_<float> tmp = this->m_vvvNormalizedProfiles[l][i][j].GetTheProfile().col(k).t();
                    tmp.copyTo(tmpRow);
                }

                // OK! Now We Calculate the Covariance Matrix of prof for Landmark iPoint
                cv::calcCovarMatrix( allProfiles, Covar, Mean, CV_COVAR_NORMAL+CV_COVAR_ROWS+CV_COVAR_SCALE, CV_32F);
//                cv::calcCovarMatrix( allProfiles, Covar, Mean, CV_COVAR_SCRAMBLED+CV_COVAR_ROWS, CV_32F);
                this->m_vvMeanNormalizedProfile[i][j].Set1DimProfile(Mean.t(), k);

                // Explained by YAO Wei, 2008-1-29.
                // Actually Covariance Matrix is semi-positive definite. But I am not sure
                // whether it is invertible or not!!!
                // In my opinion it is un-invert, since C = X.t() * X!!!
                cv::invert(Covar, this->m_vvvCVMInverseOfSg[i][j][k], DECOMP_SVD);
            }
        }
    }
}


/**
 * @author      JIA Pei
 * @version     2010-02-22
 * @brief       Build ND Profile ASM
 * @param       allLandmarkFiles4Training   Input - all training landmark files
 * @param       allImgFiles4Training        Input - all training image files
 * @param       shapeinfoFileName           Input - shape info file
 * @param       database                    Input - which database is it?
 * @param       channels                    Input - How many channels are to be used?
 * @param       levels                      Input - multiscale levels
 * @param       profdim                     Input - ND profile, how many dimensions?
 * @param       kk                          Input - kk, refer to Cootes' paper "Statistical Model for Computer Vision"
 * @param       trm                         Input - texture representation method
 * @param       TPShape                     Input - truncated percentage for shape model
 * @param       useKnownTriangles           Input - use known triangle structures??
 * @return      void
  */
void VO_ASMNDProfiles::VO_BuildASMNDProfiles (  const vector<string>& allLandmarkFiles4Training,
                                                const vector<string>& allImgFiles4Training,
                                                const string& shapeinfoFileName,
                                                unsigned int database,
                                                unsigned int channels,
                                                unsigned int levels,
                                                unsigned int profdim,
                                                unsigned int kk,
                                                int trm,
                                                float TPShape,
                                                bool useKnownTriangles )
{
    if (allLandmarkFiles4Training.size() != allImgFiles4Training.size() )
        cerr << "allLandmarkFiles4Training should have the same number of allImgFiles4Training! " << endl;
        
	//Not checking actual supported channels image
	if( (channels == 1 && (profdim == 1 || profdim == 2) ) || (profdim == 2*channels))
		cerr << profdim <<" m_iNbOfProfileDim should only be 1, or 2 times the number of channels!" << endl;
        
    this->VO_BuildShapeModel(allLandmarkFiles4Training, shapeinfoFileName, database, TPShape, useKnownTriangles);
    this->m_iNbOfChannels                   = channels;
    this->m_iNbOfPyramidLevels              = levels;
	//2 profiles per channel, unless 1D profiles
	if(profdim == 1)
	{
		this->m_iNbOfProfileDim					= profdim;
	}else
	{
		this->m_iNbOfProfileDim					= 2*channels;
	}
	
    this->m_iTextureRepresentationMethod    = trm;
    this->m_vStringTrainingImageNames       = allImgFiles4Training;

    // Initialize all member variables
    VO_ASMNDProfiles::VO_ProduceLevelProfileNumbers(this->m_iNbOfProfilesPerPixelAtLevels, this->m_iNbOfPyramidLevels, 2*kk+1);    // m_iNbOfProfilesPerPixelAtLevels

    this->m_vvvNormalizedProfiles.resize ( this->m_iNbOfSamples );
    for(unsigned int i = 0; i < this->m_iNbOfSamples; i++)
    {
        this->m_vvvNormalizedProfiles[i].resize ( this->m_iNbOfPyramidLevels );
        for(unsigned int j = 0; j< this->m_iNbOfPyramidLevels; j++)
        {
            this->m_vvvNormalizedProfiles[i][j].resize(this->m_iNbOfPoints);
        }
    }

    this->m_vvMeanNormalizedProfile.resize(this->m_iNbOfPyramidLevels);
    this->m_vvvCVMInverseOfSg.resize(this->m_iNbOfPyramidLevels);


    for(unsigned int i = 0; i < this->m_iNbOfPyramidLevels; i++)
    {
        this->m_vvMeanNormalizedProfile[i].resize(this->m_iNbOfPoints);
        this->m_vvvCVMInverseOfSg[i].resize(this->m_iNbOfPoints);
        for(unsigned int j = 0; j < this->m_iNbOfPoints; j++)
        {
            this->m_vvvCVMInverseOfSg[i][j].resize(this->m_iNbOfProfileDim);
            this->m_vvMeanNormalizedProfile[i][j].Resize(this->m_iNbOfProfilesPerPixelAtLevels[i], this->m_iNbOfProfileDim);
            for(unsigned int k = 0; k < this->m_iNbOfProfileDim; k++)
            {
                this->m_vvvCVMInverseOfSg[i][j][k] = Mat_<float>::zeros(this->m_iNbOfProfilesPerPixelAtLevels[i], this->m_iNbOfProfilesPerPixelAtLevels[i]);
            }
        }
    }
	//can only force 1 channel
	if(channels == 1)
	{
		//returns a grayscale image
		this->VO_LoadProfileTrainingData (0);
	}else
	{
		//should load the alpha channel as well.
		this->VO_LoadProfileTrainingData(-1);
	}
    this->VO_CalcStatistics4AllProfiles();
}


void VO_ASMNDProfiles::VO_Save ( const string& fd )
{
    VO_AXM::VO_Save(fd);

    // create ASM subfolder for just ASM model data
    string fn = fd+"/ASMNDProfiles";
    MakeDirectory(fn);

    ofstream fp;
    string tempfn;

    // ASMNDProfiles
    tempfn = fn + "/ASMNDProfiles" + ".txt";
    fp.open(tempfn.c_str (), ios::out);
    fp << "m_iNbOfProfileDim" << endl << this->m_iNbOfProfileDim << endl;
    fp << "m_iNbOfProfilesPerPixelAtLevels[0]" << endl << this->m_iNbOfProfilesPerPixelAtLevels[0] << endl;
    fp.close();fp.clear();

    // m_vvMeanNormalizedProfile
    tempfn = fn + "/m_vvMeanNormalizedProfile" + ".txt";
    fp.open(tempfn.c_str (), ios::out);
    fp << "m_vvMeanNormalizedProfile" << endl;
    // fp << this->m_vvMeanNormalizedProfile;
    // You can output everything by the above line, but you won't have level and node description in the output file.
    for (unsigned int i = 0; i < this->m_iNbOfPyramidLevels; i++)
    {
        for (unsigned int j = 0; j < this->m_iNbOfPoints; j++)
        {
            fp << "level " << i << " node " << j << endl;
            fp << this->m_vvMeanNormalizedProfile[i][j] << endl;
        }
    }
    fp.close();fp.clear();

    // m_vvvCVMInverseOfSg
    tempfn = fn + "/m_vvvCVMInverseOfSg" + ".txt";
    fp.open(tempfn.c_str (), ios::out);
    fp << "m_vvvCVMInverseOfSg" << endl;
    //fp << this->m_vvvCVMInverseOfSg;
    // You can output everything by the above line, but you won't have level and node description in the output file.
    for (unsigned int i = 0; i < this->m_iNbOfPyramidLevels; i++)
    {
        for (unsigned int j = 0; j < this->m_iNbOfPoints; j++)
        {
            for(unsigned int k = 0; k < this->m_iNbOfProfileDim; k++)
            {
                fp << "level " << i << " node " << j << " dim " << k << endl;
                fp << this->m_vvvCVMInverseOfSg[i][j][k] << endl;
            }
        }
    }
    fp.close();fp.clear();
}


void VO_ASMNDProfiles::VO_Load ( const string& fd )
{
    VO_AXM::VO_Load(fd);

    string fn = fd+"/ASMNDProfiles";
    if (!MakeDirectory(fn) )
    {
        cout << "ASMNDProfiles subfolder is not existing. " << endl;
        exit(EXIT_FAILURE);
    }

    ifstream fp;
    string tempfn;
    string temp;

    // ASMNDProfiles
    tempfn = fn + "/ASMNDProfiles" + ".txt";
    SafeInputFileOpen(fp, tempfn);
    fp >> temp >> this->m_iNbOfProfileDim;                      // m_iNbOfProfileDim
    this->m_iNbOfProfilesPerPixelAtLevels.resize(this->m_iNbOfPyramidLevels);
    fp >> temp >> this->m_iNbOfProfilesPerPixelAtLevels[0];     // m_iNbOfProfilesPerPixelAtLevels[0]
    VO_ASMNDProfiles::VO_ProduceLevelProfileNumbers(this->m_iNbOfProfilesPerPixelAtLevels, this->m_iNbOfPyramidLevels, this->m_iNbOfProfilesPerPixelAtLevels[0]);
    fp.close();fp.clear();

    // m_vvMeanNormalizedProfile
    tempfn = fn + "/m_vvMeanNormalizedProfile" + ".txt";
    SafeInputFileOpen(fp, tempfn);
    fp >> temp;
    this->m_vvMeanNormalizedProfile.resize(this->m_iNbOfPyramidLevels);
    for (unsigned int i = 0; i < this->m_iNbOfPyramidLevels; i++)
    {
        this->m_vvMeanNormalizedProfile[i].resize(this->m_iNbOfPoints);
        for (unsigned int j = 0; j < this->m_iNbOfPoints; j++)
        {
            fp >> this->m_vvMeanNormalizedProfile[i][j];
        }
    }
    fp.close();fp.clear();

    // m_vvvCVMInverseOfSg
    tempfn = fn + "/m_vvvCVMInverseOfSg" + ".txt";
    SafeInputFileOpen(fp, tempfn);
    fp >> temp;
    this->m_vvvCVMInverseOfSg.resize(this->m_iNbOfPyramidLevels);
    for (unsigned int i = 0; i < this->m_iNbOfPyramidLevels; i++)
    {
        this->m_vvvCVMInverseOfSg[i].resize(this->m_iNbOfPoints);
        for (unsigned int j = 0; j < this->m_iNbOfPoints; j++)
        {
            this->m_vvvCVMInverseOfSg[i][j].resize(this->m_iNbOfProfileDim);
             for(unsigned int k = 0; k < this->m_iNbOfProfileDim; k++)
            {
                fp >> this->m_vvvCVMInverseOfSg[i][j][k];
            }
        }
    }
    fp.close();fp.clear();
}


void VO_ASMNDProfiles::VO_LoadParameters4Fitting ( const string& fd )
{
    VO_AXM::VO_LoadParameters4Fitting(fd);

    string fn = fd+"/ASMNDProfiles";
    if (!MakeDirectory(fn) )
    {
        cout << "ASMNDProfiles subfolder is not existing. " << endl;
        exit(EXIT_FAILURE);
    }

    ifstream fp;
    string tempfn;
    string temp;

    // ASMNDProfiles
    tempfn = fn + "/ASMNDProfiles" + ".txt";
    SafeInputFileOpen(fp, tempfn);
    fp >> temp >> this->m_iNbOfProfileDim;                      // m_iNbOfProfileDim
    this->m_iNbOfProfilesPerPixelAtLevels.resize(this->m_iNbOfPyramidLevels);
    fp >> temp >> this->m_iNbOfProfilesPerPixelAtLevels[0];     // m_iNbOfProfilesPerPixelAtLevels[0]
    VO_ASMNDProfiles::VO_ProduceLevelProfileNumbers(this->m_iNbOfProfilesPerPixelAtLevels, this->m_iNbOfPyramidLevels, this->m_iNbOfProfilesPerPixelAtLevels[0]);
    fp.close();fp.clear();

    // m_vvMeanNormalizedProfile
    tempfn = fn + "/m_vvMeanNormalizedProfile" + ".txt";
    SafeInputFileOpen(fp, tempfn);
    getline(fp, temp);
    this->m_vvMeanNormalizedProfile.resize(this->m_iNbOfPyramidLevels);
    for (unsigned int i = 0; i < this->m_iNbOfPyramidLevels; i++)
    {
        this->m_vvMeanNormalizedProfile[i].resize(this->m_iNbOfPoints);
        for (unsigned int j = 0; j < this->m_iNbOfPoints; j++)
        {
            fp >> temp;
            while ( !fp.eof() && temp!="level")
            {
                fp >> temp;
            }
            getline(fp, temp);
            this->m_vvMeanNormalizedProfile[i][j].Resize(this->m_iNbOfProfilesPerPixelAtLevels[i], this->m_iNbOfProfileDim);
            fp >> this->m_vvMeanNormalizedProfile[i][j];
        }
    }

    fp.close();fp.clear();
    

    // m_vvvCVMInverseOfSg
    tempfn = fn + "/m_vvvCVMInverseOfSg" + ".txt";
    SafeInputFileOpen(fp, tempfn);
    getline(fp, temp);
    this->m_vvvCVMInverseOfSg.resize(this->m_iNbOfPyramidLevels);
    for (unsigned int i = 0; i < this->m_iNbOfPyramidLevels; i++)
    {
        this->m_vvvCVMInverseOfSg[i].resize(this->m_iNbOfPoints);
        for (unsigned int j = 0; j < this->m_iNbOfPoints; j++)
        {
            this->m_vvvCVMInverseOfSg[i][j].resize(this->m_iNbOfProfileDim);
             for(unsigned int k = 0; k < this->m_iNbOfProfileDim; k++)
            {
                fp >> temp;
                while ( !fp.eof() && temp!= "level")
                {
                    fp >> temp;
                }
                getline(fp, temp);
                this->m_vvvCVMInverseOfSg[i][j][k] = 
                    Mat_<float>::zeros( this->m_iNbOfProfilesPerPixelAtLevels[i],
                                        this->m_iNbOfProfilesPerPixelAtLevels[i]);
                fp >> this->m_vvvCVMInverseOfSg[i][j][k];
            }
        }
    }
    fp.close();fp.clear();
}

