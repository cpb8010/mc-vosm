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
#include <fstream>

#include <boost/filesystem.hpp>
#include <boost/regex/v4/fileiter.hpp>

#include "opencv/cv.h"
#include "opencv/highgui.h"

#include "smb.h"

using namespace std;
using namespace cv;


void usage_build()
{
    cout << "Usage: test_smbuilding [options] save_directory annotation_directory image_directory shapeinfo_path database channels type levels percentage " << endl
        << "options: " << endl
        << "   -o    output directory (default './') " << endl
        << "   -a    annotation directory (required) " << endl
        << "   -i    image directory (required) " << endl
        << "   -s    path of the file shapeinfo (required) " << endl
        << "   -d    training database (required, IMM, AGING, BIOID, XM2VTS, FRANCK, EMOUNT, JIAPEI, BOSPHORUS ) " << endl
        << "   -c    channels (1, 2, or 3, default 3) " << endl
        << "   -t    statistical model type (SM, TM, AM, IA, FM, SMLTC, SMNDPROFILE. default SMNDPROFILE ) " << endl
        << "   -l    level of parymid (default 4) " << endl
        << "   -p    percentage of shape, texture and appearance PCA (default 0.95) " << endl;

    exit(0);
}


void parse_option(  int argc,
                    char **argv,
                    string& outputDir,
                    vector<string>& annotationFNs, 
                    vector<string>& imageFNs, 
                    string& shapeinfoFN, 
                    unsigned int& database,
                    unsigned int& channels,
                    unsigned int& type,
                    unsigned int& levels,
                    double& percentage)
{
    char *arg = NULL;
    int optindex;
    
    /* parse options */
    optindex = 0;
    while (++optindex < argc)
    {
        if(argv[optindex][0] != '-') break;
        if(++optindex >= argc) usage_build();

        switch(argv[optindex-1][1])
        {
        case 'o':
            outputDir       = argv[optindex];
            break;
        case 'a':
        {
            if ( ! MakeDirectory( argv[optindex] ) )
            {
                cerr << "landmark path does not exist!" << endl;
                exit(EXIT_FAILURE);
            }
            annotationFNs   = VO_IO::ScanNSortAnnotationInDirectory ( argv[optindex] );
        }
            break;
        case 'i':
        {
            if ( ! MakeDirectory( argv[optindex] ) )
            {
                cerr << "image path does not exist!" << endl;
                exit(EXIT_FAILURE);
            }
            imageFNs        = VO_IO::ScanNSortImagesInDirectory ( argv[optindex] );
        }
            break;
        case 's':
        {
			
            if ( ! boost::filesystem::is_regular( argv[optindex] ) )
            {
                cerr << "shapeinfo file does not exist!" << endl;
                exit(EXIT_FAILURE);
            }
			
            shapeinfoFN     = argv[optindex];
        }
            break;
        case 'd':
        {
            if(strcmp(argv[optindex], "PUT") == 0)
                database    = CAnnotationDBIO::PUT;
            else if(strcmp(argv[optindex], "IMM") == 0)
                database    = CAnnotationDBIO::IMM;
            else if(strcmp(argv[optindex], "AGING") == 0)
                database    = CAnnotationDBIO::AGING;
            else if(strcmp(argv[optindex], "BIOID") == 0)
                database    = CAnnotationDBIO::BIOID;
            else if(strcmp(argv[optindex], "XM2VTS") == 0)
                database    = CAnnotationDBIO::XM2VTS;
            else if(strcmp(argv[optindex], "FRANCK") == 0)
                database    = CAnnotationDBIO::FRANCK;
            else if(strcmp(argv[optindex], "EMOUNT") == 0)
                database    = CAnnotationDBIO::EMOUNT;
            else if(strcmp(argv[optindex], "JIAPEI") == 0)
                database    = CAnnotationDBIO::JIAPEI;
			else if(strcmp(argv[optindex], "BOSPHORUS") == 0)
				database	= CAnnotationDBIO::BOSPHORUS;
        }
            break;
        case 'c':
            channels        = atoi(argv[optindex]);
            break;
        case 't':
        {
            if(strcmp(argv[optindex], "SM") == 0)
                type        = SHAPEMODEL;
            else if(strcmp(argv[optindex], "TM") == 0)
                type        = TEXTUREMODEL;
            else if(strcmp(argv[optindex], "AM") == 0)
                type        = APPEARANCEMODEL;
            else if(strcmp(argv[optindex], "IA") == 0)
                type        = INVERSEIMAGEALIGNMENT;
            else if(strcmp(argv[optindex], "FM") == 0)
                type        = AFM ;
            else if(strcmp(argv[optindex], "SMLTC") == 0)
                type        = ASMLTC;
            else if(strcmp(argv[optindex], "SMNDPROFILE") == 0)
                type        = ASMNDPROFILE;
        }
            break;
        case 'l':
            levels          = atoi(argv[optindex]);
            break;
        case 'p':
            percentage      = atof(argv[optindex]);
            break;
        default:
        {
            cerr << "unknown options" << endl;
            usage_build();
        }
            break;
        }
    }
    
    if (annotationFNs.size() == 0)
    {
        cerr << " No landmark loaded" << endl;
        usage_build();
        exit(EXIT_FAILURE);
    }
    else if (imageFNs.size() == 0)
    {
        cerr << " No image loaded" << endl;
        usage_build();
        exit(EXIT_FAILURE);
    }
    else if (annotationFNs.size() != imageFNs.size())
    {
        cerr << " The number of images should be equal to the number of landmarks" << endl;
        usage_build();
        exit(EXIT_FAILURE);
    }
}


int main(int argc, char **argv)
{
    string              outputDir = "./";
    vector<string>      AllImgFiles4Training;
    vector<string>      AllLandmarkFiles4Training;
    string              shapeinfoFileName;
    unsigned int        database = CAnnotationDBIO::EMOUNT;
    unsigned int        channels = 3;
    unsigned int        type = ASMNDPROFILE;
    unsigned int        levels = 4;
    double              percentage = 0.95;
    
    parse_option(   argc,
                    argv,
                    outputDir,
                    AllLandmarkFiles4Training,
                    AllImgFiles4Training,
                    shapeinfoFileName,
                    database,
                    channels,
                    type,
                    levels,
                    percentage);

    switch(type)
    {
        case SHAPEMODEL:
        {
            VO_ShapeModel shapeModel;
            shapeModel.VO_BuildShapeModel(  AllLandmarkFiles4Training,
                                            shapeinfoFileName,
                                            database,
                                            safeDoubleToFloat(percentage),
                                            false);
            shapeModel.VO_Save(outputDir);
        }
        break;
        case TEXTUREMODEL:
        {
            VO_TextureModel textureModel;
            textureModel.VO_BuildTextureModel(  AllLandmarkFiles4Training,
                                                AllImgFiles4Training,
                                                shapeinfoFileName, 
                                                database,
                                                channels,
                                                VO_Features::DIRECT,
                                                safeDoubleToFloat(percentage),
                                                safeDoubleToFloat(percentage),
                                                false );
            textureModel.VO_Save(outputDir);
        }
        break;
        case APPEARANCEMODEL:
        {
            VO_AAMBasic aamBasicModel;
            aamBasicModel.VO_BuildAppearanceModel(  AllLandmarkFiles4Training,
                                                    AllImgFiles4Training,
                                                    shapeinfoFileName, 
                                                    database,
                                                    channels,
                                                    levels,
                                                    VO_Features::DIRECT,
                                                    safeDoubleToFloat(percentage),
                                                    safeDoubleToFloat(percentage),
                                                    safeDoubleToFloat(percentage),
                                                    false );
            aamBasicModel.VO_Save(outputDir);
        }
        break;
        case INVERSEIMAGEALIGNMENT:
        {
            VO_AAMInverseIA aamInverseIAModel;
            aamInverseIAModel.VO_BuildAAMICIA(  AllLandmarkFiles4Training,
                                                AllImgFiles4Training,
                                                shapeinfoFileName,
                                                database,
                                                channels,
                                                levels,
                                                VO_Features::DIRECT,
                                                safeDoubleToFloat(percentage),
                                                safeDoubleToFloat(percentage),
                                                false );
            aamInverseIAModel.VO_Save(outputDir);
        }
        break;
        case AFM:
        {
            VO_AFM featureModel;
            featureModel.VO_BuildFeatureModel(  AllLandmarkFiles4Training,
                                                AllImgFiles4Training,
                                                shapeinfoFileName,
                                                database,
                                                channels,
                                                levels,
                                                VO_Features::DIRECT,
                                                safeDoubleToFloat(percentage),
                                                false,
                                                VO_DiscreteWavelet::HAAR,
                                                Size(16, 16) );
            featureModel.VO_Save(outputDir);
        }
        break;
        case ASMLTC:
        {
            VO_ASMLTCs asmLTCModel;
            asmLTCModel.VO_BuildASMLTCs(AllLandmarkFiles4Training,
                                        AllImgFiles4Training,
                                        shapeinfoFileName,
                                        database,
                                        channels,
                                        levels,
                                        VO_Features::DIRECT,
                                        safeDoubleToFloat(percentage),
                                        false,
                                        VO_Features::DIRECT,
                                        Size(16, 16) );
            asmLTCModel.VO_Save(outputDir);
        }
        break;
        case ASMNDPROFILE:
        {
            VO_ASMNDProfiles asmNDProfilesModel;
            asmNDProfilesModel.VO_BuildASMNDProfiles(   AllLandmarkFiles4Training,
                                                        AllImgFiles4Training,
                                                        shapeinfoFileName, 
                                                        database,
                                                        channels,
                                                        levels,
                                                        2, // profdim
                                                        8, //kk
                                                        VO_Features::DIRECT, //trm
                                                        safeDoubleToFloat(percentage), //tpsShape
                                                        false); //useKnownTriangles
            asmNDProfilesModel.VO_Save(outputDir);
        }
        break;
    }

    return 0;
}
