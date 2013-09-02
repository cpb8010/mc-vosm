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
#include <functional>
#include <numeric>
#include "float.h"

#include <boost/filesystem.hpp>
#include <boost/regex/v4/fileiter.hpp>

#include "yaml-cpp/yaml.h"

#include "opencv/cv.h"
#include "opencv/highgui.h"

#include "smf.h"
#include "VO_FaceKeyPoint.h"

using namespace std;
using namespace cv;

//implemented in file
enum dualChannelCanidate{ VJC = 0, MINMAX = 1, GT = 2, MINVJC = 3};


void usage_build()
{
    cout << "Usage: test_smfitting [options] trained_data type testing_images testing_annotations database staticORdynamic recording" << endl
        << "options: " << endl
        << "   -o    trained data directory (required) " << endl
        << "   -t    fitting method to be used (ASM_PROFILEND, ASM_LTC, AAM_BASIC, AAM_CMUICIA, AAM_IAIA. default ASM_PROFILEND ) " << endl
        << "   -i   testing image directory containing at least one image (required) " << endl
        << "   -a   testing annotation directory (can be ignored) " << endl
        << "   -d   testing database -- if annotation directory is specified, database should also be specified for further evaluation on fitting performance (can be ignored) " << endl
        << "   -s   static image sequence or dynamic image sequence (default value true) " << endl
        << "   -r   recording the fitting results or not (default value false) " << endl
		<< "   -c   enables two-channel fitting technique(gray = channel 1 only, depth = channel 2 only, pyr = hybrid pyramid, sub = hybrid subset, default = full 2-channel) " << endl 
		<< "   -l   canidate location finding technique for two-channel fitting (min-max = search channel 2, GT = taken from annotations, VJc = OpenCV's Adaboost, min-VJc = Hybrid" << endl
        << "   -y   yaml config file" << endl
		<< "   -w   use webcam number" << endl
		<< endl << endl;

    cout << "Note: current smfitting doesn't defaultly afford 1D Profile ASM. " << endl
        << "If you would like to try 1D Profile ASM for static images in current smfitting, "
        << "you have to manually change the code in function VO_Fitting2DSM::VO_StartFitting "
        << "in file VO_Fitting2DSM.cpp, say, around line 306 of file VO_Fitting2DSM.cpp "
        << "change the 5th parameter from '2' to '1' of function "
        << "dynamic_cast<VO_FittingASMNDProfiles*>(this)->VO_ASMNDProfileFitting. " << endl << endl
        << "If you would like to try 1D Profile ASM for dynamic image sequence in current smfitting, "
        << "you have to manually change the code in function main() "
        << "in file smfitting.cpp, say around line 666 of file smfitting.cpp "
        << "change the 5th parameter from '2' to '1' of function "
        << "dynamic_cast<VO_FittingASMNDProfiles*>(fitting2dsm)->VO_ASMNDProfileFitting "
        << endl << endl;
    
    cout << "Face Detection: current smfitting use Adaboost technology to detect face as well as face components "
        << "for face location initialization. Refer to CFaceDetectionAlgs in main(). "
        << "Default Adaboost detectors installed with OpenCV installation are used in current smfitting "
        << "You may manually change the Adaboost detectors according to your own cascade file paths "
        << endl << endl;
    
    cout << "Face Tracking: current smfitting deals with image sequences. "
        << "If static images are to be tested, there is no point to carry out tracking because "
        << "for every image, Adaboost detection will be carried out. "
        << "If dynamic image sequences are to be tested, current smfitting only affords Camshift tracking strategy. "
        << "Please Refer to CTrackingAlgs() in main(), the default setting of function CTrackingAlgs() is Camshift algorithm "
        << endl << endl;
        
    cout<< "Vision Open doesn't afford the video IO or webcam IO, although the author has done his own IO for all kinds. "
        << "Users are highly encouraged to use their own video file or webcam IO and use VOSM in their own real-time applications. "
        << endl << endl;
    
    exit(-1);
}

struct VOSM_Fitting_Args{
	string trainedData;
	VO_AXM::TYPE type;
    vector<string> imageFNs;
    vector<string> annotationFNs;
	unsigned int channels;
	VO_Fitting2DSM::MULTICHANNELTECH fitTech;
	vector<VO_Fitting2DSM::MULTICHANNELTECH> fitTechs;
	dualChannelCanidate canidateLoc;
	CAnnotationDBIO::DB database;
    bool staticImgs;
    bool recordResults;
	int webcamNumber;
	string webcamName;
};


void parse_option(  char argType,
                    std::string argVal,
					VOSM_Fitting_Args& output
                    )
{

    switch(argType)
    {
    case 'o':
        output.trainedData     = argVal;
        break;
    case 't':
    {
        if(argVal == "ASM_PROFILEND")
            output.type        = VO_AXM::ASM_PROFILEND;
        else if(argVal == "ASM_LTC")
            output.type        = VO_AXM::ASM_LTC;
        else if(argVal == "AAM_BASIC")
            output.type        = VO_AXM::AAM_BASIC;
        else if(argVal == "AAM_CMUICIA")
            output.type        = VO_AXM::AAM_CMUICIA;
        else if(argVal == "AAM_IAIA")
            output.type        = VO_AXM::AAM_IAIA;
        else
        {
            cerr << "Wrong fitting type parameters!" << endl;
            exit(EXIT_FAILURE);
        }
    }
        break;
    case 'i':
    {
        if ( ! MakeDirectory( argVal ) )
        {
            cerr << "image path does not exist!" << endl;
            exit(EXIT_FAILURE);
        }
        output.imageFNs        = VO_IO::ScanNSortImagesInDirectory ( argVal );
    }
        break;
    case 'a':
    {
        if ( ! MakeDirectory( argVal ) )
        {
            cerr << "landmark path does not exist!" << endl;
            exit(EXIT_FAILURE);
        }
        output.annotationFNs   = VO_IO::ScanNSortAnnotationInDirectory ( argVal );
    }
        break;
    case 'd':
    {
        if(argVal == "PUT")
            output.database    = CAnnotationDBIO::PUT;
        else if(argVal == "IMM")
            output.database    = CAnnotationDBIO::IMM;
        else if(argVal == "AGING")
            output.database    = CAnnotationDBIO::AGING;
        else if(argVal == "BIOID")
            output.database    = CAnnotationDBIO::BIOID;
        else if(argVal == "XM2VTS")
            output.database    = CAnnotationDBIO::XM2VTS;
        else if(argVal == "FRANCK")
            output.database    = CAnnotationDBIO::FRANCK;
        else if(argVal == "EMOUNT")
            output.database    = CAnnotationDBIO::EMOUNT;
		else if(argVal == "BOSPHORUS")
			output.database	= CAnnotationDBIO::BOSPHORUS;
        else if(argVal == "JIAPEI")
            output.database    = CAnnotationDBIO::JIAPEI;
        else
        {
            cerr << "Wrong database parameters!" << endl;
            exit(EXIT_FAILURE);
        }
    }
        break;
    case 's':
    {
        if(argVal == "false")
            output.staticImgs    = false;
        else if(argVal == "true")
            output.staticImgs    = true;
        else
        {
            cerr << "Wrong StaticOrNot parameter!" << endl;
            exit(EXIT_FAILURE);
        }
			
	}
		break;
	case 'c':
	{
		//Uses assumpions for BS database
		output.channels 		= 2;
		if(argVal == "gray")
			output.fitTech = VO_Fitting2DSM::FIRSTCHANNELONLY;
		else if(argVal == "first")
			output.fitTech = VO_Fitting2DSM::FIRSTCHANNELONLY;
		else if(argVal == "depth")
			output.fitTech = VO_Fitting2DSM::SECONDCHANNELONLY;
		else if(argVal == "second")
			output.fitTech = VO_Fitting2DSM::SECONDCHANNELONLY;
		else if(argVal == "third")
			output.fitTech = VO_Fitting2DSM::THIRDCHANNELONLY;
		else if(argVal == "pyr")
			output.fitTech = VO_Fitting2DSM::HYBRIDPYRAMID;
		else if(argVal == "sub")
			output.fitTech = VO_Fitting2DSM::HYBRIDSUBSET;
		else
		{
			output.fitTech = VO_Fitting2DSM::FULLHYBRID;
		}
	}
		break;
	case 'l':
		{
			if(argVal == "min-max")
				output.canidateLoc = MINMAX;
			else if(argVal == "GT")
				output.canidateLoc = GT;
			else if(argVal == "VJc")
				output.canidateLoc = VJC;
			else 
			{
				output.canidateLoc = MINVJC;
			}
    }
        break;
    case 'r':
    {
        if(argVal == "false")
            output.recordResults    = false;
        else if(argVal == "true")
            output.recordResults    = true;
        else
        {
            cerr << "Wrong recordResults parameter!" << endl;
            exit(EXIT_FAILURE);
        }
    }
        break;
	case 'y':
    {
		// Recursivly calls parse_option for certain values
		// so, clearly don't call parse_option('y',...) from here)
		YAML::Node config = YAML::LoadFile(argVal);

        if(config["config-version"])
			std::cout << "config-version=" << config["version"].as<std::string>() << "\n";

		// Verbose lets us skip verbose checks for existence
		if(config["verbose"].as<bool>()){
		
			parse_option('i', config["test-image-directory"].as<std::string>(),output);
			parse_option('c',config["fitting"]["channel-tech"].as<std::string>(),output);
			parse_option('a',config["annotation"]["directory"].as<std::string>(),output);
			parse_option('l',config["fitting"]["starting-location"].as<std::string>(),output);
			parse_option('d',config["annotation"]["format"].as<std::string>(),output);
			parse_option('t',config["fitting"]["method"].as<std::string>(),output);
			parse_option('w',config["webcam"].as<std::string>(),output);
	
			output.recordResults = config["fitting"]["record"].as<bool>();
			output.staticImgs = config["fitting"]["static-images"].as<bool>();
			output.trainedData = config["trained-model-directory"].as<std::string>();
			output.channels = config["fitting"]["channels"].as<int>();//right now, this is only way to specify 3 channels

		}else{

			if(config["test-image-directory"])
				parse_option('i', config["test-image-directory"].as<std::string>(),output);
			if(config["fitting"]["method"])
				parse_option('t', config["fitting"]["method"].as<std::string>(),output);
			if(config["fitting"]["channel-tech"])
				parse_option('c',config["fitting"]["channel-tech"].as<std::string>(),output);
			if(config["annotation"]["directory"])
				parse_option('a',config["annotation"]["directory"].as<std::string>(),output);
			if(config["fitting"]["starting-location"])
				parse_option('l',config["fitting"]["starting-location"].as<std::string>(),output);
			if(config["annotation"]["format"])
				parse_option('d',config["annotation"]["format"].as<std::string>(),output);
			if(config["webcam"])
				parse_option('w',config["webcam"].as<std::string>(),output);

			if(config["fitting"]["record"])
				output.recordResults = config["fitting"]["record"].as<bool>();
			if(config["fitting"]["static-images"])
				output.staticImgs = config["fitting"]["static-images"].as<bool>();
			if(config["trained-model-directory"])
				output.trainedData = config["trained-model-directory"].as<std::string>();
			if(config["fitting"]["channels"])
				output.channels = config["fitting"]["channels"].as<int>();		
		}

		//The only way to specify a subset is with the yaml file
		if(output.fitTech == VO_Fitting2DSM::HYBRIDSUBSET){
			if( !config["fitting"]["techniques"]){
				cerr << "Complete subset specification required" << endl;
				exit(EXIT_FAILURE);
			}
			//We don't know how many points are in the model yet, so we can't verify the completeness
			for(YAML::iterator iter = config["fitting"]["techniques"].begin(); iter != config["fitting"]["techniques"].end(); iter++){
				//convert from strings to enums
				if( iter->as<std::string>() == "default")
					output.fitTechs.push_back(VO_Fitting2DSM::FULLHYBRID);
				else if( iter->as<std::string>() == "gray")
					output.fitTechs.push_back(VO_Fitting2DSM::FIRSTCHANNELONLY);
				else if( iter->as<std::string>() == "first")
					output.fitTechs.push_back(VO_Fitting2DSM::FIRSTCHANNELONLY);
				else if( iter->as<std::string>() == "depth")
					output.fitTechs.push_back(VO_Fitting2DSM::SECONDCHANNELONLY);
				else if( iter->as<std::string>() == "second")
					output.fitTechs.push_back(VO_Fitting2DSM::SECONDCHANNELONLY);
				else if( iter->as<std::string>() == "third")
					output.fitTechs.push_back(VO_Fitting2DSM::THIRDCHANNELONLY);
				else if( iter->as<std::string>() == "pyr")
					output.fitTechs.push_back(VO_Fitting2DSM::HYBRIDPYRAMID);
				else if( iter->as<std::string>() == "sub")
					output.fitTechs.push_back(VO_Fitting2DSM::HYBRIDSUBSET);
				else{
					cerr << "Incorrect subset specification = " << iter->as<std::string>() << endl;
					exit(EXIT_FAILURE);
				}
			}
		}
    }
		break;
	case 'w':
	{
		//assume that device numbers will be between 0-9, and names will be 2+ characters
		if(argVal.length() == 1){
			output.webcamNumber = atoi(argVal.c_str());
			output.webcamName = "";
		}else{
			output.webcamNumber = -1;
			output.webcamName = argVal;
		}
			
        
	}
		break;
    default:
    {
        cerr << "unknown options" << endl;
        usage_build();
    }
        break;
    }
}

void cmd_line_parse( int argc,
                    char **argv,
                    VOSM_Fitting_Args& fittingArgs)
{
	char *arg = NULL;
    int optindex, handleoptions=1;
	;
    /* parse options */
    optindex = 0;
    while (++optindex < argc)
    {
        if(argv[optindex][0] != '-') break;
        if(++optindex >= argc) usage_build();

		parse_option(   argv[optindex-1][1],
						argv[optindex],
						fittingArgs);
	}

	if (fittingArgs.imageFNs.size() == 0)
    {
        cerr << " No image loaded" << endl;
        usage_build();
        exit(EXIT_FAILURE);
    }
    if (fittingArgs.annotationFNs.size() != 0 && fittingArgs.annotationFNs.size() != fittingArgs.imageFNs.size() )
    {
        cerr << " If annotations are loaded, then, the number of landmarks should be equal to the number of images " << endl;
        usage_build();
        exit(EXIT_FAILURE);
    }
}
void PrintSummary(const int nb, const int detectionTimes, const Mat& nbOfIterations, const Mat& times, bool doEvaluation,
					   const vector<float>& ptsErrorAvg, const Mat& deviations, const Mat& ptsErrorFreq){

	cout << "Detection Times = " << detectionTimes << endl;
	float sumIter = safeDoubleToFloat(cv::sum(nbOfIterations).val[0]);
	cout << "Average Number of Iterations = " << (sumIter / detectionTimes) << endl;
    float sumTime = safeDoubleToFloat(cv::sum(times).val[0]);
    cout << "Average Detection time (in ms) = " << (sumTime / detectionTimes) << endl;
    Scalar avgDev, stdDev;
    if(doEvaluation)
    {

		cout << "Average Pt Distance = " << std::accumulate(ptsErrorAvg.begin(),ptsErrorAvg.end(),0.00)/detectionTimes << endl;

        cv::meanStdDev(deviations, avgDev, stdDev);
        cout << "Average Deviation of Errors = " << avgDev.val[0] << endl;
		cout << "Standard Deviation of Errors = " << stdDev.val[0] << endl << endl;
        vector<float> avgErrorFreq(nb, 0.0f);
        for(int j = 0; j < nb; j++)
        {
            Mat_<float> col = ptsErrorFreq.col(j);
            avgErrorFreq[j] = safeDoubleToFloat(cv::mean(col).val[0]);
            cout << avgErrorFreq[j] << " percentage of points are in " << j << " pixels" << endl;
        }
    }
}

//Difficult to abstract, so it stays closeby.
void	SaveSequentialResultsInFolder(const Mat& img, const VO_Shape& refShape, vector<VO_Fitting2DSM::DrawMeshInfo>& meshInfos, const string& fdname){
	
	MakeDirectory(fdname);

	Mat tempMat;
	vector<float> ptErrorFreq;
	float deviation = 0.0f;
	vector<unsigned int> unsatisfiedPtList;
	vector<float> ptErrorPerPoint;
	for(unsigned int i = 0;i<meshInfos.size();++i){
		
		//get the error on the image
		ptErrorFreq.clear();
		ptErrorPerPoint.resize(meshInfos[i].drawPts.GetNbOfPoints(),0.0f);
		unsatisfiedPtList.clear();
		CRecognitionAlgs::CalcShapeFittingEffect(	refShape,
			(meshInfos[i].drawPts / meshInfos[i].f2Scale),
													deviation,
													ptErrorFreq,
													20,
													&ptErrorPerPoint);

		//Save that result to a folder based on the name, and the file based on the current index
		CRecognitionAlgs::SaveShapeResults(	fdname,
											static_cast<ostringstream*>( &(ostringstream() << i) )->str(),
											deviation,
											ptErrorPerPoint,
											ptErrorFreq,
											(meshInfos[i].drawPts / meshInfos[i].f2Scale));
		//draw the image
		VO_Fitting2DSM::VO_DrawMeshOnImage(meshInfos[i],img,tempMat);
		//Save the Image
		string fn = fdname +"/"+ static_cast<ostringstream*>( &(ostringstream() << i) )->str() +".jpg";
		imwrite(fn.c_str(), tempMat);
	}	
}

void drawFaceBoxes(cv::Rect* faceRect, cv::Mat* bgrimg, IplImage* depth, cv::Point leftEye, cv::Point rightEye, cv::Point nose){

	//dark blue rectangle for nose search
		cv::Rect noseRect;
		noseRect.x = faceRect->x + 1 + (faceRect->width/4);
		noseRect.y = faceRect->y + (1 + faceRect->height/8);
		noseRect.height = ((faceRect->y) + 7*(faceRect->height)/8) - noseRect.y;
		noseRect.width = (faceRect->x + 3*(faceRect->width/4)) - noseRect.x;
		//sloppy repeat!
		cv::rectangle (*(bgrimg),noseRect, colors[2], 1, 8, 0);


		//decide which side it is on by dividing up the nose region
		if((nose.x - noseRect.x) < (3*noseRect.width)/8){
			//left green
			cv::circle(*(bgrimg),cv::Point(nose.x,nose.y),5,colors[3],3,8,0);
			
		}else if((nose.x - noseRect.x) < (5*noseRect.width)/8){
			//mid blue
			cv::circle(*(bgrimg),cv::Point(nose.x,nose.y),5,colors[4],3,8,0);
			
		}else{
			//right red
			cv::circle(*(bgrimg),cv::Point(nose.x,nose.y),5,colors[5],3,8,0);
			
			//only do expression on front facing?
		}
		//right eye search box is blue, based on nose position.
		cv::Rect rEyeRect(noseRect.x,noseRect.y,noseRect.width/2,noseRect.height/4);
		cv::rectangle (*(bgrimg), rEyeRect, colors[1], 1, 8, 0);
		//right eye is blue
		cv::circle(*(bgrimg),cv::Point(rightEye.x,rightEye.y),2,colors[1],2,8,0);


		//left eye red
		cv::Rect lEyeRect(noseRect.x + noseRect.width/2,noseRect.y,noseRect.width/2,noseRect.height/4);
		cv::rectangle (*(bgrimg), lEyeRect, colors[0], 1, 8, 0);
		//left is red 
		cv::circle(*(bgrimg),cv::Point(leftEye.x,leftEye.y),2,colors[0],2,8,0);
}

unsigned short findNose(const Rect& faceRect, const Mat& depthMat8, Point2i& nosePoint){

	//nose variables
	unsigned short minThresh = 65535;
	unsigned short maxThresh = 0;
	int oneD_index = 0;
		
	//limited search area to center of facebox (1/4 of the whole)
	for(int rowi = (faceRect.height/2); rowi < 7*(faceRect.height)/8;rowi++){
		for(int coli = (faceRect.width/4); coli < 3*(faceRect.width/4);coli++){
			//rowi and coli are internal indexes, they are converted to image co-ordinates during extraction
			oneD_index = ((faceRect.y)+rowi)*depthMat8.cols + coli+faceRect.x;
			//Compare based on mapped value
			if(depthMat8.data[oneD_index] > minThresh && depthMat8.data[oneD_index] != 0){
				minThresh = depthMat8.data[oneD_index];
			}
			//max
			if(depthMat8.data[oneD_index] > maxThresh){
				maxThresh = depthMat8.data[oneD_index];
				nosePoint.y = faceRect.y + rowi;
				nosePoint.x = faceRect.x + coli;
			}
		}
	}
	return maxThresh;
}

//This is always off by a few pixels because of the shape of the eyes on the face
//co-ordinates are relative to cropped head image
void findEyes(const Rect& faceRect, const Mat& depthMat8, Point2i& ptRightEye, Point2i& ptLeftEye){
	int oneD_index = 0;

	unsigned char min = 255;
	int pxCount = 0;
    Rect noseRect;
	noseRect.x = faceRect.x + 1 + (faceRect.width/4);
	noseRect.y = faceRect.y + (1 + faceRect.height/8);
	noseRect.height = ((faceRect.y) + 7*(faceRect.height)/8) - noseRect.y;
	noseRect.width = (faceRect.x + 3*(faceRect.width/4)) - noseRect.x;
	cv::Rect rEyeRect(noseRect.x,noseRect.y,noseRect.width/2,noseRect.height/4);
	
	//right eye, use a smaller area
	for(int ecoli = rEyeRect.x;ecoli < (rEyeRect.x + rEyeRect.width);ecoli++){
			//use eye boundries directly
		for(int erowi = rEyeRect.y;erowi < (rEyeRect.y + rEyeRect.height);erowi++){	
			oneD_index = (erowi*depthMat8.cols) + ecoli;
				//check if it is within 10mm of the nose depth
				if(depthMat8.data[oneD_index] <= min ){
					min = depthMat8.data[oneD_index];
					if(ptRightEye.y == erowi){
						++pxCount;
					}
					else{
						pxCount = 0;
					}
					//including pxCount offset might be helpful if applied to both eyes
					ptRightEye.x = ecoli;
					ptRightEye.y = erowi;
				}
			}
		}//end of right eye search

		min = 255;
		//left eye, start where the right eye search left off, red box
		
		Rect lEyeRect(noseRect.x + noseRect.width/2,noseRect.y,noseRect.width/2,noseRect.height/4);
	
		//CAUTION: NOT SYMMETRIC WITH PREVIOUS SEARCH
		for(int erowi = lEyeRect.y;erowi < (lEyeRect.y + lEyeRect.height);erowi++){
			for(int ecoli = lEyeRect.x;ecoli < (lEyeRect.x + lEyeRect.width);ecoli++){
				oneD_index = (erowi*depthMat8.cols) + ecoli;
				if(depthMat8.data[oneD_index] <= min){
					min = depthMat8.data[oneD_index];
					ptLeftEye.x = ecoli;
					ptLeftEye.y = erowi;
				}
			}
		}//end of left eye search	
}

cv::Point2f partCenter(const VO_FaceParts& faceParts, const VO_Shape& refShape,unsigned int facePartEnum){

	const VO_FacePart& facepart = faceParts.VO_GetOneFacePart(facePartEnum);
	vector<unsigned int> facepart_indexes = facepart.GetIndexes();
	cv::Point2f partSum(0,0);
	for(unsigned int index = 0; index < facepart_indexes.size();++index){
		partSum +=  refShape.GetA2DPoint(facepart_indexes[index]);
	}
	return partSum / facepart_indexes.size();
}

struct VOSM_Fitting_Results{

	vector<VO_Fitting2DSM::DrawMeshInfo> oMeshInfo;
	int iterations;
	float fittingTime;
	VO_Shape finalShape;
	Point2f test_points[3];
	Point2f truth_points[3];

};

bool RunFitting(const VOSM_Fitting_Args& fittingArgs,const Mat& iImage, const bool validRefShape,
				 const VO_Shape& refShape, VO_Fitting2DSM* const fitting2dsm, CFaceDetectionAlgs& fd,
				 const int nbOfPyramidLevels, VOSM_Fitting_Results& output, const bool warpBeforeFitting = true){
	
	vector<Mat> multiChannelMat;
	Mat vjDetImage, drawImage, fittedImage;
	Point2f ptLeftEyeCenter, ptRightEyeCenter, ptMouthCenter;
	//These are only valid if doEvaluation is true.
	Point2f gt_mouthCenter;
	Point2f gt_creye;
	Point2f gt_cleye;

	Rect faceRect;
			
	//Assume the first channel is for face detection
	if(fittingArgs.channels > 1){
		cv::split(iImage,multiChannelMat);
	}
	if((fittingArgs.canidateLoc == VJC || fittingArgs.canidateLoc == MINVJC)){
		vjDetImage = multiChannelMat[0];
	}
	else{
		iImage.copyTo(vjDetImage);
	}
	//iImage.copyTo(fittedImage);
	// Size(240,240)
	fd.FullFaceDetection(   vjDetImage, NULL, true, true, true, true, 1.0,
							Size(80,80),
							Size( min(vjDetImage.rows,vjDetImage.cols), min(vjDetImage.rows,vjDetImage.cols) ) ); 
	if( fd.IsFaceDetected() )
	{
		fd.CalcFaceKeyPoints();
		float tmpScaleX = static_cast<float>(iImage.cols)/static_cast<float>(vjDetImage.cols);
		float tmpScaleY = static_cast<float>(iImage.rows)/static_cast<float>(vjDetImage.rows);
		faceRect = fd.GetDetectedFaceWindow();
		ptLeftEyeCenter = fd.GetDetectedFaceKeyPoint(VO_KeyPoint::LEFTEYECENTER);
		ptRightEyeCenter = fd.GetDetectedFaceKeyPoint(VO_KeyPoint::RIGHTEYECENTER);
		ptMouthCenter = fd.GetDetectedFaceKeyPoint(VO_KeyPoint::MOUTHCENTER);
		ptLeftEyeCenter.x *= tmpScaleX;
		ptLeftEyeCenter.y *= tmpScaleY;
		ptRightEyeCenter.x *= tmpScaleX;
		ptRightEyeCenter.y *= tmpScaleY;
		ptMouthCenter.x *= tmpScaleX;
		ptMouthCenter.y *= tmpScaleY;
		faceRect.x *= static_cast<int>(tmpScaleX);
		faceRect.y *= static_cast<int>(tmpScaleY);
		faceRect.height *= static_cast<int>(tmpScaleY);
		faceRect.width *= static_cast<int>(tmpScaleX);
	}else if(fittingArgs.canidateLoc == MINVJC || fittingArgs.canidateLoc == VJC)
	{
		std::cout << "Face detection failed, trying next face." << std::endl;
		return false;
	}				

	if(validRefShape){
		const VO_FaceParts& facePts = fitting2dsm->GetFaceParts();
		//Find the average of whatever points were given in the shape_info file
		gt_mouthCenter = partCenter(facePts,refShape,VO_FacePart::MOUTHCORNERPOINTS);
		gt_creye = partCenter(facePts,refShape,VO_FacePart::RIGHTEYE);
		gt_cleye = partCenter(facePts,refShape,VO_FacePart::LEFTEYE);
	}

	//These methods assume a certain strcuture of the face that may not always be true.
		// These are not real "face detectors"
	if(fittingArgs.channels > 1 &&
		(fittingArgs.canidateLoc == MINMAX || fittingArgs.canidateLoc == MINVJC))
	{
		//Reduce search area to the middle of the face
		faceRect.width = (2*iImage.cols)/4;
		faceRect.height = (2*iImage.rows)/4;
		faceRect.x = (1*iImage.cols)/4;
		faceRect.y = (1*iImage.rows)/4;
	
		//search only gives pixel resolution, whereas facedetection gives sub-pixel resolution
		Point2i nosePoint;
		findNose(faceRect,multiChannelMat[1],nosePoint);

		Point2i rightEyeLoc, leftEyeLoc;
		if(fittingArgs.canidateLoc == MINMAX)
		{
			findEyes(faceRect,multiChannelMat[1],rightEyeLoc, leftEyeLoc);
			ptRightEyeCenter = rightEyeLoc;
			ptLeftEyeCenter = leftEyeLoc;
		}
		ptMouthCenter = nosePoint;
		//Again, adjusted based on nose distance from mouth & resolution of images used in testing
		// will not hold true for more cases
		ptMouthCenter.y = float(nosePoint.y) + (iImage.rows/8);
	}//end of custom candidate finding

	if(fittingArgs.canidateLoc == GT)
	{
		ptRightEyeCenter = gt_creye;
		ptLeftEyeCenter = gt_cleye;
		ptMouthCenter = gt_mouthCenter;
	}

	// Explained by JIA Pei, you can save to see the detection results.
	iImage.copyTo(drawImage);

	cv::rectangle(drawImage, Point(static_cast<int>(ptLeftEyeCenter.x)-1, static_cast<int>(ptLeftEyeCenter.y)-1),
		Point(static_cast<int>(ptLeftEyeCenter.x)+1, static_cast<int>(ptLeftEyeCenter.y)+1),
		colors[5], 2, 8, 0);
	cv::rectangle(drawImage, Point(static_cast<int>(ptRightEyeCenter.x)-1, static_cast<int>(ptRightEyeCenter.y)-1),
		Point(static_cast<int>(ptRightEyeCenter.x)+1, static_cast<int>(ptRightEyeCenter.y)+1),
		colors[6], 2, 8, 0);
	cv::rectangle(drawImage, Point(static_cast<int>(ptMouthCenter.x)-1, static_cast<int>(ptMouthCenter.y)-1),
		Point(static_cast<int>(ptMouthCenter.x)+1, static_cast<int>(ptMouthCenter.y)+1),
		colors[7], 2, 8, 0);
		//imwrite("drawImage.jpg", drawImage);
		//imwrite("resizedImage.jpg", resizedImage);
				
	fitting2dsm->VO_StartFitting(   iImage,
									output.oMeshInfo,
									fittingArgs.type,
									ptLeftEyeCenter,
									ptRightEyeCenter,
									ptMouthCenter,
									VO_Fitting2DSM::EPOCH, // at most, how many iterations will be carried out
									nbOfPyramidLevels, // read from file AXM\AXM.txt
									fittingArgs.recordResults,
									fittingArgs.fitTechs,
									warpBeforeFitting);
											
	output.iterations = fitting2dsm->GetNbOfIterations();
	output.finalShape = fitting2dsm->VO_GetFittedShape();
	output.fittingTime = fitting2dsm->GetFittingTime();
	output.test_points[0] = ptLeftEyeCenter;
	output.test_points[1] = ptRightEyeCenter;
	output.test_points[2] = ptMouthCenter;
	output.truth_points[0] = gt_cleye;
	output.truth_points[1] = gt_creye;
	output.truth_points[2] = gt_mouthCenter;

	return true;
}

int main(int argc, char **argv)
{
	VOSM_Fitting_Args fittingArgs;
	//default to single channel support
	fittingArgs.fitTech = VO_Fitting2DSM::FIRSTCHANNELONLY;
	fittingArgs.webcamNumber = -1;
	fittingArgs.database =  CAnnotationDBIO::JIAPEI;
    
	cmd_line_parse( argc,
                    argv,
					fittingArgs);

    VO_Fitting2DSM* fitting2dsm = NULL;

	unsigned int nbOfPyramidLevels = 0;

	// :(
    switch(fittingArgs.type)
    {
    case VO_AXM::AAM_BASIC:
    case VO_AXM::AAM_DIRECT:
        fitting2dsm = new VO_FittingAAMBasic();
        dynamic_cast<VO_FittingAAMBasic*>(fitting2dsm)->VO_LoadParameters4Fitting(fittingArgs.trainedData);
		nbOfPyramidLevels = (&(*(VO_AXM*)(&*((*(VO_FittingAAMBasic*)(fitting2dsm)).m_VOAAMBasic))))->GetNbOfPyramidLevels();
        break;
    case VO_AXM::CLM:
    case VO_AXM::AFM:
        fitting2dsm = new VO_FittingAFM();
        dynamic_cast<VO_FittingAFM*>(fitting2dsm)->VO_LoadParameters4Fitting(fittingArgs.trainedData);
		nbOfPyramidLevels = (&(*(VO_AXM*)(&*((*(VO_FittingAFM*)(fitting2dsm)).m_VOAFM))))->GetNbOfPyramidLevels();
        break;
    case VO_AXM::AAM_IAIA:
    case VO_AXM::AAM_CMUICIA:
        fitting2dsm = new VO_FittingAAMInverseIA();
        dynamic_cast<VO_FittingAAMInverseIA*>(fitting2dsm)->VO_LoadParameters4Fitting(fittingArgs.trainedData);
		nbOfPyramidLevels = (&(*(VO_AXM*)(&*((*(VO_FittingAAMInverseIA*)(fitting2dsm)).m_VOAAMInverseIA))))->GetNbOfPyramidLevels();
        break;
    case VO_AXM::AAM_FAIA:
        fitting2dsm = new VO_FittingAAMForwardIA();
        dynamic_cast<VO_FittingAAMForwardIA*>(fitting2dsm)->VO_LoadParameters4Fitting(fittingArgs.trainedData);
		nbOfPyramidLevels = (&(*(VO_AXM*)(&*((*(VO_FittingAAMForwardIA*)(fitting2dsm)).m_VOAAMForwardIA))))->GetNbOfPyramidLevels();
        break;
    case VO_AXM::ASM_LTC:
        fitting2dsm = new VO_FittingASMLTCs();
        dynamic_cast<VO_FittingASMLTCs*>(fitting2dsm)->VO_LoadParameters4Fitting(fittingArgs.trainedData);
		nbOfPyramidLevels = (&(*(VO_AXM*)(&*((*(VO_FittingASMLTCs*)(fitting2dsm)).m_VOASMLTC))))->GetNbOfPyramidLevels();
        break;
    case VO_AXM::ASM_PROFILEND:
        fitting2dsm = new VO_FittingASMNDProfiles();
		((*(VO_TextureModel*)(&(*(VO_AXM*)(&*((*(VO_FittingASMNDProfiles*)(fitting2dsm)).m_VOASMNDProfile)))))).SetNbOfChannels(fittingArgs.channels);
        dynamic_cast<VO_FittingASMNDProfiles*>(fitting2dsm)->VO_LoadParameters4Fitting(fittingArgs.trainedData);
		nbOfPyramidLevels = (&(*(VO_AXM*)(&*((*(VO_FittingASMNDProfiles*)(fitting2dsm)).m_VOASMNDProfile))))->GetNbOfPyramidLevels();
        break;
    }

	unsigned int numModelPts = (*((VO_ShapeModel*)(&(*((VO_TextureModel*)(&(*((VO_AXM*)((*((VO_FittingASMNDProfiles*)(fitting2dsm))).m_VOASMNDProfile))))))))).GetNbOfPoints();
	if(fittingArgs.fitTechs.empty()){
		fittingArgs.fitTechs.resize(numModelPts);
		fill(fittingArgs.fitTechs.begin(),fittingArgs.fitTechs.end(),fittingArgs.fitTech);
	}else if(fittingArgs.fitTechs.size() != numModelPts){
		std::cout << "Cannot fit with subset list of incorrect length" << std::endl;
		exit(EXIT_FAILURE);
	}

    vector<VO_Shape> oShapes;
	static const int nb = 20; //number of pixels away in frequency format for error reporting
    bool doEvaluation = false;
    unsigned int nbOfTestingSamples = fittingArgs.imageFNs.size();
    Mat_<int> nbOfIterations = Mat_<int>::zeros(1, nbOfTestingSamples);
    Mat_<float> deviations;
    Mat_<float> ptsErrorFreq;
    Mat_<float> times = Mat_<float>::zeros(1, nbOfTestingSamples);
	vector<float> ptsErrorAvg;
	if (fittingArgs.annotationFNs.size() == nbOfTestingSamples && fittingArgs.recordResults)
    {
        doEvaluation = true;
        deviations = Mat_<float>::zeros(1, nbOfTestingSamples);
        ptsErrorFreq = Mat_<float>::zeros(nbOfTestingSamples, nb);
    }else{
		std::cout << "Cannot do evaluation with mismatched landmark files" << std::endl;
		exit(EXIT_FAILURE);
	}
    CAnnotationDBIO::VO_LoadShapeTrainingData( fittingArgs.annotationFNs, fittingArgs.database, oShapes);

    CFaceDetectionAlgs fd;
    
	std::string openCv_path(getenv("OPENCV_DIR"));
	fd.SetConfiguration(openCv_path + "/data/lbpcascades/lbpcascade_frontalface.xml", 
						openCv_path + "/data/haarcascades/haarcascade_profileface.xml",
						openCv_path + "/data/haarcascades/haarcascade_mcs_lefteye.xml",
						openCv_path + "/data/haarcascades/haarcascade_mcs_righteye.xml",
						openCv_path + "/data/haarcascades/haarcascade_mcs_nose.xml",
						openCv_path + "/data/haarcascades/haarcascade_mcs_mouth.xml",
				        VO_AdditiveStrongerClassifier::BOOSTING,
				        CFaceDetectionAlgs::FRONTAL );

    Mat iImage;
	
    unsigned int detectionTimes = 0;
    
    // For static images from standard face fittingArgs.databases
    // (Detection only, no tracking) + ASM/AAM
    if(fittingArgs.staticImgs)
    {
        for(unsigned int i = 0; i < fittingArgs.imageFNs.size(); i++)
        {
            iImage = imread(fittingArgs.imageFNs[i]);
			VOSM_Fitting_Results fittingResults;

			//If face can't be found, go to the next face
			if(RunFitting(fittingArgs,iImage,doEvaluation,oShapes[i],fitting2dsm,fd,nbOfPyramidLevels,
				fittingResults))
			{
				detectionTimes++;
				nbOfIterations(0,i) = fittingResults.iterations;
				times(0,i) = fittingResults.fittingTime;
			}
			else
				continue;

			//Now report results
			size_t found1 = fittingArgs.imageFNs[i].find_last_of("/\\");
			size_t found2 = fittingArgs.imageFNs[i].find_last_of(".");
			string prefix = fittingArgs.imageFNs[i].substr(found1+1, found2-1-found1);
            
			if(fittingArgs.recordResults)
			{
				// Inputs: Shapes, an Image, a method to draw meshes from given shapes onto image
				// Outputs: Result file like Save Shape Results, image file
				SaveSequentialResultsInFolder(iImage,oShapes[i],fittingResults.oMeshInfo,prefix);

				string fn = prefix+".jpg";
				if(fittingResults.oMeshInfo.size() > 0)
					imwrite(fn.c_str(), fitting2dsm->GetFittedImage());
			}
			// Computes errors vs annotations
			if(doEvaluation)
			{
				
				vector<float> ptErrorFreq;
				float deviation = 0.0f;
				vector<unsigned int> unsatisfiedPtList;
				vector<float> ptErrorPerPoint(fittingResults.finalShape.GetNbOfPoints(),0.0f);
				unsatisfiedPtList.clear();
				CRecognitionAlgs::CalcShapeFittingEffect(	oShapes[i],
															fittingResults.finalShape,
															deviation,
															ptErrorFreq,
															nb,
															&ptErrorPerPoint);
				deviations(0,i) = deviation;
				for(int j = 0; j < nb; j++)
					ptsErrorFreq(i, j) = ptErrorFreq[j];

				CRecognitionAlgs::SaveFittingResults(	"./",
															prefix,
															deviation,
															ptErrorPerPoint,
															ptErrorFreq,
															fittingResults.finalShape,
															fittingResults.truth_points,
															fittingResults.test_points,
															times(0,i));

				ptsErrorAvg.push_back(std::accumulate(ptErrorPerPoint.begin(),ptErrorPerPoint.end(),0.0f)/ptErrorPerPoint.size());
			}

		} //end image fitting of loop
        
		cout << endl << endl;
        PrintSummary(nb,detectionTimes,nbOfIterations,times,doEvaluation,ptsErrorAvg,deviations, ptsErrorFreq);
        
    }
    else
    {
	    // For dynamic image sequences
		// (Detection or Tracking) + ASM/AAM
        CTrackingAlgs*    trackAlg = new CTrackingAlgs();
        bool isTracked = false;
        detectionTimes = 0;
		if(!fittingArgs.webcamName.empty() || fittingArgs.webcamNumber >= 0){
			VOSM_Fitting_Results liveTrackingResults;
			VideoCapture cap;
			if(fittingArgs.webcamName.empty())
				cap.open(fittingArgs.webcamNumber);
			else
				cap.open(fittingArgs.webcamName);

			if(!cap.isOpened())  // check if we succeeded
				return -1;

			//should read until you kill the program
			while(cap.read(iImage))
			{
				if(!isTracked)
				{
					detectionTimes += RunFitting(fittingArgs,iImage,false,NULL,fitting2dsm,fd,nbOfPyramidLevels,
					liveTrackingResults);

					// Whenever the face is re-detected, initialize the tracker and set isTracked = true;
					Rect rect1 =    liveTrackingResults.finalShape.GetShapeBoundRect();
					trackAlg->UpdateTracker(iImage, rect1);
					isTracked =  true;
				}
				else
				{
            
					RunFitting(fittingArgs,iImage,false,NULL, fitting2dsm, fd, nbOfPyramidLevels,
						liveTrackingResults,false);
					// Explained by JIA Pei. For every consequent image, whose previous image is regarded as tracked, 
					// we have to double-check whether current image is still a tracked one.
					isTracked = CRecognitionAlgs::EvaluateFaceTrackedByProbabilityImage(
															trackAlg,
															iImage,
															liveTrackingResults.finalShape,
															Size(80,80),
															Size( min(iImage.rows,iImage.cols), min(iImage.rows,iImage.cols) ) );
				}
				if(fittingArgs.recordResults)
				{
					string fn = detectionTimes+".jpg";
					imwrite(fn.c_str(), fitting2dsm->GetFittedImage());
				}
			}
			cout << "detection times = " << detectionTimes << endl;
			exit(EXIT_SUCCESS);
		}

        for(unsigned int i = 0; i < fittingArgs.imageFNs.size(); i++)
        {
			VOSM_Fitting_Results fittingResults;
            iImage = imread(fittingArgs.imageFNs[i]);

            size_t found1 = fittingArgs.imageFNs[i].find_last_of("/\\");
            size_t found2 = fittingArgs.imageFNs[i].find_last_of(".");
            string prefix = fittingArgs.imageFNs[i].substr(found1+1, found2-1-found1);

            if(!isTracked)
            {
				detectionTimes += RunFitting(fittingArgs,iImage,doEvaluation,oShapes[i],fitting2dsm,fd,nbOfPyramidLevels,
				fittingResults);

                // Whenever the face is re-detected, initialize the tracker and set isTracked = true;
				Rect rect1 =    fittingResults.finalShape.GetShapeBoundRect();
                trackAlg->UpdateTracker(iImage, rect1);
                isTracked =  true;
            }
            else
            {
            
				RunFitting(fittingArgs,iImage,doEvaluation,oShapes[i], fitting2dsm, fd, nbOfPyramidLevels,
					fittingResults,false);
                // Explained by JIA Pei. For every consequent image, whose previous image is regarded as tracked, 
                // we have to double-check whether current image is still a tracked one.
//                isTracked = true;
                isTracked = CRecognitionAlgs::EvaluateFaceTrackedByProbabilityImage(
                                                        trackAlg,
                                                        iImage,
														fittingResults.finalShape,
                                                        Size(80,80),
                                                        Size( min(iImage.rows,iImage.cols), min(iImage.rows,iImage.cols) ) );
            }

			if(fittingArgs.recordResults)
            {
                string fn = prefix+".jpg";
                imwrite(fn.c_str(), fitting2dsm->GetFittedImage());
            }
            
            // For evaluation
            if(doEvaluation)
            {
                vector<float> ptErrorFreq;
                float deviation = 0.0f;
                vector<unsigned int> unsatisfiedPtList;
                unsatisfiedPtList.clear();
                CRecognitionAlgs::CalcShapeFittingEffect(   oShapes[i],
															fittingResults.finalShape,
                                                            deviation,
                                                            ptErrorFreq,
                                                            nb);
                deviations(0,i) = deviation;
                for(int j = 0; j < nb; j++)
                    ptsErrorFreq(i, j) = ptErrorFreq[j];
                CRecognitionAlgs::SaveShapeRecogResults(    "./",
                                                            prefix,
                                                            deviation,
                                                            ptErrorFreq);
            }
        }

        cout << endl << endl;
        PrintSummary(nb,detectionTimes,nbOfIterations,times,doEvaluation,ptsErrorAvg,deviations, ptsErrorFreq);
        
		delete trackAlg;
    }
    
    delete fitting2dsm;

    return 0;
}
