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


#ifndef __VO_SHAPEMODEL_H__
#define __VO_SHAPEMODEL_H__


#include <vector>
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv/cvaux.h"

#include "VO_Edge.h"
#include "VO_Shape.h"
#include "VO_Triangle2DStructure.h"
#include "VO_Shape2DInfo.h"
#include "VO_FaceParts.h"
#include "VO_Point2DDistributionModel.h"

using namespace std;
using namespace cv;


/** 
* @author        JIA Pei
* @brief        Statistical shape model.
*/
class VO_ShapeModel
{
friend class VO_Fitting2DSM;
friend class VO_FittingAAMBasic;
friend class VO_FittingAAMForwardIA;
friend class VO_FittingAAMInverseIA;
friend class VO_FittingASMLTCs;
friend class VO_FittingASMNDProfiles;
protected:
    /** PCA transform for shape, including eigenvectors, eigenvalues, and mean */
    PCA                             m_PCAAlignedShape;

    /** Number of training samples. For IMM: 240 */
    unsigned int                    m_iNbOfSamples;

    /** 2D or 3D, 116/58 = 2 */
    unsigned int                    m_iNbOfShapeDim;

    /** Number of points to describe per shape. For IMM: 58 */
    unsigned int                    m_iNbOfPoints;

    /** length of shape vector in format of xxx...yyy.... For IMM: 58*2=116 */
    unsigned int                    m_iNbOfShapes;

    /** Shape eigens at most before PCA. For IMM: min (116, 240) = 116 */
    unsigned int                    m_iNbOfEigenShapesAtMost;

    /** Number of shape model eigens. For IMM: 21 */
    unsigned int                    m_iNbOfShapeEigens;

    /** Number of edges */
    unsigned int                    m_iNbOfEdges;

    /** Number of triangles */
    unsigned int                    m_iNbOfTriangles;

    /** Reference shape which is of size close to "1", but not guaranteed */
    VO_Shape                        m_VOAlignedMeanShape;

    /** Reference shape which is scaled back to the original size and translated to (left top most = origin) */
    VO_Shape                        m_VOReferenceShape;

    /** The reference shape average size : 582.425 */
    float                           m_fAverageShapeSize;

    /** Truncate Percentage for shape model PCA. Normally, 0.95 */
    float                           m_fTruncatedPercent_Shape;

    /** All loaded shapes in a vector format. For IMM, 240*(58*2) */
    vector<VO_Shape>                m_vShapes;

    /** All aligned shapes in a vector format. For IMM, 240*(58*2) */
    vector<VO_Shape>                m_vAlignedShapes;

    /** shape information */
    vector<VO_Shape2DInfo>          m_vShape2DInfo;

    /** face parts information, seems to have the duplicate information as m_vShape2DInfo, but not! */
    VO_FaceParts                    m_FaceParts;

    /** Edges in the template shape, only contain the index pairs. For IMM,  152 */
    vector<VO_Edge>                 m_vEdge;

    /** Unnormalized triangles in the template shape. For IMM, 95 */
    vector<VO_Triangle2DStructure>  m_vTemplateTriangle2D;

    /** Normalized triangles in the template shape - just replace the vertexes in m_vTemplateTriangle2D by corresponding vertexes of m_VOAlignedMeanShape. For IMM, 95 */
    vector<VO_Triangle2DStructure>  m_vNormalizedTriangle2D;
    
    /** Normalized Point distribution model */
    VO_Point2DDistributionModel     m_VOPDM;
    
    /** Initialization */
    void                            init();

public:
    /** Default constructor to create a VO_ShapeModel object */
    VO_ShapeModel();

    /** Destructor */
    virtual ~VO_ShapeModel();

    /** Align all shapes */
    static float                    VO_AlignAllShapes(const vector<VO_Shape>& vShapes, vector<VO_Shape>& alignedShapes);

    /** Rescale all aligned shapes */
    static void                     VO_RescaleAllAlignedShapes(const VO_Shape& meanAlignedShape, vector<VO_Shape>& alignedShapes);

    /** Rescale one aligned shape */
    static void                     VO_RescaleOneAlignedShape(const VO_Shape& meanAlignedShape, VO_Shape& alignedShape);

    /** Returns the mean shape of all shapes */
    static void                     VO_CalcMeanShape(const vector<VO_Shape>& vShapes, VO_Shape& meanShape);

    /** Judge whether the point "pt" is in convex hull "ch" */
    static bool                     VO_IsPointInConvexHull(const Point2f pt, const Mat_<float>& ch, bool includingBoundary);

    /** Judge whether edge indexed by (ind1+ind2) is already in the vector of "edges" */
    static bool                     VO_EdgeHasBeenCounted(const vector<VO_Edge>& edges, unsigned int ind1, unsigned int ind2);

    /** Judge whether triangle t is already in the vector of "triangles" */
    static bool                     VO_TriangleHasBeenCounted(const vector<VO_Triangle2DStructure>& triangles, const vector<unsigned int>& t);

    /** Build Edges */
    static unsigned int             VO_BuildEdges(const VO_Shape& iShape, const CvSubdiv2D* Subdiv, vector<VO_Edge>& outEdges);

    /** Build triangles */
    static unsigned int             VO_BuildTriangles(const VO_Shape& iShape, const vector<VO_Edge>& edges, vector<VO_Triangle2DStructure>& outTriangles);

    /** Build Delaunay triangulation mesh */
    static void                     VO_BuildTemplateMesh(const VO_Shape& iShape, vector<VO_Edge>& edges, vector<VO_Triangle2DStructure>& triangles);

    /** Build triangulation mesh according to shapeInfo */
    static void                     VO_BuildTemplateMesh(const VO_Shape& iShape, const VO_FaceParts& iFaceParts, vector<VO_Edge>& edges, vector<VO_Triangle2DStructure>& triangles);

    /** Judge whether the shape is inside an image */
    static bool                     VO_IsShapeInsideImage(const VO_Shape& iShape, const Mat& img);

    /** Reference shape inShape back to aligned outShape */
    static void                     VO_ReferenceShapeBack2Aligned(const VO_Shape& inShape, float shapeSD,VO_Shape& outShape);

    /** Calculate the bounding rectangle from a list of rectangles */
    static Rect                     VO_CalcBoundingRectFromTriangles(const vector <VO_Triangle2DStructure>& triangles);

    /** Shape parameters constraints */
    void                            VO_ShapeParameterConstraint(Mat_<float>& ioP, float nSigma = 4.0f);

    /** Shape projected to shape parameters*/
    void                            VO_AlignedShapeProjectToSParam(const VO_Shape& iShape, Mat_<float>& outP) const;

    /** Shape parameters back projected to shape */
    void                            VO_SParamBackProjectToAlignedShape(const Mat_<float>& inP, VO_Shape& oShape, int dim = 2) const;
    void                            VO_SParamBackProjectToAlignedShape(const Mat_<float>& inP, Mat_<float>& oShapeMat) const;

    /** shape -> Procrustes analysis -> project to shape parameters */
    void                            VO_CalcAllParams4AnyShapeWithConstrain(const Mat_<float>& iShape, Mat_<float>& oShape, Mat_<float>& outP, float& norm, vector<float>& angles, Mat_<float>& translation );
    void                            VO_CalcAllParams4AnyShapeWithConstrain(VO_Shape& ioShape, Mat_<float>& outP, float& norm, vector<float>& angles, Mat_<float>& translation );
    //void                          VO_CalcAllParams4AnyShapeWithConstrain(const Mat_<float>& iShape, Mat_<float>& oShape, Mat_<float>& outP, Mat_<float>& outQ );
    //void                          VO_CalcAllParams4AnyShapeWithConstrain(VO_Shape& ioShape, Mat_<float>& outP, Mat_<float>& outQ );
    //void                          VO_BuildUpShapeFromRigidNonRigidParams(const Mat_<float>& inP, const Mat_<float>& inQ, VO_Shape& oShape );

    /** Build Shape Model */
    void                            VO_BuildShapeModel( const vector<string>& allLandmarkFiles4Training,
                                                        const string& shapeinfoFileName, 
                                                        unsigned int database,
                                                        float TPShape = 0.95, 
                                                        bool useKnownTriangles = false);

    /** Save Shape Model, to a specified folder */
    void                            VO_Save(const string& fd);

    /** Load all parameters */
    void                            VO_Load(const string& fd);

    /** Load parameters for fitting */
    void                            VO_LoadParameters4Fitting(const string& fd);

    /** Gets and Sets */
    Mat_<float>                     GetAlignedShapesMean() const {return this->m_PCAAlignedShape.mean;}
    Mat_<float>                     GetAlignedShapesEigenValues() const {return this->m_PCAAlignedShape.eigenvalues;}
    Mat_<float>                     GetAlignedShapesEigenVectors() const {return this->m_PCAAlignedShape.eigenvectors;}
    unsigned int                    GetNbOfSamples() const {return this->m_iNbOfSamples;}
    unsigned int                    GetNbOfShapeDim() const {return this->m_iNbOfShapeDim;}
    unsigned int                    GetNbOfPoints() const {return this->m_iNbOfPoints;}
    unsigned int                    GetNbOfShapes() const {return this->m_iNbOfShapes;}
    unsigned int                    GetNbOfEigenShapesAtMost() const {return this->m_iNbOfEigenShapesAtMost;}
    unsigned int                    GetNbOfShapeEigens() const {return this->m_iNbOfShapeEigens;}
    unsigned int                    GetNbOfEdges() const {return this->m_iNbOfEdges;}
    unsigned int                    GetNbOfTriangles() const {return this->m_iNbOfTriangles;}
    VO_Shape                        GetAlignedMeanShape() const { return this->m_VOAlignedMeanShape;}
    VO_Shape                        GetReferenceShape() const {return this->m_VOReferenceShape;}
    float                           GetAverageShapeSize() const {return this->m_fAverageShapeSize;}
    float                           GetTruncatedPercent_Shape() const {return this->m_fTruncatedPercent_Shape;}
    vector<VO_Shape>                GetShapes() const {return this->m_vShapes;}
    vector<VO_Shape>                GetAlignedShapes() const {return this->m_vAlignedShapes;}
    vector<VO_Shape2DInfo>          GetShapeInfo() const {return this->m_vShape2DInfo;}
    VO_FaceParts                    GetFaceParts() const {return this->m_FaceParts;}
    vector<VO_Edge>                 GetEdge() const {return this->m_vEdge;}
    vector<VO_Triangle2DStructure>  GetTriangle2D() const {return this->m_vTemplateTriangle2D;}
    vector<VO_Triangle2DStructure>  GetNormalizedTriangle2D() const {return this->m_vNormalizedTriangle2D;}

//    void                            SetAlignedShapesEigenVectors(const Mat_<float>& inAlignedShapesEigenVectors)
//                                    {inAlignedShapesEigenVectors.copyTo(this->m_PCAAlignedShape.eigenvectors);}
//    void                            SetAlignedShapesEigenValues(const Mat_<float>& inAlignedShapesEigenValues)
//                                    {inAlignedShapesEigenValues.copyTo(this->m_PCAAlignedShape.eigenvalues);}
//    void                            SetNbOfSamples(unsigned int iNbOfSamples) {this->m_iNbOfSamples = iNbOfSamples;}
//    void                            SetNbOfPoints(unsigned int iNbOfPoints) {this->m_iNbOfPoints = iNbOfPoints;}
//    void                            SetNbOfShapes(unsigned int iNbOfShapes) {this->m_iNbOfShapes = iNbOfShapes;}
//    void                            SetNbOfShapeDim(unsigned int iNbOfShapeDim) {this->m_iNbOfShapeDim = iNbOfShapeDim;}
//    void                            SetNbOfTruncatedShapes(unsigned int iNbOfTruncatedShapes)
//                                    {this->m_iNbOfShapeEigens = iNbOfTruncatedShapes;}
//    void                            SetNbOfEigenShapesAtMost(unsigned int iNbOfEigenShapesAtMost)
//                                    {this->m_iNbOfEigenShapesAtMost = iNbOfEigenShapesAtMost;}
//    void                            SetReferenceShape(const VO_Shape& iReferenceShape)
//                                    {this->m_VOReferenceShape = iReferenceShape;}
//    void                            SetAverageShapeSize(float inAverageShapeSize) {this->m_fAverageShapeSize = inAverageShapeSize;}
//    void                            SetTruncatedPercent_Shape(float inTP) {this->m_fTruncatedPercent_Shape = inTP;}
//    void                            SetShapes(const vector<VO_Shape>& iShapes) {this->m_vShapes = iShapes;}
//    void                            SetAlignedShapes(const vector<VO_Shape>& iAlignedShapes)
//                                    {this->m_vAlignedShapes = iAlignedShapes;}
//    void                            SetShapeInfo(const vector<VO_Shape2DInfo>& iShape2DInfo)
//                                    {this->m_vShape2DInfo = iShape2DInfo;}
//    void                            SetEdge(const vector<VO_Edge>& iEdge) {this->m_vEdge = iEdge;}
//    void                            SetTriangle2D(const vector<VO_Triangle2DStructure>& iTriangle2D)
//                                    {this->m_vTemplateTriangle2D = iTriangle2D;}
//    void                            SetNormalizedTriangle2D(const vector<VO_Triangle2DStructure>& iNormalizedTriangle2D)
//                                    {this->m_vNormalizedTriangle2D = iNormalizedTriangle2D;}
};

#endif // __VO_SHAPEMODEL_H__

