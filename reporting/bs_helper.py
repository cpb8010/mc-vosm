# bs_helpers.py
# This file has all the Boshorous specific parsing functions
import numpy
import cv2
import os
import struct

from collections import Counter
from collections import defaultdict 

def read_BSlm2(filepath):
    """ Read types and values of points from .lm2 file
        
        Requires exact whitespace format in .lm2 file. :(
        
        Arguments: filepath
            Filepath -- String containing the path to and name of face without extension, 
            e.g. "../Bosphorous/bs000/bs000_LFAU_9_0"
        
        Returns: myText, myPoints
            myText -- List of strings containing the point annotation
            myPoints -- List of strings containing the point values 
            
    """        
    fid = open('%s.lm2' % (filepath), 'r')
    if(fid == -1):
        print('Could not open file', filepath)
        return
    
    #2D Landmark File
    fid.readline() 
    #empty line
    fid.readline() 
    landmarkString = fid.readline()
    landmarkNumber = int(landmarkString.split(' ')[0])
    
    while fid.readline().rstrip('\n') != "Labels:":
        pass
        
    myText = [fid.readline().rstrip('\n') for landmarkName in range(landmarkNumber)]
    
    while not fid.readline().rstrip('\n') == "2D Image coordinates:":
        pass
        
    myPoints = [(fid.readline().rstrip('\n').split(' ')) for pointValue in range(landmarkNumber)]
    
    fid.close()
    return myText, myPoints
    
def ConvertFaces(db_path, faces, picSize, faceSize, landmarkList):
    """ Load images and points and convert into equally sized images
    
    Arguments: db_path, faces, picSize, faceSize
        db_path -- root of BSDB
        faces -- Dict with keys that contain lists of face names
        picSize -- output image resolution (square)
        facesize -- non-square maxiumum dimension of the face
        landmarkList -- used to remove points
    
    Returns: a dictionary with keys as names, and a list of values
        The list of values lists of points, colorimg, depthimg, grayimg
    """
    convFaceDict = defaultdict(defaultdict)  
    for faceType in faces.keys():
        #print("faceType",faceType)
        typeSubjectDict = defaultdict(list)
        for faceName in faces[faceType]:
            #print("faceName",faceName)
            facePath = '%s%s/%s' % (db_path, faceName.split('_')[0], faceName)
            lm2txt, lm2pts = read_BSlm2(facePath)
            filteredTxt = [txtLine for txtLine in lm2txt if txtLine in landmarkList]
            filteredPts = [ptRow for ln, ptRow in enumerate(lm2pts) if lm2txt[ln] in landmarkList]
            #color image
            #print(faceName)
            imageName = '%s.png' % (facePath)
            fullColorFace = cv2.imread(imageName)
            fullGrayFace = cv2.cvtColor(fullColorFace,cv2.cv.CV_RGB2GRAY)
            #DEBUG
            # for ptr, ptl in filteredPts:
                # #print('pt[1,:]',pt[1,:])
                # print('ptl, ptr',ptl, ptr)
                # cv2.circle(fullColorFace, (int(float(ptr)), int(float(ptl))), 2, (0, 0, 255)) 
            # cv2.imshow('Color Image w/ pts', fullColorFace)
            # cv2.waitKey()
            #depth image
            df = open('%s.bnt' % (facePath), 'rb')
            nrows = struct.unpack('H', df.read(2))[0]
            ncols = struct.unpack('H', df.read(2))[0]
            zmin = struct.unpack('d', df.read(8))[0]
            nameLen = struct.unpack('H', df.read(2))[0]
            imfile = struct.unpack('B'*nameLen, df.read(1*nameLen))
            #print("imfile", "".join(map(chr,imfile)))
            dataLen = struct.unpack('I', df.read(4))[0]
            dataRows = dataLen / 5
            #Must unpack in order, even if data is left unused.
            imdata_1 = struct.unpack('d'*dataRows, df.read(8*dataRows))
            imdata_2 = struct.unpack('d'*dataRows, df.read(8*dataRows))
            imdata_3 = struct.unpack('d'*dataRows, df.read(8*dataRows))
            imdata_4 = struct.unpack('d'*dataRows, df.read(8*dataRows))
            imdata_5 = struct.unpack('d'*dataRows, df.read(8*dataRows))
            df.close()
            
            
            #following matlab script (line 51)
            bsdz = numpy.array(imdata_3)
            bsdz_noBackground = bsdz[bsdz > zmin]
            
            #keep 2d locations
            bsdz_min = min(bsdz_noBackground)
            #shifts min to 0, scales max to 255
            bsdz_shift = bsdz - bsdz_min
            #speeds up next
            bsdz_max = max(bsdz_shift) 
            bsdz_scale = (255/bsdz_max) * bsdz_shift
            
            bsdz_trunk = numpy.array(bsdz_scale.clip(0,255),numpy.uint8)
            
            #reverse and eliminate background
            #bsdz_rever = 255 - bsdz_trunk
            #reshape to 2d image
            bsdz_resha = numpy.flipud(numpy.reshape(bsdz_trunk,(nrows,ncols)))
            
            aspectRatio = float(fullGrayFace.shape[1])/float(fullGrayFace.shape[0])
            #resize and preserve aspect ratio when scaling face to faceSize
            if(fullGrayFace.shape[0] > fullGrayFace.shape[1] and bsdz_resha.shape[0] > bsdz_resha.shape[1]):
                #face is taller than wide
                #scale to keep aspect ratio
                bsdz_resiz = cv2.resize(bsdz_resha,(int(aspectRatio*float(faceSize)),faceSize))
                #the dimensions of the 3 images need to be identical
                bsg_resize = cv2.resize(fullGrayFace,(bsdz_resiz.shape[1],faceSize))
                bsc_resize = cv2.resize(fullColorFace,(bsdz_resiz.shape[1],faceSize))
                
            
            elif(fullGrayFace.shape[0] < fullGrayFace.shape[1] and bsdz_resha.shape[0] < bsdz_resha.shape[1]):      
                #face is wider than tall                
                bsdz_resiz = cv2.resize(bsdz_resha,(faceSize,int(aspectRatio*float(faceSize))))
                #force equal sizes, should be close aspect ratio anyways
                bsg_resize = cv2.resize(fullGrayFace,(faceSize,bsdz_resiz.shape[0]))
                bsc_resize = cv2.resize(fullColorFace,(faceSize,bsdz_resiz.shape[0]))
                            
            else:
                #square?
                print("A terrible calmity has occured")
                print(float(fullGrayFace.shape[0])/float(fullGrayFace.shape[1]), float(bsdz_resha.shape[0])/float(bsdz_resha.shape[1]))
                assert(fullGrayFace.shape[0]/fullGrayFace.shape[1] - bsdz_resha.shape[0]/bsdz_resha.shape[1] == 0)
                bsdz_resiz = cv2.resize(bsdz_resha,(faceSize,faceSize))
                bsg_resize = cv2.resize(fullGrayFace,(faceSize,faceSize))
                bsc_resize = cv2.resize(fullColorFace,(faceSize,faceSize))
                
            assert(bsg_resize.shape == bsdz_resiz.shape)
            assert(bsc_resize.shape[0] == bsdz_resiz.shape[0] and bsc_resize.shape[1] == bsdz_resiz.shape[1])
                       
           
            #place in center of picSize x picSize image
            bs_ysize = bsg_resize.shape[0]
            bs_xsize = bsg_resize.shape[1]
            startInsertx = numpy.round((picSize/2) - (bs_xsize/2))
            startInserty = numpy.round((picSize/2) - (bs_ysize/2))
            endInsertx = startInsertx + bs_xsize
            endInserty = startInserty + bs_ysize
            faceDepth = numpy.zeros((picSize, picSize), numpy.uint8)
            faceGray = numpy.zeros((picSize, picSize), numpy.uint8)
            faceColor = numpy.zeros((picSize, picSize,3), numpy.uint8)
            faceDepth[startInserty:endInserty,startInsertx:endInsertx] = bsdz_resiz
            faceGray[startInserty:endInserty,startInsertx:endInsertx] = bsg_resize
            faceColor[startInserty:endInserty,startInsertx:endInsertx,:] = bsc_resize
            

            #Update 2d annotations
            #shift scale points to image
            y_Scale = float(bsg_resize.shape[0]) / float(fullGrayFace.shape[0])
            x_Scale = float(bsg_resize.shape[1]) / float(fullGrayFace.shape[1])
            lm2pts_x = [(float(mypt[0]) * x_Scale) + startInsertx for mypt in filteredPts] 
            lm2pts_y = [(float(mypt[1]) * y_Scale) + startInserty for mypt in filteredPts]
            
            pts = numpy.vstack((lm2pts_x,lm2pts_y))
            
            #DEBUG
            # for ptr, ptl in pts.T:
                # cv2.circle(faceColor, (int(ptl), int(ptr)), 1, (0, 255, 0)) 
            # cv2.imshow('Resized Depth Image', faceDepth)
            # cv2.imshow('Resized Color Image', faceColor)
            # cv2.imshow('Resized Gray Image', faceGray)
            # cv2.waitKey()
            
            typeSubjectDict[faceName].append(pts)
            typeSubjectDict[faceName].append(faceColor)
            typeSubjectDict[faceName].append(faceGray)
            typeSubjectDict[faceName].append(faceDepth)
            typeSubjectDict[faceName].append(filteredTxt)
            
        convFaceDict[faceType] = typeSubjectDict
        
    return convFaceDict

def SortSubjectList(sl):
    """ Flattens list into grouped, sorted dict
    
        Arguments: sl
            sl -- sorted list of strings that have 3 _'s
        
        Returns: subDict
            subDict -- defaultdict(list) with keys as the string between the 2 outer _'s
    """
    subDict = defaultdict(list)    
    for sub in sl:
        for face in sub:
            faceType = "_".join(face.split('_')[1:3])
            subDict[faceType].append(face)   
    return subDict
    
def FilterSubjects (db_path, landmarkList):
    """ Search the db_path and return sufficiently annotated subjects
    
        Arguments:
            db_path -- Path to Bosphorous DB location
            landmarkList -- List of strings to require for loaded faces
            
        Returns:
            subjectList -- a list of subjects(folders) that contain a list of face names (unique filenames without extensions)
    
    """
    #Inner helper function to check files for landmarks
    def pointsExist(face):

        lm2_f = open('%s%s/%s.lm2' % (db_path, face.split('_')[0], face),'r')
        lm3_f = open('%s%s/%s.lm3' % (db_path, face.split('_')[0], face),'r')
        lm2_s = lm2_f.read()
        lm3_s = lm3_f.read()
        lm2_f.close()
        lm3_f.close()
        #check that the files contain exactly 1 of each listed landmark
        facePass = False
        for landmarkName in landmarkList:
            lm2_lmf = landmarkName in lm2_s
            lm3_lmf = landmarkName in lm3_s
            if lm2_lmf and lm3_lmf:
               facePass = True
            elif lm2_lmf:
               print("lm2 missing points")
               facePass = False
               break
            elif lm3_lmf:
                print("lm3 missing points")
                facePass = False
                break
            else:
                facePass = False
                break
                
        #could just return in loop, but whatever        
        return facePass
        
    # this should be a lamda, too small for a function
    def FaceName(fileName):
        return fileName.split('.')[0]
        
    subjectList = []    
    for root, dirs, files in os.walk(db_path):
        # print(len(files))
        #only search leaf directories
        if(len(dirs) <= 0):
            faceMap = (map(FaceName, files))
            faceCounter = Counter(faceMap)
            faceSet = set(faceMap)
            #remove incomplete faces (faces that dont have 4 files)
            for d_key, d_val in faceCounter.iteritems():
                if(d_val != 4):
                    faceSet.remove(d_key)
                    # print(d_key)
            # Discard images that don't have the requisite annotations       
            filteredFaceSet = filter(pointsExist,faceSet)
                   
            subjectList.append(filteredFaceSet)   
    return subjectList
    
def WriteGroup(dir_root,fold,groups,convertedFaces):
    """ Writes 3 channel file and points to folders
    
        dir_root -- Existing writable directory to create folders
        fold -- subject list to write from faces
        groups -- folder names to create and populate
        convertedFaces -- defaultdict:defaultdict:list with types:facenames:(points,cImg,gImg,dImg,annotations)
        """
    for group in groups:
        groupFolder = "%s/%s" % (dir_root,group)
        if(not os.path.exists(groupFolder)):
            os.makedirs(groupFolder)

        faceType = convertedFaces[group]
        #check to combine groups
        if(len(faceType) == 0):
            #print(type(faceType))
            startMatchKeys = [fullType for fullType in convertedFaces.keys() if fullType.startswith(group)]
            #print(startMatchKeys)
            
            for matchKey in startMatchKeys:
                faceType.update(convertedFaces[matchKey])
            
        
        for faceSub in fold:
            for face in faceType.keys():
                sub = face.split('_')[0]
                if(sub == faceSub):
                   
                    #found a face that should be saved
                    fgdg = numpy.dstack((faceType[face][2],faceType[face][3],faceType[face][2]))
                    facepath = '%s/%s' % (groupFolder,face)
                    cv2.imwrite('%s.png' % facepath,fgdg)
                    #write points
                    txtheader = 'version: 1\nn_points: %s\n{' % faceType[face][0].shape[1]
                    numpy.savetxt('%s.pts' % facepath,numpy.transpose(faceType[face][0]),fmt='%3.3f',delimiter=' ',header=txtheader,footer='}',newline='\n',comments='')
                