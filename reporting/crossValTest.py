#crossValTest.py
import pickle
import numpy
import cv2
import os

from sklearn.cross_validation import train_test_split

# General utility functions for VOSM arguments and results
import vosm_helper
# Specific functions designed to run VOSM on bosphorous db
import bs_helper

numFolds = 10
# Root directory for storing script results
root_path = 'K:/CrossVerify/'
vosm_path = 'C:/data/vosm-0.3.3/Debug/'
# Root directory of image database
db_path = 'C:/data/BosphorusDB/' #this is the real deal, the full set!
# Changing these paths will require cause scraping and conversion steps to be redone
subjectPkl_path = os.path.join(root_path,'subjectList.pkl')
sortedPkl_path = os.path.join(root_path,'sortedFaces.pkl')
convertedPkl_path = os.path.join(root_path,'convertedFaces.pkl')
allFoldsDB_path = os.path.join(root_path,'allResults.sqlite')

#Convert Image Parameters.
# (changing requires deleting convertedPkl_path)
PICSIZE = 100
FACESIZE = 60
#Building parameters
shape_info_path = os.path.join(root_path,'shape_info_bs16.txt')
sm_building_path = os.path.join(vosm_path,"test_smbuilding.exe")
sm_building_args = ' -t ASM_PROFILEND -d JIAPIA -l 1 -c 2'

#Fitting/Testing parameters
sm_fitting_path = os.path.join(vosm_path,"test_smfitting.exe")
sm_subset_yaml_path = os.path.join(root_path,'subset.yaml')
sm_fitting_args = '-y "%s"' % sm_subset_yaml_path

channelTypes = ['default', 'gray', 'depth', 'pyr', 'sub']
locationTypes = ['min-max','VJc', 'minVJc', 'GT']
#Landmarks to require.
# (changing requires deleting all pickles)
landmarkList = [
'Outer left eyebrow',     #0
'Middle left eyebrow',    #1
'Inner left eyebrow',     #2
'Inner right eyebrow',    #3
'Middle right eyebrow',   #4
'Outer right eyebrow',    #5
'Outer left eye corner',  #6
'Inner left eye corner',  #7 
'Inner right eye corner', #8
'Outer right eye corner', #9
'Nose tip',               #10
'Left mouth corner',      #11
'Upper lip outer middle', #12
'Right mouth corner',     #13
'Lower lip inner middle', #14
'Chin middle',            #15
];

print(root_path)

#Restart from last saved point
subjectList = vosm_helper.Load(subjectPkl_path)
sortedFaces = vosm_helper.Load(sortedPkl_path)
convertedFaces = vosm_helper.Load(convertedPkl_path)
allFoldsDB = vosm_helper.LoadDB(allFoldsDB_path)

# Check for load failures to prevent duplicate work
if( not subjectList):
    print("Filtering Subjects")
    subjectList = bs_helper.FilterSubjects(db_path,landmarkList)
    vosm_helper.Save(subjectList,subjectPkl_path)
    
if( not sortedFaces):
    print("Sorting Faces")
    sortedFaces = bs_helper.SortSubjectList(subjectList)
    vosm_helper.Save(sortedFaces,sortedPkl_path)
   
if( not convertedFaces):
    print("Converting Faces")
    convertedFaces = bs_helper.ConvertFaces(db_path, sortedFaces,PICSIZE,FACESIZE,landmarkList) 
    vosm_helper.Save(convertedFaces,convertedPkl_path)
    
if( not allFoldsDB):
    print("Creating Master Result database")
    allFoldsDB = vosm_helper.CreateMasterDB(allFoldsDB_path)
    
# Cross-Validation folds
resultPathList = list()
for foldNum in range(numFolds):
    fold_name = 'Set_%d' % foldNum
    print(fold_name)
    
    trainTestSplitPkl_path = os.path.join(root_path,fold_name,'trainTestSplit.pkl')
    test_image_path = os.path.join(root_path,fold_name,'Faces/Test/')
    train_image_path = os.path.join(root_path,fold_name,'Faces/Train/')
    build_models_path = os.path.join(root_path,fold_name,'Models/')
    test_results_path = os.path.join(root_path,fold_name,'Results/')
    resultsDB_path = os.path.join(root_path,fold_name,'testResults.sqlite')

    resultPathList.append(resultsDB_path)
    
    trainTestSplit = vosm_helper.Load(trainTestSplitPkl_path)
    resultDB = vosm_helper.LoadDB(resultsDB_path)
    
    if( not trainTestSplit):
        print("Splitting faces into training and test folds")
        #pull out folder names
        subjectFolderList = [sList[0].split('_')[0] for sList in subjectList]
        # Randomly break subjects into train / test
        train_fold, test_fold = train_test_split(subjectFolderList)
        #test categories from type (sortedFaces.keys())
        test_groups = sortedFaces.keys()
        #train categories combines many test categories
        train_groups = list(set([faceType.split('_')[0] for faceType in sortedFaces.keys()]))
        #train_groups.append('')  #for all categories?
        trainTestSplit = (train_fold, test_fold, test_groups, train_groups)
            
        if( not os.path.exists(os.path.join(root_path,fold_name))):
            os.makedirs(os.path.join(root_path,fold_name))
            
        vosm_helper.Save(trainTestSplit,trainTestSplitPkl_path)
    else:
        #required for many later steps
        train_fold, test_fold, test_groups, train_groups = trainTestSplit

    if( not os.path.exists(test_image_path) ):
        print("Writing Test Files")
        bs_helper.WriteGroup(test_image_path, test_fold,test_groups, convertedFaces)
        
    if( not os.path.exists(train_image_path) ):
        print("Writing Train Files")
        bs_helper.WriteGroup(train_image_path, train_fold,train_groups, convertedFaces)

    if( not os.path.exists(build_models_path)):
        print("Building VOSM Models")
        allBuildsSucceeded = vosm_helper.BuildModels(build_models_path, train_image_path, shape_info_path, sm_building_path, sm_building_args, train_groups)

        if(not allBuildsSucceeded):
            # This is a serious error, but it should be handeled by later steps, so a warning is appropriate
            print("WARNING: At least one VOSM model failed building!!")
        
    if( not os.path.exists(test_results_path)):
        print("Testing VOSM Models")
        allTestsSucceeded = vosm_helper.TestModels(test_results_path, sm_fitting_path, locationTypes, build_models_path, train_groups,  channelTypes, test_image_path, test_groups, sm_fitting_args )
        
        if(not allTestsSucceeded):
            # This could be caused by failed models or by errors/exceptions in VOSM
            print("WARNING: At least one VOSM test failed during run!!")
            
    if( not resultDB):
        print("Creating ResultDB")
        resultDB = vosm_helper.CreateDB(resultsDB_path,len(landmarkList))
        print("Scraping and storing results")
        vosm_helper.StoreResults(resultDB,test_results_path,len(landmarkList),foldNum)
        resultDB.close()
    
    # Collect and join all the resultDB's
    print("Adding results to master DB")
    vosm_helper.MergeDB(allFoldsDB,resultsDB_path,foldNum)

#These two dictionaries are used as references in graphing
resultTypeDict = {
'detTimes': ('Detection Times','Faces fitted'),
'avgIterations':('Average Number of Iterations','Saved stages'),
'avgFitTime':('Average Detection time','milliseconds'),
'avgDist':('Average Pt Distance','Euclidian distance in pixels'),
'avgDev':('Average Deviation of Errors','Pixel deviation'),
'stdDev':('Standard Deviation of Errors','Pixel deviation'),
};
paramTypeDict = {
'locTech' : ('Candidate location techniques',locationTypes),
'trainModel' : ('Training models',train_groups),
'fitTech' : ('Fitting techniques',channelTypes),
'testSet' : ('Test sets',test_groups),
};

# Everything below this line is for graphing
# It can and should be changed depending on the needed views.

# print("Summary Graphs");
# #vosm_helper.SummaryGraphs(allFoldsDB,paramTypeDict,resultTypeDict) #This plots all possible combinations
summaryResultDict = {'avgDist':('Average Pt Distance','Euclidian distance in pixels')}
summaryParamDict = {'testSet' : ('Test sets',test_groups)}
# vosm_helper.SummaryGraphs(allFoldsDB,numFolds,summaryParamDict,summaryResultDict)
    
# print("Versus Graphs");
# #vosm_helper.VersusGraphs(allFoldsDB,paramTypeDict,resultTypeDict) #This plots all possible combinations
versusResultDict = {'avgDist':('Average Pt Distance','Euclidian distance in pixels')}
# #Plots pairs of parameters against each other for every result specified
versusParamDict = {'testSet' : ('Test sets',test_groups), 'fitTech' : ('Fitting techniques',channelTypes)}
# vosm_helper.VersusGraphs(allFoldsDB,numFolds,versusParamDict,versusResultDict)
 
# print("All Points over Time Graph")
graphDict = {'trainModel' : train_groups[0], 'locTech' : locationTypes[0], 'testSet' : test_groups[0], 'fitTech' : channelTypes[0]}
#vosm_helper.AllPtsAvgTimeGraph(allFoldsDB,numFolds,landmarkList,graphDict,landmarkList) 

# print("Avg Result over Time");
indpVarDict = {'fitTech' : channelTypes}
depVarDict = {'trainModel' : train_groups[0], 'locTech' : locationTypes[0], 'testSet' : test_groups[0]}
# vosm_helper.AvgTimeGraph(allFoldsDB,numFolds,paramTypeDict,landmarkList,indpVarDict,depVarDict,landmarkList)

# indpVarDict = {'locTech' : locationTypes}
# depVarDict = {'trainModel' : train_groups[0], 'fitTech' : channelTypes[0], 'testSet' : test_groups[0]}
# vosm_helper.AvgTimeGraph(allFoldsDB,numFolds,paramTypeDict,landmarkList,indpVarDict,depVarDict,landmarkList)

# indpVarDict = {'fitTech' : channelTypes,'locTech' : locationTypes}
# depVarDict = {'trainModel' : train_groups[0],  'testSet' : test_groups[0]}
# vosm_helper.AvgTimeGraph(allFoldsDB,numFolds,paramTypeDict,landmarkList,indpVarDict,depVarDict,landmarkList)

print("Completed crossValTest")
#interactive debug handles
resultDB_list = list()
for foldNum in range(numFolds):
    resultDB_list.append(vosm_helper.LoadDB(resultPathList[foldNum]))
    
