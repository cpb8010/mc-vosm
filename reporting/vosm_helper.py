#vosm_bs_helpers.py
import pickle
import numpy
import os
import subprocess
import matplotlib.pyplot
import sqlite3
import itertools
from pandas import *

from collections import defaultdict 

def MergeDB(masterDB,tempDB_path,foldNum):
    """ Insert all tables from tempDB into the same masterDB database
    
        masterDB -- existing open database connection with correct schema to write into
        tempDB_path -- path to existing databaste to read from
        db_name -- name to attach database
        
    """
    mdbc = masterDB.cursor()
    db_name = 'Set_%d' % foldNum
    print("ATTACH '%s' AS %s" % (tempDB_path,db_name))
    mdbc.execute("ATTACH '%s' AS %s" % (tempDB_path,db_name))
    masterDB.commit()    
    
def buildQueryString(indpVarDict,depVarDict,whatString):
    selectString = "SELECT "
    for indpKey in indpVarDict:
        selectString = selectString + indpKey + ", "
        
    whereString = " WHERE "
    for depKey in depVarDict:
        whereString = whereString + depKey + "='" + depVarDict[depKey] + "' AND "

    iterQueryString = selectString + whatString + whereString + " iterNum=:iterNum"
    
    return iterQueryString
    
def runIndvErrQuery(dbc,iterQueryList,errQueryString,errTimeArray,indpVars,landmarkList,delPts,first):
            
    errDict = defaultdict(list)
    #Get individual face results and save average
    for errTuple in iterQueryList:
        errKey = errTuple[len(errTuple)-1]
        for tupIndex in range(len(errTuple)-1):
            errType = errTuple[tupIndex]
            dbc.execute(errQueryString, {'errTableKey' :errKey})
            singleError = list(dbc.fetchall().pop())
            singleError.remove(errKey)
            #remove points not in list
            if len(singleError) != len(landmarkList):
                printf("Incorrect landmark list length")
                exit()
            singleError = [pt for ptIndex, pt in enumerate(singleError) if not ptIndex in delPts]
            #print('.append',numpy.mean(numpy.array(singleError)))        
        
            errDict[errType].append(numpy.mean(numpy.array(singleError)))
    
    #NaN fill
    if(len(errDict.keys()) != len(indpVars)):
        missingVars = [missingVar for missingVar in indpVars if not missingVar in errDict.keys()]
        print('missingVars',missingVars)
        for missingVar in missingVars:
            errDict[missingVar].append(float('NaN'))
        
    #avgList = [sum(errDict[errKey]) / float(len(errDict[errKey])) for errKey in errDict]
    avgList = []
    for varKey in indpVars:
        avgList.append(sum(errDict[varKey]) / float(len(errDict[varKey])))
    
    numpyAvg = numpy.array(avgList)

    if first:
        errTimeArray = numpyAvg
    else:
        errTimeArray = numpy.vstack((errTimeArray,numpyAvg))
        #print("errTimeArray",errTimeArray)
        
    return errTimeArray
        

def graphErrorTimeArray(errTimeArray,titleString,graphColumns):

    lineColors = ['b','g','r','c','m','y','k']
    lineTypes = ['-','--','-.',':']
    lineMarkers = []
    for m in matplotlib.pyplot.Line2D.markers:
        try:
            if len(m) == 1 and m != ' ':
                lineMarkers.append(m)
        except TypeError:
            pass
            
    lineColorMarkers = [''.join(ele) for ele in zip(lineColors,lineMarkers)]
    lineStyles = [''.join(ele) for ele in itertools.product(lineTypes,lineColorMarkers)]
    
    print('errTimeArray.shape',errTimeArray.shape)
    df = DataFrame(errTimeArray, index=range(errTimeArray.shape[0]), columns=graphColumns)
    df.plot(style=lineStyles, title=titleString).set_ylabel("Avg Distance for " + str(len(graphColumns)) + " points" )
    matplotlib.pyplot.legend(loc='best')
    matplotlib.pyplot.show()

def avgTimeGraph(resultDB,paramTypeDict,landmarkList,indpVarDict,depVarDict,ptList):
    """ Legacy no-validation fold support graphing method"""

    dbc = resultDB.cursor()
    #flatten list to count elements
    indpVars = [ paramItem for indpVKey in indpVarDict for paramItem in paramTypeDict[indpVKey][1]]
    #print('indpVars',indpVars)
    #find points to remove from total
    delPts = [ index for index, landmark in enumerate(landmarkList) if not landmark in ptList] 
    #print('delPts',delPts)

    iterQueryString = buildQueryString(indpVarDict,depVarDict,"errPtsTableKey FROM InterIndvResults")
    errQueryString = "SELECT * FROM PtsErrors WHERE key=:errTableKey"
    
    #Assuming at most 25 iterations recorded
    firstRun = True
    errTimeArray = False
    for iterNum in range(25):
        #print("iterQueryString",iterQueryString)
        dbc.execute(iterQueryString,{'iterNum': iterNum})
        iterQueryList = dbc.fetchall()
        #print('iterQueryList',iterQueryList)
        
        if(len(iterQueryList) == 0):
            print('Empty search query',iterNum)
            continue
            
        errTimeArray = runIndvErrQuery(dbc,iterQueryList,errQueryString,errTimeArray,indpVars,landmarkList,delPts,firstRun)
        firstRun = False
        
    titleString = "Test variables: "
    for depVarKey in depVarDict:
        titleString = titleString + depVarKey + "=" + depVarDict[depVarKey] + " "
        
    graphErrorTimeArray(errTimeArray,titleString,indpVars)
    
    
def AvgTimeGraph(resultDB,numFolds,paramTypeDict,landmarkList,indpVarDict,depVarDict,ptList):
    """ Time series line plots of independant variables over time
    
    Keyword arguments:
        resultDB -- opened sqlite3.connection created by StoreResults etc
        numFolds -- number of folds in database, 0 for no folds
        paramTypeDict -- dictionary with keys of sql column names and values of tuples whose second value contains a list of strings corresponding to the possible values in the sql columns
        landmarkList -- list of strings with the names of all landmarks, must be the same size as the points retrieved
        indpVarDict -- dict or list with keys of sql column names.
            All possible values of these parameters are plotted, can contain more than one column name
        depVarDict -- dict with keys of sql column names and value of a fixed parameters
            This dictionary lists the parameters to fix during plotting, they remain the same across all indpVar keys
        ptList -- list of strings with the names of landmarks to include in the average
            If this is the same as landmarkList, all points will be used for the average.
    
    """
    
    if(numFolds <= 0):
        return avgTimeGraph(resultDB,paramTypeDict,landmarkList,indpVarDict,depVarDict,ptList)
    
    dbc = resultDB.cursor()
    #flatten list to count elements
    indpVars = [ paramItem for indpVKey in indpVarDict for paramItem in paramTypeDict[indpVKey][1]]
    #print('indpVars',indpVars)
    #find points to remove from total
    delPts = [ index for index, landmark in enumerate(landmarkList) if not landmark in ptList] 
    #print('delPts',delPts)
    
    iterQueryString = buildQueryString(indpVarDict,depVarDict,"errPtsTableKey FROM Set_%d.InterIndvResults")
    errQueryString = "SELECT * FROM Set_%d.PtsErrors WHERE key=:errTableKey"

    
            
    #Assuming at most 25 iterations recorded
    firstRun = True
    for iterNum in range(25):
        for foldNum in range(numFolds):
            print("iterQueryString",(iterQueryString % (foldNum)))
            dbc.execute(iterQueryString % foldNum,{'iterNum': iterNum})
            iterQueryList = dbc.fetchall()
            if(len(iterQueryList) == 0):
                print('Empty search query',iterNum)
                continue
            
            errTimeArray = runIndvErrQuery(dbc,iterQueryList,errQueryString % foldNum,errTimeArray,indpVars,landmarkList,delPts,firstRun)
            firstRun = False

    titleString = "Test variables: "
    for depVarKey in depVarDict:
        titleString = titleString + depVarKey + "=" + depVarDict[depVarKey] + " "
        
    graphErrorTimeArray(errTimeArray,titleString,indpVars)


def runErrQuery(dbc,errQueryString,iterQueryList,landmarkList,keepPts,first,errTimeArray):
    innerFirst = True
    for errKey in iterQueryList:
        dbc.execute(errQueryString,errKey)

        singleError = list(dbc.fetchall().pop())
        singleError.remove(errKey[0])
        #remove points not in list
        if len(singleError) != len(landmarkList):
            printf("Incorrect landmark list length")
            exit()
        singleError = [pt for ptIndex, pt in enumerate(singleError) if ptIndex in keepPts]
        
        if innerFirst:
            innerFirst = False
            numpySum = numpy.array(singleError)
        else:
            numpySum = numpySum + numpy.array(singleError)

    numpyAvg = numpy.divide(numpySum,len(iterQueryList))
    #print("numpyAvg",numpyAvg)
    if first:
        errTimeArray = numpyAvg
    else:
        errTimeArray = numpy.vstack((errTimeArray,numpyAvg))
        #print("errTimeArray",errTimeArray)
        
    return errTimeArray
    
def allPtsAvgTimeGraph(resultDB,landmarkList,graphDict,ptNameList):
    """ This is a helper function that supports the legacy case of no validation folds
        It graphs Time series line plots individual points over time
    
    Keyword arguments:
        resultDB -- opened sqlite3.connection created by StoreResults etc
        landmarkList -- list of strings with the names of all landmarks, must be the same size as the points retrieved
        graphDict -- dict with keys of sql column names and values of a fixed parameter
            This dictionary lists the parameters to use for graphing
        ptNameList -- list of strings with the names of landmarks to plot
            If this is the same as landmarkList, all points will be plotted.
    
    """
    
    iterQueryString = "SELECT errPtsTableKey FROM InterIndvResults WHERE locTech=? AND trainModel=? AND fitTech=? AND testSet=? AND iterNum=?"
    errQueryString = "SELECT * FROM PtsErrors WHERE key=?"

    dbc = resultDB.cursor()
    #find points to keep in total
    keepPts = [ index for index, landmark in enumerate(landmarkList) if landmark in ptNameList] 

    first = True
    errTimeArray = False
    #hardcoded 25 because the max number of iterations isn't stored anywhere as a parameter yet
    for iterNum in range(25):
        dbc.execute(iterQueryString,(graphDict['locTech'],graphDict['trainModel'],graphDict['fitTech'],graphDict['testSet'],iterNum))
        iterQueryList = dbc.fetchall()
    
        if(len(iterQueryList) == 0):
            print('Empty search query',iterNum)
            continue
        
        errTimeArray = runErrQuery(dbc,errQueryString,iterQueryList,landmarkList,keepPts,first,errTimeArray)
        first = False
        
    titleString = "Test variables: " + graphDict['locTech'] + ", " + graphDict['trainModel'] + ", " + graphDict['fitTech'] + ", " + graphDict['testSet']
    graphErrorTimeArray(errTimeArray,titleString,ptNameList)
    
    return True
    

def AllPtsAvgTimeGraph(resultDB,numFolds,landmarkList,graphDict,ptNameList):
    """ Time series line plots individual points over time
    
    Keyword arguments:
        resultDB -- opened sqlite3.connection created by StoreResults etc
        numFolds -- number of folds in database
        landmarkList -- list of strings with the names of all landmarks, must be the same size as the points retrieved
        graphDict -- dict with keys of sql column names and values of a fixed parameter
            This dictionary lists the parameters to use for graphing
        ptNameList -- list of strings with the names of landmarks to plot
            If this is the same as landmarkList, all points will be plotted.
            
    Returns True if numFolds is valid
    
    """
    if(numFolds < 0):
        print("Must have a positive number of validtion folds to use this graphing method")
        return False
        
    if(numFolds == 0):
        return allPtsAvgTimeGraph(resultDB,landmarkList,graphDict,ptNameList)
        
        
    dbc = resultDB.cursor()
    #find points to keep in total
    keepPts = [ index for index, landmark in enumerate(landmarkList) if landmark in ptNameList] 

    iterQueryString = "SELECT errPtsTableKey FROM Set_%d.InterIndvResults WHERE locTech=? AND trainModel=? AND fitTech=? AND testSet=? AND iterNum=?"
    errQueryString = "SELECT * FROM Set_%d.PtsErrors WHERE key=?"
        
    outerFirst = True
    errTimeArray = False
    #hardcoded 25 because the max number of iterations isn't stored anywhere as a parameter yet
    for iterNum in range(25):
        for foldNum in range(numFolds):
            dbc.execute(iterQueryString % foldNum,(graphDict['locTech'],graphDict['trainModel'],graphDict['fitTech'],graphDict['testSet'],iterNum))
            iterQueryList = dbc.fetchall()
    
            if(len(iterQueryList) == 0):
                print('Empty search query',iterNum,foldNum)
                continue
            
            errTimeArray = runErrQuery(dbc,iterQueryList,errQueryString % foldNum,first,errTimeArray)
            first = False
            
    graphErrorTimeArray(errTimeArray,graphDict,ptNameList)
    
    return True
    
def SummaryGraphs(resultDB,numFolds,paramTypeDict,resultTypesDict):
    """ Creates average bar graphs by running queries on the SummaryResults table
    
    Keyword arguments:
    resultDB -- opened sqlite3.connection created by StoreResults
    numFolds -- number of folds in database, 0 for no folds
    paramTypeDict -- Dictionary with keys corresponding to input columns in SummaryResults 
        and values that are tuples of text description and parameter list (values used for axes lables)
    resultTypesDict -- Dictionary with keys corresponding to output columns in SummaryResults 
        and values that are tuples of text descriptions and text measurement labels
    
    """

    resultTypesKeys = resultTypesDict.keys()
    dbc = resultDB.cursor()
    figList = []

    #for each ResultType (eg: detTimes) show avg Param performance
    # eg: average detection times for location techniques
    for resultKey in resultTypesKeys:
        #eg: resultKey=detTimes
        for paramKey in paramTypeDict:
            paramList = paramTypeDict[paramKey][1]
            paramTypeName = paramTypeDict[paramKey][0]
            #eg: paramList=['min-max','VJc', 'minVJc', 'GT']
            #eg: paramKey=locTech
            resultList = list()
            #iterate over a copy so we can remove empty elems
            for param in list(paramList):
                if(numFolds == 0):
                    queryString = "SELECT %s FROM SummaryResults WHERE %s='%s'" % (resultKey,paramKey,param)
                    dbc.execute(queryString)
                    queryResults = dbc.fetchall()
                    if(len(queryResults) == 0):
                        # Need to add a number to keep dimensions the same,
                        #but this approach also invalidates the 2D mean
                        queryResults.append(float('NaN'))
                        continue
                    resultList.append(numpy.mean(numpy.array(queryResults)))
                else:
                    verifList = list()
                    for foldNum in range(numFolds):
                        #db_name and foldNum are redundant for safety
                        db_name = "Set_%d" % foldNum
                        queryString = "SELECT %s FROM %s.SummaryResults WHERE %s='%s' AND verfSet=%d" % (resultKey,db_name,paramKey,param,foldNum)
                        dbc.execute(queryString)
                        queryResults = dbc.fetchall()
                        if(len(queryResults) == 0):
                            # Need to add a number to keep dimensions the same,
                            #but this approach also invalidates the 2D mean
                            verifList.append(float('NaN'))
                            continue
                        verifList.append(numpy.mean(numpy.array(queryResults)))
                    resultList.append(verifList)
                    
            resultArray = numpy.array(resultList)
            
            meanVec = numpy.mean(resultArray,axis=1,keepdims=True)
            stdVec = numpy.std(resultArray,axis=1,keepdims=True)
            mv1 = meanVec.flatten()
            stdv1 = stdVec.flatten()
            numX = numpy.arange(mv1.shape[0])
            fig = matplotlib.pyplot.figure()
            figList.append(fig)
            ax = fig.add_subplot(111)
            rects1 = ax.bar(numX, mv1.flatten(), 0.35, color='r', yerr=stdv1.flatten())
            ax.set_ylabel(resultTypesDict[resultKey][1])
            ax.set_title(resultTypesDict[resultKey][0] + " of " + paramTypeName)
            ax.set_xticks(numX+0.35)
            ax.set_xticklabels( paramList )
            matplotlib.pyplot.show()
    
    return figList
            
def VersusGraphs(resultDB,numFolds,paramTypeDict,resultTypesDict):
    """
    Creates bar graphs with 2 input columns from resultDB's SummaryResults table
    
    Keyword arguments:
    resultDB -- opened sqlite3.connection created by StoreResults
    numFolds -- number of folds in database, 0 for no folds
     paramTypeDict -- Dictionary with keys corresponding to input columns in SummaryResults 
        and values that are tuples of text description and parameter list (values used for axes lables)
    resultTypesDict -- Dictionary with keys corresponding to output columns in SummaryResults 
        and values that are tuples of text descriptions and text measurement labels
    """
    resultTypesKeys = resultTypesDict.keys()
    dbc = resultDB.cursor()

    #for each ResultType (eg: detTimes) show avg Param performance
    # eg: average detection times for location techniques
    for resultKey in resultTypesKeys:
        #eg: resultKey=detTimes
        #double iterate through dict (x-axis vs legend)
        # this could be done better by something in itertools (combinations)
        for paramKeyX in paramTypeDict:
            for paramKeyL in paramTypeDict:
                if(paramKeyX == paramKeyL):
                    continue
                paramListX = paramTypeDict[paramKeyX][1]
                paramListL = paramTypeDict[paramKeyL][1]
                
                paramTypeNameX = paramTypeDict[paramKeyX][0]
                paramTypeNameL = paramTypeDict[paramKeyL][0]
                #eg: paramListX=['min-max','VJc', 'minVJc', 'GT']
                #eg: paramListL=['default', 'gray', 'depth', 'pyr', 'sub']
                #eg: paramKeyX=locTech
                #eg: paramKeyL=fitTech

                meanResultArray = numpy.empty((len(paramListL),len(paramListX)))
                stdResultArray = numpy.empty((len(paramListL),len(paramListX)))
                #iterate over a copy so we can remove empty elems
                for xIndex, paramX in enumerate(paramListX):
                    resultListL = list()
                    for lIndex, paramL in enumerate(paramListL):
                        if (numFolds == 0):
                            queryString = "SELECT %s FROM SummaryResults WHERE %s='%s' AND %s='%s' " % (resultKey,paramKeyX,paramX,paramKeyL,paramL)
                            dbc.execute(queryString)
                            queryResults = dbc.fetchall()
                            #print('len(queryResults)',len(queryResults))
                            if(len(queryResults) == 0):
                                print("No results for",paramX,paramL)
                                #paramListX.remove(param)
                                meanResultArray[lIndex][xIndex] = float('NaN')
                                stdResultArray[lIndex][xIndex] = float('NaN')
                                continue
                            meanResultArray[lIndex][xIndex] = numpy.mean(queryResults)
                            stdResultArray[lIndex][xIndex] = numpy.std(queryResults)

                        else:
                            queryList = list()
                            for foldNum in range(numFolds):
                                queryString = "SELECT %s FROM Set_%d.SummaryResults WHERE %s='%s' AND %s='%s' " % (resultKey,foldNum,paramKeyX,paramX,paramKeyL,paramL)
                                dbc.execute(queryString)
                                queryResults = dbc.fetchall()
                                #print('len(queryResults)',len(queryResults))
                                if(len(queryResults) == 0):
                                    print("No results for",paramX,paramL)
                                    #paramListX.remove(param)
                                    meanResultArray[lIndex][xIndex] = float('NaN')
                                    stdResultArray[lIndex][xIndex] = float('NaN')
                                    continue
                                queryList.append(numpy.mean(queryResults))
                                queryList.append(numpy.std(queryResults))
                            #doing this outside the loop is harder
                            meanResultArray[lIndex][xIndex] = numpy.mean(queryList)
                            stdResultArray[lIndex][xIndex] = numpy.std(queryList)

                
                df2 = DataFrame(meanResultArray, columns=paramListX, index=paramListL)
                plotax = df2.plot(kind='bar', title=resultTypesDict[resultKey][0] + " of " + paramTypeNameX + " and " + paramTypeNameL).set_ylabel(resultTypesDict[resultKey][1])
                matplotlib.pyplot.show()
            
#combine this with parseFaceResFile and add a flag once typos are fixed   
def parseIterFaceResFile(resultFile):
    resDict = dict()
    with open(resultFile, 'r') as rfile:
        if(not "Error per point" in rfile.readline()):
            print("Parse: Error per point failed", resultFile)
            return False
        
        #Error per point
        eList = list()
        line = rfile.readline()
        while(not "\n" == line):
            eList.append(float(line))
            line = rfile.readline()
        resDict['Distance per point'] = eList
            
        if(not "Total landmark error" in rfile.readline()):
            print("Parse: Total landmark error failed", resultFile)
            return False
            
        resDict['Total landmark distance'] = float(rfile.readline())
        
        if(not "Average landmark distance" in rfile.readline()):
            print("Parse: Average landmark distance failed", resultFile)
            return False
            
        resDict['Average landmark distance'] = float(rfile.readline())
        
        #empty line
        rfile.readline()
        
        if(not "Total Deviation" in rfile.readline()):
            print("Parse: Total Deviation failed", resultFile)
            return False
            
        resDict['Total deviation'] = float(rfile.readline())
            
        if(not "Point Error" in rfile.readline()):
            print("Parse: Point Error failed", resultFile)
            return False
            
        resDict['Distance frequency'] = rfile.readline().split()
            
        #empty line
        rfile.readline()
        
        fplist = list()
        if(not "Fitted points" in rfile.readline()):
            print("Parse: Fitted points failed", resultFile)
            return False
        
        line = rfile.readline()
        while(not "\n" == line):
            xp = line.split(' ')[0]
            yp = line.split(' ')[1]
            fplist.append((xp,yp))
            line = rfile.readline()
        resDict['Fitted points'] = numpy.array(fplist)
            
    rfile.close()
    
    return resDict

def parseFaceResFile(resultFile):
    """ Parses individual fit result
    
        Arguments:
            resultFile - path to a .res file with indv results
            
        returns a dictionary with the following keys
        Distance per point -- a list of floats
        Total landmark distance 
        Average landmark distance
        Candidate point error -- a tuple of lele, reye, mouth error
        Fitting time
        Total deviation
        Distance frequency -- a list of 20 values
        
    
    """
    resDict = dict()
    with open(resultFile, 'r') as rfile:
    
        if(not "Error per point" in rfile.readline()):
            print("Parse: Error per point failed", resultFile)
            return False
        
        eList = list()
        line = rfile.readline()
        while(not "\n" == line):
            eList.append(float(line))
            line = rfile.readline()
        resDict['Distance per point'] = eList
        
        if(not "Total landmark error" in rfile.readline()):
            print("Parse: Total landmark error failed", resultFile)
            return False
            
        resDict['Total landmark distance'] = float(rfile.readline())
        
        if(not "Average landmark distance" in rfile.readline()):
            print("Parse: Average landmark distance failed", resultFile)
            return False
            
        resDict['Average landmark distance'] = float(rfile.readline())
        
        if(not "Candidate point error" in rfile.readline()):
            print("Parse: Candidate point error failed", resultFile)
            return False
            
        #tuple: leye, reye, mouth
        resDict['Candidate point distance'] = (float(rfile.readline()),float(rfile.readline()),float(rfile.readline()))
        
        #empty line?
        rfile.readline()
        
        if(not "Fitting time" in rfile.readline()):
            print("Parse: Fitting time failed", resultFile)
            return False
            
        resDict['Fitting time'] = float(rfile.readline())
            
        #empty line?
        rfile.readline()
        
        if(not "Total deviation" in rfile.readline()):
            print("Parse: Total deviation failed", resultFile)
            return False
            
        resDict['Total deviation'] = float(rfile.readline())
          
        #WatchOut! the string changes depending on the format.
        if(not "Point error" in rfile.readline()):
            print("Parse: Point error failed", resultFile)
            return False
            
        resDict['Distance frequency'] = rfile.readline().split()
        
        #empty line
        rfile.readline()
        
        #Fix this spelling error also.
        if( not "Canidate points" in rfile.readline()):
            print("Parse: Canidate points failed", resultFile)
            return False
            
        cplist = list()
        line = rfile.readline()
        while(not "Fitted points" in line):
            xp = line.split(' ')[0]
            yp = line.split(' ')[1]
            cplist.append((xp,yp))
            line = rfile.readline()
        resDict['Candidate points'] = numpy.array(cplist)
        
        #impossible?
        if(not "Fitted points" in line):
            print("Parse: Fitted points failed", resultFile)
            return False
            
        fplist = list()
        line = rfile.readline()
        while(not "\n" == line):
            xp = line.split(' ')[0]
            yp = line.split(' ')[1]
            fplist.append((xp,yp))
            line = rfile.readline()
        resDict['Fitted points'] = numpy.array(fplist)
            
    rfile.close()
    
    return resDict
          
def parseResultsFile(resultTextFile):
    """ Parse average results file into dictionary
    
        Arguments:
            resultTextFile -- path to .txt file that was produced from the standard output of VOSM
            
        Returns:
            a dictionary with the following keys that have single values
                Detection Times
                Average Iteration Times
                Average Detection time (in ms)
                Average Pt Distance
                Average Deviation of Errors
                Standard Deviation of Errors
            And the key 
                Distance Frequency
            contains a list of 20 floats
    
    
    """
    resDict = dict()
    #print('resultTextFile',resultTextFile)
    with open(resultTextFile, 'r') as rfile:
        firstLine = rfile.readline().strip()
        #check for invalid run
        if( "Usage:" in firstLine):
            return False
        #check for valid run
        if( not "m_vChin" in firstLine):
            return False
        errFreq = list()    
        for line in rfile:
            if("Detection Times" in line):
                # Yet another validity check
                if( int(line.split('=')[1]) <= 0):
                    return False
                resDict['Detection Times'] = int(line.split('=')[1])
            if("Average Number of Iterations" in line):
                resDict['Average Number of Iterations'] = float(line.split('=')[1])
            if("Average Detection time (in ms)" in line):
                resDict['Average Detection time (in ms)'] = float(line.split('=')[1])
            if("Average Pt Distance" in line):
                resDict['Average Pt Distance'] = float(line.split('=')[1])
            if("Average Deviation of Errors" in line):
                resDict['Average Deviation of Errors'] = float(line.split('=')[1])
            if("Standard Deviation of Errors" in line):
                resDict['Standard Deviation of Errors'] = float(line.split('=')[1])
            if("percentage of points are in" in line):
                #assume in order traversal (0-19)
                errFreq.append(float(line.split('percentage of points are in')[0]))
            
        resDict['Distance Frequency'] = errFreq
        
    rfile.close()
    
    return resDict

def parseResultName(resultFileName):
    resultFormat = resultFileName.rstrip('_results.txt')
    resultFormat = resultFormat.strip('result_')

    resParts = resultFormat.split('_')

    if(len(resParts) == 4):
        return resParts[0], resParts[1], resParts[2], resParts[3]
    elif(len(resParts) == 5):
        return resParts[0], resParts[1], resParts[2], (resParts[3] + "_" + resParts[4])
    else:
        print("Unknown result file name format!",resultFileName,resultFormat)
        return False
    
def CreateMasterDB(resultDbFilePath):
    return sqlite3.connect(resultDbFilePath)
    
def CreateDB(resultDbFilePath,numPts):
    #no duplication checking is done
    with sqlite3.connect(resultDbFilePath) as resultDB:
        dbc = resultDB.cursor()
        #All database names are hardcoded here, might be good to pull them out in documentation
        print("SQLite db %s connected" % resultDbFilePath)
        # Summary table: 11 columns(5 inputs, 6 outputs), 1 row per test run
        dbc.execute("create table if not exists SummaryResults(verfSet INTEGER, locTech TEXT, trainModel TEXT, fitTech TEXT, testSet TEXT, detTimes INTEGER, avgIterations REAL, avgFitTime REAL, avgDist REAL, avgDev REAL, stdDev REAL)")
        #2 Individual result tables: 1 final result table, 1 intermediate result table
        #4 Individual result sub-tables: 2 error tables, 2 point storage tables
        #Fitted points table width depends on number of points
        xPtsList = ",".join(["x" + str(colIndex) + " REAL" for colIndex in range(numPts)])
        yPtsList = ",".join(["y" + str(colIndex) + " REAL" for colIndex in range(numPts)])
        pList = ",".join(["p" + str(colIndex) + " REAL" for colIndex in range(numPts)])
        xyPtsList = xPtsList + "," + yPtsList
        #PtsErrors table: 1+numPts columns(1 primary key, n points), 1 row per fitting iteration+1
        dbc.execute("create table if not exists PtsErrors(key INTEGER PRIMARY KEY, %s)" % pList)
        #Pts table: 1 + 2*numPts columns(1 primary key, 2n columns), 1 rows per fitting iteration+1
        dbc.execute("create table if not exists Pts(key INTEGER PRIMARY KEY, %s)" % xyPtsList)
        #CPtsErrors: 4 columns(1 primary key, 3 outputs), 1 row per fitting iteration+1
        dbc.execute("create table if not exists CPtsErrors(key INTEGER PRIMARY KEY,leftEye REAL,rightEye REAL,mouth REAL)")
        #CPts: 5 columns(1 primary key, 1 input, 3 outputs), 2 rows per fitting iteration+1
        dbc.execute("create table if not exists CPts(key INTEGER PRIMARY KEY, leftEyeX REAL, leftEyeY REAL, rightEyeX REAL, rightEyeY REAL, mouthX REAL, mouthY REAL)")
        #Final result table: 11 columns(6 inputs, 4 linked tables, 1 output), 1 row per image
        dbc.execute("create table if not exists FinalIndvResults(verfSet INTEGER, subNum INTEGER, locTech TEXT, trainModel TEXT, fitTech TEXT, testSet TEXT, errPtsTableKey INTEGER, ptsTableKey INTEGER, errCpTableKey INTEGER, cpTableKey INTEGER, fitTime REAL, FOREIGN KEY(errPtsTableKey) REFERENCES PtsErrors(key), FOREIGN KEY(ptsTableKey) REFERENCES Pts(key) , FOREIGN KEY(errCpTableKey) REFERENCES CPtsErrors(key), FOREIGN KEY(cpTableKey) REFERENCES CPts(key))")
        #Intermediate result table: 9 columns(7 inputs, 2 linked tables), 1 row per fitting iteration
        dbc.execute("create table if not exists InterIndvResults(verfSet INTEGER, subNum INTEGER, locTech TEXT, trainModel TEXT, fitTech TEXT, testSet TEXT, iterNum INTEGER, errPtsTableKey INTEGER, ptsTableKey INTEGER, FOREIGN KEY(errPtsTableKey) REFERENCES PtsErrors(key), FOREIGN KEY(ptsTableKey) REFERENCES Pts(key) )")
    resultDB.commit()
    
    return resultDB

def StoreResults(resultDB,resultPath,numPts,verifySet):
    """
    Scrape TestModels's result directory and put individual text result files in sqlite db
    
    Keyword arguments:
    resultDB -- sqlite connection object opened by CreateDB
    resultPath -- root directory of the result folder
    numPts -- number of points expected in model results
    verifySet -- int stored with all the results collected
    """
    dbc = resultDB.cursor()

    for root, dirs, files in os.walk(resultPath):
        #Result run dir
        txtFileSearch = [file for file in files if file.endswith("_results.txt")]

        #SummaryResults
        if( len(txtFileSearch) == 1 ):
            resultsFile = txtFileSearch[0]
            testResults = parseResultsFile(os.path.join(root,resultsFile))
            if(testResults):
                #parse name of file
                locTech, trainModel, fitTech, testSet = parseResultName(resultsFile)
                sqlInsert = "INSERT INTO SummaryResults VALUES (?,?,?,?,?,?,?,?,?,?,?)"

                #cross-verify set is fixed for now.
                dbc.execute(sqlInsert,(verifySet,locTech,trainModel,fitTech,testSet,
                    testResults['Detection Times'], testResults['Average Number of Iterations'],
                    testResults['Average Detection time (in ms)'], testResults['Average Pt Distance'],
                    testResults['Average Deviation of Errors'], testResults['Standard Deviation of Errors']))
                    
        #for individual results, use different tables depending on the type of result
        resFileSearch = [file for file in files if file.endswith(".res")]
        insertArg = ",".join(["?" for colIndex in range(numPts)])
        ptsErrorsInsert = "INSERT INTO PtsErrors VALUES (NULL,%s)" % insertArg
        ptsInsert = "INSERT INTO Pts VALUES (NULL,%s)" % (insertArg + "," + insertArg)
        cPtsErrorsInsert = "INSERT INTO CPtsErrors VALUES (NULL,?,?,?)"
        cPtsInsert = "INSERT INTO CPts VALUES (NULL,?,?,?,?,?,?)"
        
        if( len(resFileSearch) > 0):
            for resFile in resFileSearch:
                #intermediate iteration result, need subject number and params
                if(resFile.rstrip('.res').isdigit()):
                    iterNum = int(resFile.rstrip('.res'))
                    resFolderPath, imgFolder = os.path.split(root)
                    subNum = imgFolder.split('_')[0].strip('bs')
                    locTech, trainModel, fitTech, testSet = parseResultName(os.path.split(resFolderPath)[1])
                    
                    iterResDict = parseIterFaceResFile(os.path.join(root,resFile))
                    if(len(iterResDict['Distance per point']) != numPts):
                        print("ERROR: Result point count mismatch",root,resFile)
                        exit()

                    dbc.execute(ptsErrorsInsert,(iterResDict['Distance per point']))
                    PtsErrorsKey = dbc.lastrowid
                    
                    dbc.execute(ptsInsert,(iterResDict['Fitted points'].flatten('F')))
                    PtsKey = dbc.lastrowid
                    
                    dbc.execute("INSERT INTO InterIndvResults VALUES (?,?,?,?,?,?,?,?,?)",(verifySet,subNum,locTech,trainModel,
                        fitTech,testSet,iterNum,PtsErrorsKey,PtsKey))
                else:
                    #should have good names
                    subNum = resFile.split('_')[0].strip('bs')
                    resFolderPath, imgFolder = os.path.split(root)
                    locTech, trainModel, fitTech, testSet = parseResultName(imgFolder)
                    resDict = parseFaceResFile(os.path.join(root,resFile))
                    if(len(resDict['Distance per point']) != numPts):
                        print("ERROR: Result point count mismatch",root,resFile)
                        exit()
                    
                    dbc.execute(ptsErrorsInsert,(resDict['Distance per point']))
                    PtsErrorsKey = dbc.lastrowid
                    
                    dbc.execute(ptsInsert,(resDict['Fitted points'].flatten('F')))
                    PtsKey = dbc.lastrowid
                    
                    dbc.execute(cPtsErrorsInsert,(resDict['Candidate point distance']))
                    CPtsErrorsKey = dbc.lastrowid
                    
                    dbc.execute(cPtsInsert,(resDict['Candidate points'].flatten('F')))
                    CPtsKey = dbc.lastrowid
                    
                    dbc.execute("INSERT INTO FinalIndvResults VALUES (?,?,?,?,?,?,?,?,?,?,?)",(verifySet,subNum,locTech,trainModel, fitTech,testSet,PtsErrorsKey,PtsKey,CPtsErrorsKey,CPtsKey,resDict['Fitting time']))
            
        if( len(resFileSearch) > 0 and len(txtFileSearch) == 1):
            #intermediate commits
            resultDB.commit()
            
    # Final commit
    resultDB.commit()
    
    return resultDB

def BuildModels(modelPath, imgPath, shapeInfoPath, buildProgramName, requiredArgs, trainingFolderNameList): 
    """ Builds VOSM models given training images and argument lists
    
        Arguments:
            modelPath -- Root directory where model folders will be created
            imgPath -- Root directory where images will be read
            shapeInfoPath -- Path of shapeInfo file required for building shape model
            buildProgramName -- Path of executable that builds VOSM model
            requiredArgs -- Various argments that remain the same between models
            trainingFolderNameList -- list of folder names in imgPath to read images, each entry then corresponds to a model folder at modelPath
            
        Returns:
            allPass -- true if all models build without errors
    """
    
    allPass = True
    for model_index in range(0,len(trainingFolderNameList)):
        trainingModelFolderName = modelPath + trainingFolderNameList[model_index]
        trainImgFolderName = imgPath + trainingFolderNameList[model_index]
        buildingCall = '%s -o "%s" -a "%s" -i "%s" -s %s %s' % (buildProgramName,trainingModelFolderName,trainImgFolderName,trainImgFolderName,shapeInfoPath,requiredArgs)
        if(os.path.exists(trainingModelFolderName)):
            print("Overwriting existing model")
        else:
            os.makedirs(trainingModelFolderName)
        #print(trainingFolderNameList[model_index])
        if(subprocess.call(buildingCall)):
            #print failed calls
            print('buildingCall',buildingCall)
            allPass = False
        
    return allPass
    
def TestModels(resultRoot, fittingProgramName, locationTypes, modelPath, modelTypes, channelTypes, testPath, testTypes, requiredFittingArgs): 
    """ Tests VOSM models with images and lists of parameters and saves results in folders named after the parameters used

        Arguments:
            resultRoot -- Path to the folder where the result folders will be created
            fittingProgramName -- Path of the executable to create the models
            locationTypes -- List of options fot candidate techniques
            modelPath -- Path to the folder with model folders
            modelTypes -- List of names of the folders in model path to run
            channelTypes -- List of options for channel selection techniques
            testPath -- Path to testing images
            testTypes -- List of folder names in testPath to run
            requiredFittingArgs -- invariant arguments across all test runs
            
        Returns:
            allPass -- True if all models ran without errors, false otherwise
            
    """
    curdir = os.getcwd()
    
    #print("resultRoot",resultRoot)
    #print("img_folder", testTypes)
    allPass = True
    for candidate_technique in locationTypes:
        for model_folder in modelTypes:
            for channel_technique in channelTypes:
                for img_folder in testTypes:
                    resultName = '%s_%s_%s_%s' % (candidate_technique, model_folder, channel_technique, img_folder)
                    resultFile = '%s_results.txt' % (resultName)
                    resultDir = '%sresult_%s' % (resultRoot, resultName)
                    modelLocation = modelPath + model_folder
                    testLocation = testPath + img_folder
                    fittingCall = '%s -o "%s" -c %s -i "%s" -a "%s" %s -l %s > %s' % (fittingProgramName,modelLocation, channel_technique, testLocation, testLocation, requiredFittingArgs, candidate_technique,resultFile)
                    if(os.path.exists(resultDir)):
                        print("Overwriting existing results")
                    else:
                        os.makedirs(resultDir)
                    os.chdir(resultDir)
                    print('fittingCall',fittingCall)
                    #print(testFolderNameList[model_index])
                    if(subprocess.call(fittingCall,shell=True)):
                        #print failed calls
                        #print('fittingCall',fittingCall)
                        print(candidate_technique, model_folder, channel_technique, img_folder)
                        allPass = False
                        
    
    os.chdir(curdir)
    return allPass

def Load(file_path):
    """ 
        pickleExists -- true if pickle exists in file_path, false otherwise
        loadedVar -- Unpickled last saved pklName
        """
    if(os.path.exists(file_path) and (os.path.isfile(file_path))):

        lfile = open(file_path, 'rb')
        loadedVar = pickle.load(lfile)
        lfile.close()
        return loadedVar
    else:
        return False   
        
def LoadDB(db_path):
    if(os.path.exists(db_path) and (os.path.isfile(db_path))):
        return sqlite3.connect(db_path)
    else:
        return False 

def Save(save_var, file_path, overwrite=False):        
    """ Pickles save_var to save_var, overwrites existing pickle if specified
        Returns false if pickle exists and overwrite is not specified, true otherwise
        """
    if(os.path.exists(file_path) and (os.path.isfile(file_path)) and not overwrite):
        return False
        
    sfile = open(file_path, 'wb')
    #dump in most compressed format
    pickle.dump(save_var, sfile, -1)
    sfile.close()
    return True
