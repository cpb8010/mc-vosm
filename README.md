mc-vosm
=======

Multi-channel ASM enhancements to the [Visual Open Statistical Models project](http://www.visionopen.com/)

##Origin
[VOSM 0.3.3](http://www.visionopen.com/downloads/open-source-software/VOSM/)

The core VOSM package implements multiple AAM and ASM techniques.
It supports a few annotation formats and is designed primarily for faces.

### About the author
I've worked on this as part of my master's thesis and I have permission from VOSM's primary author, Jia Pei, to publish my enhancments here.

##Current state
MC-VOSM is a fork of VOSM that is independatly maintained with a focus on ASM enhancements for color+depth image processing and improved result reporting.
The multi-channel ASM techniques have been tested using the [Bosphorus Facial Expression Database](http://bosphorus.ee.boun.edu.tr/default.aspx)

### Current features
* 3 New multi-channel fitting techniques
* 2 New initial shape placement techniques
* Automated cross validation testing
* Automated result collection and graph generation
* Support for Bosphorus depth images and annotation information
* Up to 3 image channels supported for building and fitting


### Current external dependancies
Older versions might work, but the versions listed have been tested to work
C++ is for building the ASM building and fitting programs
Python is required for automating testing and graph generation

* C++
  * [Boost v1.53](http://www.boost.org/)
      * Only uses boost::filesystem, so older versions work fine too
  * [OpenCV 2.4.5](http://opencv.org/)
      * Requires 2.4 or newer because of Mat syntax changes
  * [YAML-CPP 0.5.1](http://code.google.com/p/yaml-cpp/)
  * [*Visual Studio 2012*](http://www.microsoft.com/visualstudio/eng/downloads#d-2012-express)
      * Windows only :( until CMake file are updated, other platforms remain untested
      * All .sln and .vxproj were converted to vs2012 any may not be backwards compatiable
      
Most of these are dependant on each other, so get whatever versions are supported
* [Python 2.7.5](http://www.python.org/download/)
  * [Matplotlib](http://matplotlib.org/downloads.html)
  * [NumPy](http://www.numpy.org/)
  * [SciPy](http://www.scipy.org/)
  * [Pandas](http://pandas.pydata.org/)
  * [SciKit-Learn](http://scikit-learn.org/stable/install.html)
  
# Expected Use Case
As a completly open source library, MC-VOSM can serve as a base for testing new ASM techniques or new face databases.
Given a new database of faces with facial feature point information and a compiling version of mc-vosm, the following is necessary to completly test the new database or method with MC-VOSM:

1. A shape-info file, detailing how each facial feature point relates to its neighbors and relative to the face.
2. A Python function to convert (if necessary) and sort images based on types or experimental parameters
3. An optional C++ code path to implement a new ASM technique (if testing a new ASM technique)
  * An additional YAML setting to enable/disable that code path
* If annotation format is not supported, C++ code to read annotation file format into vosm.
  * An additional YAML setting for the new annotation file format

After these steps are completed, running a single python script will call your convert and sorting functions to create N(default 10) validation folds.
The script will then automatically train all the necessary models and run all of the fitting tests.
MC-VOSM will record the results as images and text files, which the python script will scrape into a database for graphing.
The script will then plot the accuracy and efficiency results for each sorted group across all the validation folds.

#### Additional notes
* Real-time tracking is supported in the test program, but has not been evaulated.
* mc-vosm has a seperate test fitting executable, so it can be integrated with a different program.
* AAM techniques are left unchanged from VOSM, but will use the same result reporting functions.
  * For questions about AAM techniques, check out the VOSM site, but feel free to ask ASM questions here.
* More Detailed changes from VOSM are listed [here](VOSM_changelist.txt)
