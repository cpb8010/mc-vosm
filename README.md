# mc-vosm
_Multi-channel ASM enhancements to the [Visual Open Statistical Models project](http://www.visionopen.com/)_

This project can utilize the registered depth and color information from a Kinect to improve Active Shape Model (ASM) fitting accuracy and efficiency. Two ASM profile models are constructed based on the gradient profiles of depth and color intensity information. These two channels are fused using full or selective depth information. The algorithm determines the optimal point localization using the color/depth profiles during the update step of the ASM fitting process. This project extended the VOSM (Visual Open Statistical Models) library and uses the Bosphorus 3D Facial Expression Database for validation. Results on the validation database can show small improvements in accuracy and efficiency over single channel approaches depending on the pose and expression of the face.

=======
## How to Use
Clone this repository, install prerequisite libraries, compile, and run.

Cite this work:

C. Bellmore, R. Ptucha, and A. Savakis, “Fusing of Depth and Color for an Improved Active Shape Model,” Proc. IEEE Int. Conf. Image Processing, (ICIP 2013), Melbourne Australia, 2013.


More help can be found on the [wiki](https://github.com/cpb8010/mc-vosm/wiki)

##Origin
[VOSM 0.3.3](http://www.visionopen.com/downloads/open-source-software/VOSM/)

The core VOSM package implements multiple AAM and ASM techniques.
It supports a few annotation formats and is designed primarily for faces.

### LGPL 
Because VOSM is liscensed under the LGPL, so is this work.
Jia Pei, the primary author and maintainer of VOSM, also gave his permission for these enhancements to be published.

##Current state
MC-VOSM is a fork of VOSM that is independently maintained with a focus on ASM enhancements for color+depth image processing and improved result reporting.
The multi-channel ASM techniques have been tested using the [Bosphorus Facial Expression Database](http://bosphorus.ee.boun.edu.tr/default.aspx)

The library is ready for general use, but comes with no implicit or explicit warranty whatsoever, and still has many opportunities for enhancements.

### New features compared to VOSM 0.3.3
* 3 New multi-channel fitting techniques
* 2 New initial shape placement techniques
* Automated cross validation testing
* Automated result collection and graph generation
* Support for Bosphorus depth images and annotation information
* Up to 3 image channels supported for building and fitting ASMs


### Current external dependencies
Older versions might work, but the versions listed have been tested to work
C++ is for building the ASM building and fitting programs
Python is required for automating testing and graph generation

* C++
  * [Boost v1.53](http://www.boost.org/)
      * Only uses boost::filesystem, so older versions work fine too
  * [OpenCV 2.4.5](http://opencv.org/)
      * Requires 2.4 or newer because of Mat syntax changes
  * [YAML-CPP 0.5.1](http://code.google.com/p/yaml-cpp/)
      * Only required to example test fitting program
  * [*Visual Studio 2012*](http://www.microsoft.com/visualstudio/eng/downloads#d-2012-express)
      * This is the only part of the tool-chain that keeps this project on Windows, until CMake file are updated, other platforms remain untested
      * All .sln and .vxproj were converted to vs2012 any may not be backwards compatible
      
Most of these are dependant on each other, so get whatever versions are supported
* [Python 2.7.5](http://www.python.org/download/)
  * [Matplotlib](http://matplotlib.org/downloads.html)
  * [NumPy](http://www.numpy.org/)
  * [SciPy](http://www.scipy.org/)
  * [Pandas](http://pandas.pydata.org/)
  * [SciKit-Learn](http://scikit-learn.org/stable/install.html)
  
# Expected Use Case
As a completely open source library, MC-VOSM can serve as a base for testing new ASM techniques or new face databases.
Given a new database of faces with facial feature point information and a compiling version of mc-vosm, the following is necessary to completely test the new database or method with MC-VOSM:

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
* Real-time tracking is supported in the test program, but has not been evaluated.
* mc-vosm has a separate test fitting executable, so it can be integrated with a different program.
* AAM techniques are left unchanged from VOSM, but will use the same result reporting functions.
* For help adding another database to VOSM's supported organization/formats, please file an issue and I'll be happy to assist you!
* A detailed change list from VOSM 0.3.3 is [here](VOSM_changelist.txt)


[![Bitdeli Badge](https://d2weczhvl823v0.cloudfront.net/cpb8010/mc-vosm/trend.png)](https://bitdeli.com/free "Bitdeli Badge")

