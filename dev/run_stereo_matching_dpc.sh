#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling Project Stereo Matching Optimized - 1 of 1
rm -rf stereo-matching-dpc/build
cd stereo-matching-dpc &&
mkdir build &&  
cd build &&  
cmake ../. &&  
make
make run

