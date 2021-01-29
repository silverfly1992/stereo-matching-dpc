# stereo-matching-dpc
Stereo matching demo written using OneAPI dpc++, including Basic kernel and ND-Range kernel

source /opt/intel/oneapi/setvars.sh

git clone https://github.com/silverfly1992/stereo-matching-dpc.git
rm -rf stereo-matching-dpc/build
cd stereo-matching-dpc &&
mkdir build &&  
cd build &&  
cmake ../. &&  
make
make run
