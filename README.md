This project is an example of stereo matching algorithm written based on OneAPI dpc++, including two kernel types, Basic and ND-Range for GPU.

The cost function of stereo matching is the simple Sum of Absolute Difference (SAD) of the local window . In order to speed up the calculation, the original C++ code has converted the two-dimensional cost calculation into: first calculate the one-dimensional cost in the vertical direction, and then aggregate in the horizontal direction. The disparity position is calculated using the Winner Takes All (WTA) algorithm.

**This project has been tested on Windows system and Intel's DevCloud environment.**

### 1. Windows environment：

Windows 10 Professional Edition, 64-bit, Intel Core i7-7700 CPU, HD Graphics 630; it can run successfully in Visual Studio 2017 Release x64 mode, but there will be problems in Debug mode. After running successfully, it will display as follows:

```
Filename: ../res/imL.png W: 1280 H: 872

Running on Intel(R) Graphics [0x5912]
Max WG size is 256
Max Global memory size is 6830936064
Max Local memory size is 65536

Evaluating ND-Range Kernel...

Start image processing with offloading to GPU...
--The processing time is 0.622724 seconds
Start image processing with offloading to GPU...
--The processing time is 0.605297 seconds
Start image processing with offloading to GPU...
--The processing time is 0.597696 seconds
Start image processing with offloading to GPU...
--The processing time is 0.59721 seconds
Start image processing with offloading to GPU...
--The processing time is 0.591756 seconds

Average time for image processing:
--The average processing time was 0.602937 seconds

Evaluating Basic Kernel...

Start image processing with offloading to GPU...
--The processing time is 2.08487 seconds
Start image processing with offloading to GPU...
--The processing time is 1.31301 seconds
Start image processing with offloading to GPU...
--The processing time is 1.2861 seconds
Start image processing with offloading to GPU...
--The processing time is 1.27184 seconds
Start image processing with offloading to GPU...
--The processing time is 1.3528 seconds

Average time for image processing:
--The average processing time was 1.46172 seconds

/*********Compare Result*********/
All results Matched!
--The CPU processing time is 1.73988 seconds

Stereo matching successfully completed on the device.
The processed image has been written to ../res/disparity.png
```
The calculated disparity map is shown as follows:

![image](https://github.com/silverfly1992/stereo-matching-dpc/blob/main/images/disparity.png)

### 2. Linux system：

#### (1) Set environment variables

`source /opt/intel/oneapi/setvars.sh`

#### (2) Copy the code repository

`git clone https://github.com/silverfly1992/stereo-matching-dpc.git`

#### (3) Compile and run

`rm -rf stereo-matching-dpc/build`

`cd stereo-matching-dpc &&`

`mkdir build &&`  

`cd build &&`  

`cmake ../. &&`  

`make`

`make run`

The complete code for submitting tasks in the Intel DevCloud environment is placed in the "dev" folder of the project. The screenshot of the execution result is as follows:

![image](https://github.com/silverfly1992/stereo-matching-dpc/blob/main/images/image-20210130094805873.png)

This project refers to the CUDA code in https://github.com/dhernandez0/sgm.

