本项目为基于OneAPI dpc++编写的立体匹配示例，包含针对GPU的Basic和ND-Range两种kernel类型。

立体匹配的代价函数为简单的窗口绝对值之和（SAD），为了加速运算，原始的C++代码已将二维代价计算转化成：首先在竖直方向计算一维代价，然后在水平方向进行聚合。

**本项目已在Windows和Intel的DevCloud上进行了测试。**

1. Windows环境为：Windows 10 专业版，64位，Intel Core i7-7700 CPU，HD Graphics 630；在Visual Studio 2017 Release x64模式下能成功运行，Debug模式会有问题，待解决。成功运行的结果如下：

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

2.Linux系统下可按如下步骤执行：

- 设置环境变量


`source /opt/intel/oneapi/setvars.sh`

- 拷贝代码库


`git clone https://github.com/silverfly1992/stereo-matching-dpc.git`

- 编译执行


`rm -rf stereo-matching-dpc/build`

`cd stereo-matching-dpc &&`

`mkdir build &&`  

`cd build &&`  

`cmake ../. &&`  

`make`

`make run`

Intel DevCloud环境下提交任务的完整代码放在了项目的dev文件夹下，执行的结果截图如下：

```
![image](https://github.com/silverfly1992/stereo-matching-dpc/blob/main/images/image-20210130094805873.png)
```

本项目参考了https://github.com/dhernandez0/sgm中的CUDA代码。

