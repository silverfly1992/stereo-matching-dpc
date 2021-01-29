#ifndef _STEREOMATCHINGCPU_H_
#define _STEREOMATCHINGCPU_H_

void StereoMatchingCPU(uint8_t* imgL, uint8_t* imgR, uint8_t* out, int width, int height);
int SAD(uint8_t* imgL, uint8_t* imgR, int i, int j, int d, int width, int height, int win_width);
void aggregateCostHorizontal(int* cost, int* aggregatedCost, int width, int height);
void WTA(int* costVolumn, uint8_t* disparityMap, int width, int height);

#endif