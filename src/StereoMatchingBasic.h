#ifndef _STEREOMATCHINGBASIC_H_
#define _STEREOMATCHINGBASIC_H_

void StereoMatchingBasic(uint8_t* imgL, uint8_t* imgR, uint8_t* out, int width, int height);
void CostComputeKernel(uint8_t* imgL, uint8_t* imgR, int* cost, id<2> idx, int width, int height);
void CostAggregateKernel(int* cost, int* aggregatedCost, id<2> idx, int width, int height);
void WTAKernel(int* costVolumn, uint8_t* disparityMap, id<2> idx, int width, int height);

#endif
