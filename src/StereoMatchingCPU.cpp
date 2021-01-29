#include "headers.h"
#include "StereoMatchingCPU.h"

void StereoMatchingCPU(uint8_t* imgL, uint8_t* imgR, uint8_t* out, int width, int height) {
	int* cost = (int*)malloc(width*height*MAX_DIS * sizeof(int));
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			for (int d = 0; d < MAX_DIS; d++) {
				cost[(i*width + j)*MAX_DIS + d] = SAD(imgL, imgR, i, j, d, width, height, 1);
			}
		}
	}

	int* aggregatedCost = (int*)malloc(width*height*MAX_DIS * sizeof(int));
	aggregateCostHorizontal(cost, aggregatedCost, width, height);

	WTA(aggregatedCost, out, width, height);

	free(cost);
	free(aggregatedCost);
}


int SAD(uint8_t* imgL, uint8_t* imgR, int i, int j, int d, int width, int height, int win_width) {

	int sum = 0;
	for (int m = -WIN_HEIGHT / 2; m <= WIN_HEIGHT / 2; m++) {
		for (int n = -win_width / 2; n <= win_width / 2; n++) {
			uint8_t left = ((i + m >= 0) && (i + m < height) && (j + n >= 0) && (j + n < width)) ? imgL[(i + m) * width + j + n] : 0;
			uint8_t right = ((i + m >= 0) && (i + m < height) && (j + n - d >= 0) && (j + n - d < width)) ? imgR[(i + m) * width + j + n - d] : 0;
			sum += std::abs(int(left) - int(right));
		}
	}
	return sum;
}

void aggregateCostHorizontal(int* cost, int* aggregatedCost, int width, int height) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			for (int d = 0; d < MAX_DIS; d++) {
				int ind1 = (i * width + j) * MAX_DIS + d;
				aggregatedCost[ind1] = 0;
				for (int w = -WIN_WIDTH / 2; w <= WIN_WIDTH / 2; w++) {
					int ind2 = (i * width + j + w) * MAX_DIS + d;
					int c = ((ind2 >= i * width * MAX_DIS) && (ind2 < (i + 1) * width * MAX_DIS)) ? cost[ind2] : 0;
					aggregatedCost[ind1] += c;
				}
			}
		}
	}
}

void aggregateCostHorizontalBorder(int* cost, int* aggregatedCost, int width, int height, uint8_t* imgR) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			for (int d = 0; d < MAX_DIS; d++) {
				int ind1 = (i * width + j) * MAX_DIS + d;
				aggregatedCost[ind1] = 0;
				for (int w = -WIN_WIDTH / 2; w <= WIN_WIDTH / 2; w++) {
					int sum = 0;
					for (int h = -WIN_HEIGHT / 2; h <= WIN_HEIGHT / 2; h++) {
						uint8_t gray = ((i + h >= 0) && (i + h < height) && (j + w - d >= 0) && (j + w - d < width)) ? imgR[(i + h) * width + j + w - d] : 0;
						sum += gray;
					}
					int ind2 = (i * width + j + w) * MAX_DIS + d;
					int c = ((ind2 >= i * width * MAX_DIS) && (ind2 < (i + 1) * width * MAX_DIS)) ? cost[ind2]
						: ((ind2 >= (i + 1) * width * MAX_DIS) ? sum : 0);
					aggregatedCost[ind1] += c;
				}
			}
		}
	}
}

void WTA(int* costVolumn, uint8_t* disparityMap, int width, int height) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int disp = 0;
			int minCost = costVolumn[(i*width + j)*MAX_DIS];
			for (int d = 1; d < MAX_DIS; d++) {
				if (costVolumn[(i*width + j)*MAX_DIS + d] < minCost) {
					minCost = costVolumn[(i*width + j)*MAX_DIS + d];
					disp = d;
				}
			}
			disparityMap[i*width + j] = disp/* * 256 / MAX_DIS*/;
		}
	}
}