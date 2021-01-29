#include "headers.h"
#include "StereoMatchingBasic.h"

//SYCL_EXTERNAL int SAD(uint8_t* imgL, uint8_t* imgR, int i, int j, int d, int width, int height, int win_width = WIN_WIDTH);
extern bool warm_up;

void StereoMatchingBasic(uint8_t* imgL, uint8_t* imgR, uint8_t* out, int width, int height) {
	int image_size = width * height;
	int* cost = (int*)malloc(image_size * MAX_DIS * sizeof(int));
	int* aggregatedCost = (int*)malloc(image_size * MAX_DIS * sizeof(int));

	sycl::queue q(default_selector{}, exception_handler);
	if (warm_up) {
		std::cout << "Running on "
			<< q.get_device().get_info<sycl::info::device::name>() << "\n";
	}

	{
		buffer imgl_buf(imgL, range<1>(image_size));
		buffer imgr_buf(imgR, range<1>(image_size));
		buffer cost_buf(cost, range<1>(image_size * MAX_DIS));
		buffer aggregated_cost_buf(aggregatedCost, range<1>(image_size * MAX_DIS));
		buffer outdata_buf(out, range<1>(image_size));

		q.submit([&](handler& h) {
			auto imgl_acc = imgl_buf.get_access(h, read_only);
			auto imgr_acc = imgr_buf.get_access(h, read_only);
			auto cost_acc = cost_buf.get_access(h, write_only);

			h.parallel_for(
				range<2>(height, MAX_DIS), [=](id<2> idx) {
				CostComputeKernel(imgl_acc.get_pointer(), imgr_acc.get_pointer(), cost_acc.get_pointer(), idx, width, height);
			});
		});

		q.submit([&](handler& h) {
			accessor cost_acc = cost_buf.get_access(h, read_only);
			accessor aggregated_cost_acc = aggregated_cost_buf.get_access(h, write_only);

			h.parallel_for(
				range<2>(height, MAX_DIS), [=](id<2> idx) {
				CostAggregateKernel(cost_acc.get_pointer(), aggregated_cost_acc.get_pointer(), idx, width, height);
			});
		});

		q.submit([&](handler& h) {
			accessor aggregated_cost_acc = aggregated_cost_buf.get_access(h, read_only);
			accessor o_acc = outdata_buf.get_access(h, write_only);

			h.parallel_for(
				range<2>(height, width), [=](id<2> idx) {
				WTAKernel(aggregated_cost_acc.get_pointer(), o_acc.get_pointer(), idx, width, height);
			});
		});

		q.wait_and_throw();
	}

	free(cost);
	free(aggregatedCost);
}

int SAD2(uint8_t* imgL, uint8_t* imgR, int i, int j, int d, int width, int height, int win_width) {

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

void CostComputeKernel(uint8_t* imgL, uint8_t* imgR, int* cost, id<2> idx, int width, int height) {
	int i = idx[0];
	int d = idx[1];
	for (int j = 0; j < width; j++) {
		cost[(i*width + j)*MAX_DIS + d] = SAD2(imgL, imgR, i, j, d, width, height, 1);
	}

}

void CostAggregateKernel(int* cost, int* aggregatedCost, id<2> idx, int width, int height) {
	int i = idx[0];
	int d = idx[1];
	for (int j = 0; j < width; j++) {
		int ind1 = (i * width + j) * MAX_DIS + d;
		aggregatedCost[ind1] = 0;
		for (int w = -WIN_WIDTH / 2; w <= WIN_WIDTH / 2; w++) {
			int ind2 = (i * width + j + w) * MAX_DIS + d;
			int c = ((ind2 >= i * width * MAX_DIS) && (ind2 < (i + 1) * width * MAX_DIS)) ? cost[ind2] : 0;
			aggregatedCost[ind1] += c;
		}
	}
}

void WTAKernel(int* costVolumn, uint8_t* disparityMap, id<2> idx, int width, int height) {
	int i = idx[0];
	int j = idx[1];
	int temp_min = INT_MAX, temp_disp = 0;
	for (int d = 0; d < MAX_DIS; d++) {
		if (costVolumn[(i*width + j)*MAX_DIS + d] < temp_min) {
			temp_min = costVolumn[(i*width + j)*MAX_DIS + d];
			temp_disp = d;
		}
	}
	disparityMap[i*width + j] = temp_disp;
}