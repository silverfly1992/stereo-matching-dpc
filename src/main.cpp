//==============================================================
// Copyright ?2021 silverfly
//
// 01.22.2021 Hefei University of Technology
// =============================================================
#include "headers.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

//#include <opencv2/opencv.hpp>

#include "StereoMatchingCPU.h"
#include "StereoMatchingBasic.h"


#define PERF_NUM

#ifdef PERF_NUM
constexpr int num_tests = 5;
#endif

bool warm_up = false;

void StereoMatchingND(uint8_t* imgL, uint8_t* imgR, uint8_t* disparity_map, int width, int height) {
	sycl::queue q(default_selector{}, exception_handler);
	if (warm_up) {
		std::cout << "Running on "
			<< q.get_device().get_info<sycl::info::device::name>() << "\n";
	}

	// Query results like the following can be used to calculate how
	// large our kernel invocations should be.
	if (warm_up) {
		device dev = q.get_device();
		auto maxWG = dev.get_info<info::device::max_work_group_size>();
		auto maxGmem = dev.get_info<info::device::global_mem_size>();
		auto maxLmem = dev.get_info<info::device::local_mem_size>();
		std::cout << "Max WG size is " << maxWG
			<< "\nMax Global memory size is " << maxGmem
			<< "\nMax Local memory size is " << maxLmem << "\n";
	}

	int image_size = width * height;

	try {
		uint8_t* d_imgL = malloc_device<uint8_t>(image_size, q);
		uint8_t* d_imgR = malloc_device<uint8_t>(image_size, q);
		uint8_t* d_disparity = malloc_device<uint8_t>(image_size, q);
		int* d_SADCostVertical = malloc_device<int>(image_size*MAX_DIS, q);
		int* d_aggregatedCost = malloc_device<int>(image_size*MAX_DIS, q);

		auto e1 = q.memcpy(d_imgL, imgL, image_size);
		auto e2 = q.memcpy(d_imgR, imgR, image_size);

		auto e3 = q.submit([&](handler& h) {
			h.depends_on({ e1,e2 });

			accessor<int, 2, access::mode::read_write, access::target::local> tileMatch(range<2>(WIN_HEIGHT, MAX_DIS*2), h);
			accessor<int, 2, access::mode::read_write, access::target::local> tileBase(range<2>(WIN_HEIGHT, MAX_DIS), h);
			
			h.parallel_for(nd_range<2>({ (unsigned long)(height + WIN_HEIGHT -1) / WIN_HEIGHT, MAX_DIS }, { 1, MAX_DIS }), [=](nd_item<2> item) {
				int basei = item.get_global_id()[0];
				for (int m = 0; m < WIN_HEIGHT; m++) {
					int i = basei * WIN_HEIGHT + m;
					int j = item.get_local_id()[1];
					//if (i < height) {
						for (int q = -WIN_HEIGHT / 2; q <= WIN_HEIGHT / 2; q++) {
							tileMatch[q + WIN_HEIGHT / 2][j + MAX_DIS] = 0;
						}

						for (int n = 0; n < width; n += MAX_DIS) {
							for (int q = -WIN_HEIGHT / 2; q <= WIN_HEIGHT / 2; q++) {
								tileMatch[q + WIN_HEIGHT / 2][j] = tileMatch[q + WIN_HEIGHT / 2][j + MAX_DIS];
								tileMatch[q + WIN_HEIGHT / 2][j + MAX_DIS] = ((i + q >= 0) && (i + q < height)) ? d_imgR[(i + q)*width + n + j] : 0;
								tileBase[q + WIN_HEIGHT / 2][j] = ((i + q >= 0) && (i + q < height)) ? d_imgL[(i + q)*width + n + j] : 0;

							}
							item.barrier();

							for (int r = 0; r < MAX_DIS; r++) {
								int ind = (i*width + n + r)*MAX_DIS + j;
								d_SADCostVertical[ind] = 0;
								for (int wh = 0; wh < WIN_HEIGHT; wh++) {
									d_SADCostVertical[ind] += abs(tileBase[wh][r] - tileMatch[wh][MAX_DIS + r - j]);
								}
							}
							item.barrier();
						}
					//}
				}
			});

		});

		auto e4 = q.submit([&](handler& h) {
			h.depends_on(e3);

			int block_size = 2; //maxWG=256
			h.parallel_for(nd_range<2>({ (unsigned long)height, MAX_DIS }, { (unsigned long)block_size, MAX_DIS }), [=](nd_item<2> item) {
				int i = item.get_global_id()[0];
				int d = item.get_local_id()[1];
				for (int j = 0; j < width; j++) {
					int ind1 = (i * width + j) * MAX_DIS + d;
					d_aggregatedCost[ind1] = 0;
					for (int w = -WIN_WIDTH / 2; w <= WIN_WIDTH / 2; w++) {
						int ind2 = (i * width + j + w) * MAX_DIS + d;
						int c = ((ind2 >= i * width * MAX_DIS) && (ind2 < (i + 1) * width * MAX_DIS)) ? d_SADCostVertical[ind2] : 0;
						d_aggregatedCost[ind1] += c;
					}
				}
			});

			////  Moving data to work group local memory shows no gains.
			//accessor<int, 2, access::mode::read_write, access::target::local> tile(range<2>(block_size, MAX_DIS*WIN_WIDTH), h);

			//h.parallel_for(nd_range<2>({ height, MAX_DIS }, { block_size, MAX_DIS}), [=](nd_item<2> item) {
			//	
			//	int i = item.get_global_id()[0];
			//	int d = item.get_local_id()[1];

			//	int m = item.get_local_id()[0];
			//	//initialization
			//	for (int ww = 1; ww < 1 + WIN_WIDTH / 2; ww++) {
			//		tile[m][ww*MAX_DIS+d] = 0;
			//	}
			//	for (int ww = 1 + WIN_WIDTH / 2; ww < WIN_WIDTH; ww++) {
			//		tile[m][ww*MAX_DIS + d] = d_SADCostVertical[(i*width+ww-WIN_WIDTH/2-1)*MAX_DIS+d];
			//	}

			//	for (int j = 0; j < width; j++) {
			//		//There is no data order dependency in aggregating cost.
			//		//So overlap data instead of moving data circularly.
			//		//for (int ww = 0; ww < WIN_WIDTH - 1; ww++) {
			//		//	tile[ww][d] = tile[ww + 1][d];
			//		//}
			//		tile[m][j%WIN_WIDTH*MAX_DIS+d] = (j + WIN_WIDTH / 2 < width) ? d_SADCostVertical[(i*width + j + WIN_WIDTH / 2)*MAX_DIS + d] : 0;
			//		item.barrier();

			//		int ind = (i*width + j)*MAX_DIS + d;
			//		for (int ww = 0; ww < WIN_WIDTH; ww++) {
			//			d_aggregatedCost[ind] += tile[m][ww*MAX_DIS + d];
			//		}

			//		item.barrier();

			//	}

			//});
		});



		auto e5 = q.submit([&](handler& h) {
			h.depends_on(e4);

			h.parallel_for(nd_range<2>({ (unsigned long)height, MAX_DIS }, { 2, MAX_DIS }), [=](nd_item<2> item) {
				int i = item.get_global_id()[0];
				for (int n = 0; n < width; n += MAX_DIS) {
					int j = n + item.get_local_id()[1];
					int temp_min = INT_MAX, temp_disp = 0;
					for (int d = 0; d < MAX_DIS; d++) {
						if (d_aggregatedCost[(i*width + j)*MAX_DIS + d] < temp_min) {
							temp_min = d_aggregatedCost[(i*width + j)*MAX_DIS + d];
							temp_disp = d;
						}
					}
					d_disparity[i*width + j] = temp_disp;
				}
			});
		});


		q.submit([&](handler& h) {
			h.depends_on(e5);
			h.memcpy(disparity_map, d_disparity, image_size);
		});

		q.wait_and_throw();

		free(d_imgL, q);
		free(d_imgR, q);
		free(d_disparity, q);
		free(d_SADCostVertical, q);
		free(d_aggregatedCost, q);

	}
	catch (sycl::exception e) {
		std::cout << "SYCL exception caught: " << e.what() << "\n";
		exit(1);
	}


}
	

int main(int argc, char* argv[]) {
	double timersecs;
#ifdef PERF_NUM
	double avg_timersecs = 0;
#endif

	///***** Read in the data from the input image file *******///
	int image_width = 0, image_height = 0, num_channels = 0;
	uint8_t* indata = (uint8_t*)stbi_load(argv[1], &image_width, &image_height,
		&num_channels, STBI_grey);
	int image_width2 = 0, image_height2 = 0, num_channels2 = 0;
	uint8_t* indata2 = (uint8_t*)stbi_load(argv[2], &image_width2, &image_height2,
		&num_channels2, STBI_grey);


	if (!indata || !indata2) {
		std::cout << "The input file could not be opened. Program will now exit\n";
		return 1;
	}

	std::cout << "Filename: " << argv[1] << " W: " << image_width
		<< " H: " << image_height << "\n\n";

	uint8_t* outdata = (uint8_t*)malloc(image_width * image_height * sizeof(uint8_t));

	// Warm up for speed evaluation.
	warm_up = true;
	StereoMatchingND(indata, indata2, outdata, image_width, image_height);
	warm_up = false;

	///********** Evaluate ND-Range Kernel ***********///
#ifdef PERF_NUM
	std::cout << "\n\nEvaluating ND-Range Kernel...\n\n";
	for (int j = 0; j < num_tests; ++j) {
#endif
		std::cout << "Start image processing with offloading to GPU...\n";
		{
			TimeInterval t;
			StereoMatchingND(indata, indata2, outdata, image_width, image_height);
			timersecs = t.Elapsed();
		}
		std::cout << "--The processing time is " << timersecs << " seconds\n";
#ifdef PERF_NUM
		avg_timersecs += timersecs;
	}
#endif
#ifdef PERF_NUM
	std::cout << "\nAverage time for image processing:\n";
	std::cout << "--The average processing time was "
		<< avg_timersecs / (float)num_tests << " seconds\n";
#endif


	///********** Evaluate Basic Kernel ***********///
	uint8_t* outdata2 = (uint8_t*)malloc(image_width * image_height * sizeof(uint8_t));
	// Warm up for speed evaluation.
	StereoMatchingBasic(indata, indata2, outdata2, image_width, image_height);

#ifdef PERF_NUM
	std::cout << "\n\nEvaluating Basic Kernel...\n\n";
	avg_timersecs = 0;
	for (int j = 0; j < num_tests; ++j) {
#endif
		std::cout << "Start image processing with offloading to GPU...\n";
		{
			TimeInterval t;
			StereoMatchingBasic(indata, indata2, outdata2, image_width, image_height);
			timersecs = t.Elapsed();
		}
		std::cout << "--The processing time is " << timersecs << " seconds\n";
#ifdef PERF_NUM
		avg_timersecs += timersecs;
	}
#endif
#ifdef PERF_NUM
	std::cout << "\nAverage time for image processing:\n";
	std::cout << "--The average processing time was "
		<< avg_timersecs / (float)num_tests << " seconds\n";
#endif

	///******* Compare result *********///
	std::cout << "\n\n/*********Compare Result*********/" << std::endl;
	uint8_t* outdata_gold = (uint8_t*)malloc(image_width * image_height * sizeof(uint8_t));
	{
		TimeInterval t;
		StereoMatchingCPU(indata, indata2, outdata_gold, image_width, image_height);
		timersecs = t.Elapsed();
	}
	int err_num = 0;
	for (int i = 0; i < image_height; i++) {
		for (int j = 0; j < image_width; j++) {
			if (outdata[i*image_width + j] != outdata_gold[i*image_width + j]) {
				std::cout << "disparity[" << i << "][" << j << "] expect " << int(outdata_gold[i*image_width + j]) <<
					" but got " << int(outdata[i*image_width + j]) << '.' << std::endl;
				err_num++;
			}
		}
	}
	if (!err_num) {
		std::cout << "All results Matched!" << std::endl;
	}
	else {
		std::cout << "Totally " << err_num << " errors!" << std::endl;
	}
	std::cout << "--The CPU processing time is " << timersecs << " seconds\n";
	std::free(outdata_gold);

	///****** Display and write result. *****///
	//cv::Mat imgMat(image_height, image_width, CV_8UC1);
	//memcpy(imgMat.data, outdata, image_height*image_width);
	//cv::namedWindow("disparity map", cv::WINDOW_NORMAL);
	//cv::resizeWindow("disparity map", 800, 800 * image_height / image_width);
	//cv::imshow("disparity map",imgMat);
	//cv::waitKey(0);

	stbi_write_bmp(argv[3], image_width, image_height, 1, outdata);
	std::cout << "\n\nStereo matching successfully completed on the device.\n"
		"The processed image has been written to " << argv[3] << "\n";

	///****** Freeing dynamically allocated memory ******///
	stbi_image_free(indata);
	stbi_image_free(indata2);
	std::free(outdata);
	std::free(outdata2);

	return 0;

}
