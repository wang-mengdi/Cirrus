#pragma once

#include <iostream>
#include <fmt/core.h>
#include <cuda_runtime.h>

class GPUTimer {
public:
	cudaEvent_t mStart, mStop;

	GPUTimer() {
		cudaEventCreate(&mStart);
		cudaEventCreate(&mStop);
	}
	~GPUTimer() {
		cudaEventDestroy(mStart);
		cudaEventDestroy(mStop);
	}

	void start(void);
	float stop(const std::string& output_message);
};