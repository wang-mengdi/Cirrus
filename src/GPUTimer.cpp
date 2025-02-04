#include "GPUTimer.h"

void GPUTimer::start(void)
{
	cudaEventRecord(mStart);
}

float GPUTimer::stop(const std::string& output_message)
{
	cudaEventRecord(mStop);
	cudaEventSynchronize(mStop);
	float time;
	cudaEventElapsedTime(&time, mStart, mStop);
	fmt::print("{}: {}ms\n", output_message, time);
	return time;
}
