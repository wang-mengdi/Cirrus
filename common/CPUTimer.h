//////////////////////////////////////////////////////////////////////////
// Timer
// Copyright (c) (2022-), Mengdi Wang
// This file is part of MESO, whose distribution is governed by the LICENSE file.
//////////////////////////////////////////////////////////////////////////
#pragma once
#include <chrono>
#include <ctime>
#include <fmt/core.h>

template <typename Accuracy = std::chrono::milliseconds>
class CPUTimer 
{
	std::chrono::high_resolution_clock::time_point mStart;
public:
	CPUTimer() {}
	void start(void) {
		mStart = std::chrono::high_resolution_clock::now();
	}
	void restart(const std::string& msg) {
		stop(msg);
		start();
	}
	double stop(void) {
		auto end = std::chrono::high_resolution_clock::now();
		auto diff = std::chrono::duration_cast<Accuracy>(end - mStart).count();
		return diff;
	}
	double stop(const std::string& msg)
	{
		auto end = std::chrono::high_resolution_clock::now();
		auto diff = std::chrono::duration_cast<Accuracy>(end - mStart).count();
		fmt::print("{}: {}", msg, diff);
		if (std::is_same<Accuracy, std::chrono::microseconds>::value) {// resolved at compile-time
			fmt::print("us\n");
		}
		else if (std::is_same<Accuracy, std::chrono::milliseconds>::value) {
			fmt::print("ms\n");
		}
		else if (std::is_same<Accuracy, std::chrono::seconds>::value) {
			fmt::print("s\n");
		}
		else {
			fmt::print("unknown time unit\n");
		}
		return diff;
	}
};


