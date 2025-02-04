//////////////////////////////////////////////////////////////////////////
// Random number
// Copyright (c) (2022-), Bo Zhu, Mengdi Wang
// This file is part of MESO, whose distribution is governed by the LICENSE file.
//////////////////////////////////////////////////////////////////////////
#pragma once

#include <random>
#include <chrono>


//We will imitate interfaces of Python's random lib here
class RandomGenerator {
public:
	std::mt19937 sequential_gen;
	RandomGenerator() {
		sequential_gen.seed(5489U);
	}

	int rand(int a, int b);
	double uniform(double a, double b);
};