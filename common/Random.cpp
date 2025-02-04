#include "Random.h"

int RandomGenerator::rand(int a, int b) {
	std::uniform_int_distribution<int> uid(a, b);
	return uid(sequential_gen);
}

double RandomGenerator::uniform(double a, double b) {
    std::uniform_real_distribution<double> uid(a, b);
    return uid(sequential_gen);
}
