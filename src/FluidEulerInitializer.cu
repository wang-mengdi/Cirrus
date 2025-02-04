#include "FluidEuler.h"
#include "FluidEulerInitializer.h"
#include "Driver.h"

//template<class T>
class FluidEulerInitializer {
public:

	void Apply(json& j_all, FluidEuler& fluid) {
		json& j = j_all.at("scene");
		fluid.init(j);
	}
};



void Run_FluidEuler(json& j) {
	FluidEuler fluid;
	FluidEulerInitializer scene;
	DriverFunc::Initialize_And_Run(j, scene, fluid);
}   