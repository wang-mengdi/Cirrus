#include "Common.h"
#include "NumericalTests.h"
#include "FlowMapTests.h"
#include "FluidEulerInitializer.h"
#include "SolverTests.h"
#include <vector>
#include <fmt/core.h>

#include <vtkAMRBox.h>

int main(int argc, char** argv) {

	//// Conda environment name and Python script path
	//std::string conda_env = "py311_env";
	//std::string python_script = "C:\\Code\\HASimulator\\gen_flamingo.py  C:\\Code\\HASimulator\\data\\flamingo-flock-anim --scale 256 --isovalue 0.002 --output_path c:\\Code\\HASimulator\\data\\flamingo_bin";

	//// Command to activate conda environment and execute the Python script
	//std::string command = fmt::format(
	//	"conda run -n py311_env python \"C:\\Code\\HASimulator\\gen_flamingo.py\" \"C:\\Code\\HASimulator\\data\\flamingo-flock-anim\" --scale 256 --isovalue 0.002 --output_path \"C:\\Code\\HASimulator\\data\\flamingo_bin\""
	//);



	//// Execute the command
	//int ret_code = std::system(command.c_str());

	//if (ret_code == 0) {
	//	fmt::print("Python script executed successfully.\n");
	//}
	//else {
	//	fmt::print("Failed to execute the Python script. Return code: {}\n", ret_code);
	//}

	//return 0;

	//vtkAMRBox box(0, 0, 0, 7, 7, 7);
	//fmt::print("box invalid: {}\n", box.IsInvalid());

	//TestIOTime(1);

	//TestMemoryUsage();
	//return 0;
	
	//for (int grid_case = 0; grid_case < 7; grid_case++) {
	//	TestVelocityJacobian(grid_case);
	//}

	//TestBinaryFileIO(4);
	//TestCertainAdvection();
	//for (int i = 24200; i <= 24255; i++) {
	//	TestCertainAdvection(i);
	//}
	//TestCertainAdvection(24246);
	//TestCertainAdvection(24255);
	//return 0;

	//for (int grid_case = 0; grid_case < 7; grid_case++) {
	
	//TestAnalyticalFlowMapAdvectionVariableTime();
	//TestAnalyticalFlowMapAdvectionFixedTime();
	

	//for (int grid_case : {0,1,2,3,4,5,6}) {
	//	TestFlowMapAdvection(grid_case);
	//}

	//TestParticleToGridTransfer(1);
	
	//for (int grid_case : {0, 1, 4}) {
	//for (int grid_case : {0, 1}){
	//	TestPFMAndNFM(grid_case);
	//}

	//TestLeapFrogAdaptiveFlowMapAdvection();
	//TestLeapFrogAdaptiveNFMVorticityAdvection();

	//for (int i = 0; i < 7; i++) {
	//	TestP2GTransfer(i);
	//}
	
	//for (int i : {0,1,4}) {
	//	TestG2PAdvectionEffeciency(i);
	//}

	for (int i : {0,7,2}) {
		//TestG2PAdvectionEffeciency(i);
		TestParticleToGridTransferEfficiency(i);
	}
	
	
	//for (int i : {3}) {
	//	FlowMapTests::TestParticleAdvectionWithNFMInterpolation(i);
	//}

	////for (int i : {0,1,2,3,4,5,6,7}) {
	//for (int i : {8,0,9,7,2}){
	//	//TestNeumannBC(i);
	//	FlowMapTests::TestFlowMapAdvection(i);
	//	fmt::print("====================================\n");
	//}

	return 0;


	try {
		json j = {
			{
				"driver",
				{
					{"last_frame",10}
				}
			},
			{"scene",json::object()}
		};
		if (argc > 1) {
			fmt::print("Read json file {}\n", argv[1]);
			std::ifstream json_input(argv[1]);
			json_input >> j;
			json_input.close();
		}

		std::string simulator = Json::Value(j, "simulator", std::string("euler"));

		if (simulator == "euler") {
			Run_FluidEuler(j);
		}

	}
	catch (nlohmann::json::exception& e)
	{
		fmt::print("json exception {}\n", e.what());
	}

	return 0;
}
