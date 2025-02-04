//////////////////////////////////////////////////////////////////////////
// Simulator Driver
// Copyright (c) (2022-), Bo Zhu, Mengdi Wang
// This file is part of MESO, whose distribution is governed by the LICENSE file.
//////////////////////////////////////////////////////////////////////////
#pragma once

#include "Simulator.h"
#include "MetaData.h"

class DriverTimer {
public:
	std::chrono::time_point<std::chrono::system_clock> total_start;
	std::chrono::time_point<std::chrono::system_clock> lap_start;
	std::chrono::time_point<std::chrono::system_clock> loop_start;

	DriverTimer() {
		Reset();
	}
	void Reset(void);
	//total time
	double Total_Time(const double unit = PhysicalUnits::s);
	//lap time, and reset the lap clock
	double Lap_Time(const double unit = PhysicalUnits::s);
};

namespace DriverFunc {	//will change timer
	double Print_Frame_Info(DriverTimer& frame_timer, const DriverMetaData& meta_data, int num_steps);
	//will change timer
	void Print_Iteration_Info(DriverTimer& iteration_timer, const double dt, const double running_cfl, const double current_time, const double frame_time);

	//simulate from first_frame to last_frame
	//output frames [first_frame, last_frame]
	void Advance(Simulator& simulator, DriverMetaData& meta_data);

	template<class Initializer, class TSimulator>
	void Initialize_And_Run(json& j, Initializer& scene, TSimulator& simulator) {
		Info("Driver::Initialize_And_Run parse json: \n{}", j.dump(2));
		DriverMetaData meta_data;
		meta_data.Init(j.at("driver"));
		scene.Apply(j, simulator);
		fs::create_directories(meta_data.output_base_dir);
		//FileFunc::CreateDirectory(meta_data.output_base_dir);
		fs::path dump_file = fs::path(meta_data.output_base_dir) / fs::path("config.json");
		std::ofstream dump_output(dump_file.string());
		dump_output << std::setw(4) << j;
		dump_output.close();
		Advance(simulator, meta_data);
	}
};