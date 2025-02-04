//////////////////////////////////////////////////////////////////////////
// Metadata
// Copyright (c) (2022-), Mengdi Wang, Yuchen Sun
// This file is part of MESO, whose distribution is governed by the LICENSE file.
//////////////////////////////////////////////////////////////////////////
#pragma once
#include "Common.h"
#include "CPUTimer.h"
#include <queue>
#include <thread>
//#include <fstream>

class DriverMetaData
{
public:
	std::string output_base_dir;
	fs::path base_path;
	//std::ofstream data_output;


	//fixed part
	int fps = 25;
	double cfl = 1.0;
	double time_per_frame = 0.04;
	double min_step_frame_fraction = 0;	//if set to 0.1, it means the minimal iteration time is 0.1*time_per_frame
	int first_frame;
	int last_frame;
	int snapshot_stride = 0;

	int output_queue_size = 10;

	//queue of threads for output
	std::queue<std::shared_ptr<std::thread>> output_threads;

	//fill for every time step
	int current_frame;
	double current_time;
	double dt;
	double running_cfl;//actual cfl number of simulation

	~DriverMetaData();
	double Time_At_Frame(int frame);
	void Init(json& j);
	void Append_Output_Thread(std::shared_ptr<std::thread> thread_ptr);

	//path of an output .vts file at current frame, for example pressure, velocity
	fs::path Current_VTS_Path(const std::string identifier);
	fs::path Current_VTU_Path(const std::string identifier);
	fs::path Current_OBJ_Path(const std::string identifier);

	//snapshot things
	bool Should_Snapshot(void);
	fs::path Snapshot_Base_Path(void);
	fs::path Snapshot_Path(int frame);
	fs::path Current_Snapshot_Path(void);//snapshot path of current frame
	int Last_Snapshot_Frame(int start_frame);//last snapshotted frame < start_frame
};