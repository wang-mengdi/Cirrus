#include "Driver.h"
#include <fstream>

//timer
void DriverTimer::Reset(void)
{
	total_start = std::chrono::system_clock::now();
	lap_start = std::chrono::system_clock::now();
}

double DriverTimer::Total_Time(const double unit)
{
	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	std::chrono::duration<double, std::ratio<1> > elapse = end - total_start;
	return elapse.count() / unit;
}

double DriverTimer::Lap_Time(const double unit)
{
	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	std::chrono::duration<double, std::ratio<1> > elapse = end - lap_start;
	lap_start = std::chrono::system_clock::now();

	return elapse.count() / unit;
}

namespace DriverFunc {

	double Print_Frame_Info(DriverTimer& frame_timer, const DriverMetaData& meta_data, int num_steps) {
		int total_frames = meta_data.last_frame - meta_data.first_frame;
		int done_frames = meta_data.current_frame - meta_data.first_frame;
		auto frame_seconds = frame_timer.Lap_Time();
		auto completed_seconds = frame_timer.Total_Time();
		auto eta = completed_seconds * (total_frames - done_frames) / done_frames;
		Info("Frame {} in {}-{} done in {:.3f}s with {} substeps, ETA {:.3f}/{:.3f}s", meta_data.current_frame, meta_data.first_frame, meta_data.last_frame, frame_seconds, num_steps, eta, completed_seconds + eta);
		return frame_seconds;
	}

	void Print_Iteration_Info(DriverTimer& iteration_timer, const double dt, const double running_cfl, const double current_time, const double frame_time)
	{
		auto step_seconds = iteration_timer.Lap_Time();
		auto completed_seconds = iteration_timer.Total_Time();
		Info("Iteration {:.5f}/{:.5f}s with CFL={:.3f}, cost {:.3f}s, remaining {:.3f}s", dt, frame_time, running_cfl, step_seconds, completed_seconds * (frame_time - current_time) / current_time);
	}

	void Advance(Simulator& simulator, DriverMetaData& meta_data) {
		DriverTimer frame_timer;
		fs::create_directories(meta_data.base_path);
		//FileFunc::CreateDirectory(meta_data.base_path);

		//try to load snapshot
		if (meta_data.first_frame != 0) {
			int last_snapshot = meta_data.Last_Snapshot_Frame(meta_data.first_frame);
			if (last_snapshot != 0) {
				Info("Found snapshot at frame {}, load snapshot from {} and run from frame {}", last_snapshot, meta_data.Snapshot_Path(last_snapshot).string(), last_snapshot + 1);
				meta_data.current_frame = last_snapshot;
				//load the frame before first frame
				simulator.Load_Frame(meta_data);
				meta_data.current_frame++;
			}
			else {
				Info("No snapshots are found, automaitcally run from frame 0");
				meta_data.first_frame = 0;
			}
		}

		//run from 0
		if (meta_data.first_frame == 0) {
			//output first frame
			meta_data.current_frame = 0;
			Print_Frame_Info(frame_timer, meta_data, 0);
			Info("Output frame {} to {}", meta_data.current_frame, meta_data.base_path.string());
			simulator.Output(meta_data);
			meta_data.current_frame = 1;
		}

		//run frames
		for (int& current_frame = meta_data.current_frame; current_frame <= meta_data.last_frame; current_frame++) {
			DriverTimer iter_timer;
			int num_steps = 0;
			int next_frame = current_frame + 1;
			meta_data.current_time = meta_data.Time_At_Frame(current_frame);
			auto frame_start_time = meta_data.current_time;
			auto next_time = meta_data.Time_At_Frame(next_frame);
			while (true) {
				//can return an inf
				auto math_dt = simulator.CFL_Time(meta_data.cfl);
				meta_data.dt = std::clamp(math_dt, meta_data.min_step_frame_fraction * meta_data.time_per_frame, meta_data.time_per_frame);
				meta_data.running_cfl = meta_data.cfl * meta_data.dt / math_dt;
				bool last_iter = false;
				if (meta_data.current_time + meta_data.dt >= next_time) {
					meta_data.dt = next_time - meta_data.current_time;
					last_iter = true;
				}
				else if (meta_data.current_time + 2 * meta_data.dt >= next_time) {
					meta_data.dt = .5 * (next_time - meta_data.current_time);
				}

				simulator.Advance(meta_data);
				num_steps++;
				meta_data.current_time += meta_data.dt;
				Print_Iteration_Info(iter_timer, meta_data.dt, meta_data.running_cfl, meta_data.current_time - frame_start_time, meta_data.time_per_frame);
				if (last_iter) break;
			}

			// output current frame
			double frame_time = Print_Frame_Info(frame_timer, meta_data, num_steps);
			Info("Output frame {} to {}", meta_data.current_frame, meta_data.base_path.string());
			simulator.Output(meta_data);

			// log frame information
			fs::path log_dir = meta_data.base_path / "logs";
			fs::create_directories(log_dir);
			fs::path log_file = log_dir / fmt::format("frame_driver_info{:04d}.txt", current_frame);

			std::ofstream log_stream(log_file.string());
			if (log_stream.is_open()) {
				log_stream << "Frame: " << current_frame << "\n";
				log_stream << "Iterations: " << num_steps << "\n";
				log_stream << "Frame time (seconds): " << frame_time << "\n";
				log_stream.close();
			}
			else {
				Error("Failed to create log file: {}", log_file.string());
			}
			Info("==========================================");
		}
	}

}