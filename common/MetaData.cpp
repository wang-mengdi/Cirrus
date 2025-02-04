#include "MetaData.h"

DriverMetaData::~DriverMetaData() {
	while (!output_threads.empty()) {
		auto join_ptr = output_threads.front();
		output_threads.pop();
		join_ptr->join();
	}
}

double DriverMetaData::Time_At_Frame(int frame) {
	return frame * time_per_frame;
}

void DriverMetaData::Init(json& j) {
	output_base_dir = Json::Value(j, "output_base_dir", std::string("output"));
	//base_path = fs::current_path() / fs::path(output_base_dir);
	base_path = fs::path(output_base_dir);

	fps = Json::Value(j, "fps", 25);
	cfl = Json::Value(j, "cfl", (double)1.0);
	time_per_frame = 1.0 / fps;
	min_step_frame_fraction = Json::Value(j, "min_step_frame_fraction", (double)0);

	first_frame = Json::Value(j, "first_frame", 0);
	last_frame = Json::Value(j, "last_frame", fps * 10);
	snapshot_stride = Json::Value(j, "snapshot_stride", 0);

	output_queue_size = Json::Value(j, "queue_size", 10);
}

void DriverMetaData::Append_Output_Thread(std::shared_ptr<std::thread> thread_ptr) {
	while (output_threads.size() > output_queue_size) {
		auto join_ptr = output_threads.front();
		output_threads.pop();
		join_ptr->join();
	}
	output_threads.push(thread_ptr);
}

fs::path DriverMetaData::Current_VTS_Path(const std::string identifier)
{
	return base_path / fs::path(fmt::format("{}{:04d}.vts", identifier, current_frame));
}

fs::path DriverMetaData::Current_VTU_Path(const std::string identifier)
{
	return base_path / fs::path(fmt::format("{}{:04d}.vtu", identifier, current_frame));
}

fs::path DriverMetaData::Current_OBJ_Path(const std::string identifier)
{
	return base_path / fs::path(fmt::format("{}{:04d}.obj", identifier, current_frame));
}

bool DriverMetaData::Should_Snapshot(void)
{
	return current_frame != 0 && snapshot_stride != 0 && current_frame % snapshot_stride == 0;
}

fs::path DriverMetaData::Snapshot_Base_Path(void)
{
	return base_path / fs::path("snapshots");
}

fs::path DriverMetaData::Snapshot_Path(int frame)
{
	return Snapshot_Base_Path() / fs::path(fmt::format("{:04d}", frame));
}

fs::path DriverMetaData::Current_Snapshot_Path(void)
{
	return Snapshot_Path(current_frame);
}

int DriverMetaData::Last_Snapshot_Frame(int start_frame)
{
	fs::path snap_base = Snapshot_Base_Path();
	if (!fs::is_directory(snap_base)) return 0;//no snapshots are there

	std::vector<int> snapshots;
	for (fs::directory_iterator itr(snap_base); itr != fs::directory_iterator(); ++itr) {
		if (fs::is_directory(itr->status())) {
			std::string filename = itr->path().filename().stem().string();
			snapshots.push_back(std::stoi(filename));
		}
	}

	std::sort(snapshots.begin(), snapshots.end());
	//find the first element >= start_frame
	auto it = std::lower_bound(snapshots.begin(), snapshots.end(), start_frame);
	if (it != snapshots.begin()) {
		it--;
		return *it;
	}
	else {
		return 0;//no snapshot is read
	}
}
