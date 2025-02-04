#include "Common.h"

//namespace FileFunc {
//	void CreateDirectory(const fs::path path)
//	{
//		//recursively
//		try {
//			if (!bf::exists(path))
//				fs::create_directories(path);
//		}
//		catch (std::exception& e) { // Not using fs::filesystem_error since std::bad_alloc can throw too.
//			std::cout << e.what() << std::endl;
//		}
//	}
//
//}

void Info(const std::string& str)
{
	Info(str.c_str());
	//Info(str);
}

void Warn(const std::string& str)
{
	Warn(str.c_str());
}