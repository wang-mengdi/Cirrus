#pragma once

#include "Common.h"

#include <cuda_runtime.h>
#include <NanoVDB.h>


#if defined(__CUDACC__) || defined(__HIP__)
// Only define __hostdev__ when using NVIDIA CUDA or HIP compiler
#define __hostdev__ __host__ __device__
#else
#define __hostdev__
#endif

// CUDA programming
enum DataHolder { HOST = 0, DEVICE };

enum TileTypes { LEAF = 0b001, NONLEAF = 0b010, GHOST = 0b100, NEWLY_ADDED = 0b1000 };

//enum LaunchPolicy { EXEC_LEAF = 0b01, EXEC_NONLEAF = 0b10, EXEC_ABOVE = 0b100, EXEC_FINEFIRST = 0b1000 };
enum LaunchMode { LAUNCH_LEVEL = 0, LAUNCH_SUBTREE };
enum LaunchOrder { COARSE_FIRST = 0, FINE_FIRST };

template<>
struct fmt::formatter<nanovdb::Coord> {
	constexpr auto parse(fmt::format_parse_context& ctx) -> decltype(ctx.begin()) {
		//https://fmt.dev/latest/api.html#udt
		auto it = ctx.begin(), end = ctx.end();
		if (it != end && *it != '}') throw format_error("invalid format");

		// Return an iterator past the end of the parsed range:
		return it;
	}

	// Formats the point p using the parsed format specification (presentation)
	// stored in this formatter.
	template <typename FormatContext>
	auto format(const nanovdb::Coord& coord, FormatContext& ctx) -> decltype(ctx.out()) {
		return format_to(ctx.out(), "[{},{},{}]", coord.x(), coord.y(), coord.z());
	}
};

template<class T>
struct fmt::formatter<nanovdb::Vec3<T>> {
	constexpr auto parse(fmt::format_parse_context& ctx) -> decltype(ctx.begin()) {
		//https://fmt.dev/latest/api.html#udt
		auto it = ctx.begin(), end = ctx.end();
		if (it != end && *it != '}') throw format_error("invalid format");

		// Return an iterator past the end of the parsed range:
		return it;
	}

	// Formats the point p using the parsed format specification (presentation)
	// stored in this formatter.
	template <typename FormatContext>
	auto format(const nanovdb::Vec3<T>& v, FormatContext& ctx) -> decltype(ctx.out()) {
		return format_to(ctx.out(), "[{},{},{}]", v[0], v[1], v[2]);
	}
};

template<class Tile>
class HATileInfo {
public:
	using T = typename Tile::T;
	using Coord = typename Tile::Coord;

	Coord mTileCoord;
	Tile* mTilePtr;
	int mLevel;
	uint8_t mType;

	__hostdev__ HATileInfo() :mType(0), mLevel(0), mTilePtr(nullptr), mTileCoord(0, 0, 0) {}
	__hostdev__ HATileInfo(uint8_t _type, uint8_t _level, Tile* _tilePtr, const Coord& _tileCoord) :
		mType(_type), mLevel(_level), mTilePtr(_tilePtr), mTileCoord(_tileCoord)
	{}

	__hostdev__ bool empty(void)const { return mTilePtr == nullptr; }
	//will consider non-leafs as leafs at subtree_level
	//not that this function will still return 0 if the tile is empty
	__hostdev__ uint8_t subtreeType(const int subtree_level) const {
		if (mLevel == subtree_level && mType == NONLEAF) {
			return LEAF;
		}
		return mType;
	}
	//isLeaf(), isGhost() will return false if the tile is empty
	__hostdev__ bool isLeaf(const int subtree_level = -1)const { return subtreeType(subtree_level) & LEAF; }
	__hostdev__ bool isGhost(void)const { return mType & GHOST; }
	__hostdev__ bool isActive(void)const { return mType & (LEAF | NONLEAF); }//is leaf or non-leaf, not ghost
	__hostdev__ Tile& tile(void)const { return *mTilePtr; }
	__host__ Tile getTile(DataHolder side) const {
		Tile data;
		if (side == DEVICE) cudaMemcpy(&data, mTilePtr, sizeof(Tile), cudaMemcpyDeviceToHost);
		else std::memcpy(&data, mTilePtr, sizeof(Tile));
		return data;
	}
};