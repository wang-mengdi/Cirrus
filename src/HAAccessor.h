#pragma once

#include "HATileInfo.h"

template<class Tile>
class HACoordAccessor {
public:
	using Coord = typename Tile::CoordType;
	using Vec = typename Tile::VecType;
	using T = typename Tile::T;
	static constexpr uint32_t LOG2DIM = Tile::LOG2DIM;
	constexpr static uint32_t DIM = 1u << LOG2DIM; // this tile stores (DIM*DIM*DIM) voxels (default 8^3=512)
	static constexpr uint8_t NUMCHILDREN = 1u << 3;
	static constexpr uint8_t NUMNEIGHBORS = 6;

	T mH0;//dx of the zeroth level

	HACoordAccessor() {}
	HACoordAccessor(T h0) :mH0(h0) {}

	__hostdev__ static inline uint8_t rotateAxis(const uint8_t& axis, const int offset) {
		return (axis + offset + 3) % 3;
	}
	//rotate 0 to axis
	__hostdev__ static inline Coord rotateCoord(const uint8_t& axis, const Coord& ijk) {
		Coord ijk_new;
		ijk_new[axis] = ijk[0];
		ijk_new[rotateAxis(axis, 1)] = ijk[1];
		ijk_new[rotateAxis(axis, 2)] = ijk[2];
		return ijk_new;
	}
	__hostdev__ static uint32_t localCoordToOffset(const Coord& offset_ijk) {
		return (offset_ijk[0] << (2 * LOG2DIM)) + (offset_ijk[1] << (LOG2DIM)) + (offset_ijk[2]);
	}
	__hostdev__ static Coord localOffsetToCoord(const uint32_t offset) {
		static constexpr uint32_t MASK = ((1u << LOG2DIM) - 1u);
		Coord ijk;
		ijk[0] = (offset >> (2 * LOG2DIM)) & MASK;
		ijk[1] = (offset >> (1 * LOG2DIM)) & MASK;
		ijk[2] = offset & MASK;
		return ijk;
	}
	__hostdev__ const int localNodeCoordToOffset(const Coord& l_ijk) const {
		return l_ijk[0] * (DIM + 1) * (DIM + 1) + l_ijk[1] * (DIM + 1) + l_ijk[2];
	}
	__hostdev__ const Coord localNodeOffsetToCoord(const int offset) const {
		return Coord(offset / ((DIM + 1) * (DIM + 1)), (offset % ((DIM + 1) * (DIM + 1))) / (DIM + 1), offset % (DIM + 1));
	}

	__hostdev__ static Coord composeGlobalCoord(const Coord& tile_ijk, const Coord& offset_ijk) {
		Coord ijk;
		for (uint32_t i = 0; i < 3; i++) {
			ijk[i] = (tile_ijk[i] << LOG2DIM) + offset_ijk[i];
		}
		return ijk;
	}
	__hostdev__ static void decomposeGlobalCoord(const Coord& ijk, Coord& tile_ijk, Coord& offset_ijk) {
		for (uint32_t i = 0; i < 3; i++) {
			tile_ijk[i] = (ijk[i] >> LOG2DIM);
			offset_ijk[i] = ijk[i] - (tile_ijk[i] << LOG2DIM);
		}
	}
	__hostdev__ static inline Coord localToGlobalCoord(const HATileInfo<Tile>& info, const Coord& l_ijk) {
		return composeGlobalCoord(info.mTileCoord, l_ijk);
	}
	__hostdev__ static void decomposeGlobalCoordToOffset(const Coord& ijk, uint32_t& tile_ijk, uint32_t& local_offset) {
		Coord offset_ijk;
		decomposeGlobalCoord(ijk, tile_ijk, offset_ijk);
		local_offset = localCoordToOffset(offset_ijk);
	}

	__hostdev__ T voxelSize(const uint32_t level) const {
		return mH0 / (1 << level);
	}
	__hostdev__ T voxelSize(const HATileInfo<Tile>& info)const {
		return voxelSize(info.mLevel);
	}

	//l_ijk can range in [0,8]
	__hostdev__ Vec cellCorner(const HATileInfo<Tile>& info, const Coord& l_ijk)const {
		//even if some coordinate is 8, it's still correct
		auto g_ijk = composeGlobalCoord(info.mTileCoord, l_ijk);
		auto h = voxelSize(info);
		return Vec(g_ijk[0], g_ijk[1], g_ijk[2]) * h;
	}
	__hostdev__ Vec cellCenterGlobal(const int level, const Coord& g_ijk)const {
		auto h = voxelSize(level);
		return Vec(g_ijk[0] + 0.5, g_ijk[1] + 0.5, g_ijk[2] + 0.5) * h;
	}
	__hostdev__ Vec cellCenterGlobal(const HATileInfo<Tile>& info, const Coord& g_ijk)const {
		auto h = voxelSize(info);
		return Vec(g_ijk[0] + 0.5, g_ijk[1] + 0.5, g_ijk[2] + 0.5) * h;
	}

	__hostdev__ Vec cellCenter(const HATileInfo<Tile>& info, const Coord& l_ijk) const {
		return cellCenterGlobal(info, composeGlobalCoord(info.mTileCoord, l_ijk));
	}

	__hostdev__ Vec faceCenterGlobal(const int axis, const int level, const Coord& g_ijk)const {
		auto center = cellCenterGlobal(level, g_ijk);
		center[axis] -= 0.5 * voxelSize(level);
		return center;
	}
	__hostdev__ Vec faceCenter(const int axis, const HATileInfo<Tile>& info, const Coord& l_ijk)const {
		auto center = cellCenter(info, l_ijk);
		center[axis] -= 0.5 * voxelSize(info);
		return center;
	}

	__hostdev__ Vec worldToVoxel(const uint32_t level, const Vec& world) const {
		return world / voxelSize(level);
	}

	__hostdev__ void worldToVoxelAndFraction(const uint32_t level, const Vec& world, Coord& ijk, Vec& frac) const {
		auto v = worldToVoxel(level, world);
		ijk = Coord(floor(v[0]), floor(v[1]), floor(v[2]));
		frac = v - Vec(ijk[0], ijk[1], ijk[2]);
	}

	__hostdev__ nanovdb::BBox<Vec> tileBBox(const uint32_t level, const Coord& b_ijk) const {
		auto h = voxelSize(level);
		auto mn = Vec(b_ijk[0], b_ijk[1], b_ijk[2]) * Tile::DIM * h;
		auto mx = Vec(b_ijk[0] + 1, b_ijk[1] + 1, b_ijk[2] + 1) * Tile::DIM * h;
		return nanovdb::BBox<Vec>(mn, mx);
	}
	__hostdev__ nanovdb::BBox<Vec> tileBBox(const HATileInfo<Tile>& info)const {
		return tileBBox(info.mLevel, info.mTileCoord);
	}
	__hostdev__ nanovdb::BBox<Vec> voxelBBox(const uint32_t level, const Coord& b_ijk, const Coord& l_ijk) const {
		auto h = voxelSize(level);
		auto g_ijk = composeGlobalCoord(b_ijk, l_ijk);
		auto mn = Vec(g_ijk[0], g_ijk[1], g_ijk[2]) * h;
		auto mx = Vec(g_ijk[0] + 1, g_ijk[1] + 1, g_ijk[2] + 1) * h;
		return nanovdb::BBox<Vec>(mn, mx);
	}
	__hostdev__ nanovdb::BBox<Vec> voxelBBox(const HATileInfo<Tile>& info, const Coord& l_ijk)const {
		return voxelBBox(info.mLevel, info.mTileCoord, l_ijk);
	}

	//idx in [0,7] indicating which child
	__hostdev__ static inline Coord childIndexToOffset(const uint8_t idx) {
		return Coord(idx & 1, (idx >> 1) & 1, (idx >> 2) & 1);
	}

	//offset ranges from [0,0,0] to [1,1,1] indicating which child
	__hostdev__ static inline Coord childCoord(const Coord& ijk, const Coord& offset) {
		return Coord(ijk[0] * 2 + offset[0], ijk[1] * 2 + offset[1], ijk[2] * 2 + offset[2]);
	}
	__hostdev__ static inline Coord childCoord(const Coord& ijk, const uint8_t idx) {
		return childCoord(ijk, childIndexToOffset(idx));
	}
	//offset ranges from [0,0,0] to [1,1,1] indicating which child
	__hostdev__ static inline Coord childTileCoord(const HATileInfo<Tile>& info, const Coord& offset) {
		return childCoord(info.mTileCoord, offset);
	}

	__hostdev__ static inline Coord parentCoord(const Coord& ijk) {
		return Coord(ijk[0] >> 1, ijk[1] >> 1, ijk[2] >> 1);
	}

	//0~5 for x-,x+, y-, y+, z-, z+
	__hostdev__ static inline Coord neighborOffset(const int nb_idx) {
		Coord ijk(0, 0, 0);
		int axis = nb_idx / 2;
		int sgn = nb_idx % 2 == 0 ? -1 : 1;
		ijk[axis] += sgn;
		return ijk;
	}

	//C means coord, it's f(ijk)
	template<class FuncC>
	__hostdev__ static void iterateNeighborCoords(const Coord& ijk, FuncC f) {
		for (int i = 0; i < NUMNEIGHBORS; i++) {
			f(ijk + neighborOffset(i));
		}
	}

	template<class FuncC>
	__hostdev__ static void iterateChildCoords(const Coord& ijk, FuncC f) {
		for (int i = 0; i < NUMCHILDREN; i++) {
			f(childCoord(ijk, i));
		}
	}
};

template<class Tile>
class HATileAccessor : public HACoordAccessor<Tile> {
	using Base = HACoordAccessor<Tile>;
public:
	using Coord = typename Tile::Coord;
	using Vec = typename Tile::VecType;
	using T = typename Tile::T;

	int mNumLayers;
	int mMaxLevel;

	uint32_t* mLayerNumTiles;
	uint32_t* mLayerLog2Hashs;
	HATileInfo<Tile>** mLayerHashTablePtrs;
	//HATileInfo<Tile>** mLayerTileArrayPtrs;

	__hostdev__ HATileAccessor() {}
	__hostdev__ HATileAccessor(T h0, uint32_t num_layers, int max_level, uint32_t* layer_num_tiles, uint32_t* layer_log2_hashs,
		HATileInfo<Tile>** layer_hash_table_ptrs)://, HATileInfo<Tile>** layer_tile_array_ptrs) :
		HACoordAccessor<Tile>(h0),
		mNumLayers(num_layers),
		mMaxLevel(max_level),
		mLayerNumTiles(layer_num_tiles),
		mLayerLog2Hashs(layer_log2_hashs),
		mLayerHashTablePtrs(layer_hash_table_ptrs)//,
		//mLayerTileArrayPtrs(layer_tile_array_ptrs)
	{ }

	__hostdev__ static uint32_t hash(const Coord& ijk, const uint32_t Log2Hash) {
		constexpr uint32_t factors[3] = { 1u,2654435761u ,805459861u };//InstantNGP

		const uint32_t HASH_MASK = (1u << Log2Hash) - 1u;
		uint32_t result = 0;
		for (uint32_t i = 0; i < 3; i++) {
			result ^= ijk[i] * factors[i];
		}
		return result & HASH_MASK;
	}

	__hostdev__ uint32_t tileIdx(const uint32_t level, const Coord& b_ijk) const {
		if (level < 0 || level >= mNumLayers) return -1;//invalid
		auto Log2Hash = mLayerLog2Hashs[level];
		const uint32_t HASH_MASK = (1u << Log2Hash) - 1u;

		uint32_t key0 = hash(b_ijk, Log2Hash);
		uint32_t key1 = key0;
		do {
			auto& info = mLayerHashTablePtrs[level][key1];
			if (info.empty() || info.mTileCoord == b_ijk) {
				return key1;
			}
			key1++; key1 &= HASH_MASK;
		} while (key1 != key0);
		printf("Erorr: hash table full when trying to access level %d b_ijk %d %d %d\n", level, b_ijk[0], b_ijk[1], b_ijk[2]);
		//Assert(false, "hash table full when trying to access level {} b_ijk {}", level, b_ijk);
		return -1;//invalid
	}

	////if doesn't exist, the returned tile info is empty, but with correct level and coord
	//__hostdev__ HATileInfo<Tile> tileInfo(const uint32_t level, const Coord& b_ijk) const {
	//	if (level < 0 || level >= mNumLayers) return HATileInfo<Tile>();//return an empty tile info for non-exist level
	//	auto idx = tileIdx(level, b_ijk);
	//	if (idx == -1) return HATileInfo<Tile>();//return an empty tile info for non-exist tile
	//	return mLayerHashTablePtrs[level][idx];
	//}

	//doesn't perform valid check
	__hostdev__ HATileInfo<Tile>& tileInfo(const uint32_t level, const Coord& b_ijk) const {
		auto idx = tileIdx(level, b_ijk);
		return mLayerHashTablePtrs[level][idx];
	}

	//perform valid check, return an empty tile info for non-exist level
	__hostdev__ HATileInfo<Tile> tileInfoCopy(const uint32_t level, const Coord& b_ijk) const {
		if (0 <= level && level < mNumLayers) {
			auto idx = tileIdx(level, b_ijk);
			return mLayerHashTablePtrs[level][idx];
		}
		else {
			return HATileInfo<Tile>();//return an empty tile info for non-exist level
		}
	}

	__hostdev__ bool probeTile(const uint32_t layer, const Coord& b_ijk) const {
		HATileInfo<Tile> info = tileInfo(layer, b_ijk);
		return !info.empty();
	}

	__hostdev__ bool findVoxel(const uint32_t level, const Coord& g_ijk, HATileInfo<Tile>& tile_info, Coord& l_ijk) const {
		if (level < 0 || level >= mNumLayers) {
			tile_info = HATileInfo<Tile>();//return an empty tile info for non-exist level
			return false;
		}
		Coord b_ijk;
		decomposeGlobalCoord(g_ijk, b_ijk, l_ijk);
		tile_info = tileInfo(level, b_ijk);
		return !tile_info.empty();
	}

	//here l_ijk is node local coord, it can be in [0,8], and the coordinate calculation is correct
	//each component in off is either -1 or 0, indicating the cell just lower or the cell just upper
	__hostdev__ void findNodeNeighborLeaf(const HATileInfo<Tile>& info, const Coord& l_ijk, const Coord& off, HATileInfo<Tile>& nb_info, Coord& nb_l_ijk) const {
		Coord g_ijk = composeGlobalCoord(info.mTileCoord, l_ijk);
		Coord nb_g_ijk = g_ijk + off;

		int level = info.mLevel;
		findVoxel(level, nb_g_ijk, nb_info, nb_l_ijk);

		//from coarse to fine: NONLEAF, NONLEAF, NONLEAF..., LEAF, [GHOST], EMPTY, EMPTY, ...

		if (!nb_info.empty() && (nb_info.mType & (NONLEAF | LEAF))) {
			//should go down (larger level), or stay the same
			while (!(nb_info.mType & LEAF)) {
				Coord child_offset(0, 0, 0);
				//if off is -1, the child offset is 1, because we are looking for the face just lower
				//if off is 0, the child offset is 0, because we are looking for the face just upper
				child_offset[0] = (off[0] == 0) ? 0 : 1;
				child_offset[1] = (off[1] == 0) ? 0 : 1;
				child_offset[2] = (off[2] == 0) ? 0 : 1;
				nb_g_ijk = childCoord(nb_g_ijk, child_offset);
				findVoxel(++level, nb_g_ijk, nb_info, nb_l_ijk);
			}
		}
		else {
			//should go up (smaller level)
			while (level > 0) {
				nb_g_ijk = parentCoord(nb_g_ijk);
				findVoxel(--level, nb_g_ijk, nb_info, nb_l_ijk);
				if (nb_info.mType & (LEAF)) break;
			}
			//if can't find, just return nb_info (which is empty)
		}
	}

	//here l_ijk is node local coord, it can be in [0,8], and the coordinate calculation is correct
	//each component in off is either -1 or 0, indicating the cell just lower or the cell just upper
	//will try to find the neighbor cell, that:
	//must be LEAF or GHOST
	//is at the finest level
	__hostdev__ void findNodeNeighborLeafOrGhost(const HATileInfo<Tile>& info, const Coord& l_ijk, const Coord& off, HATileInfo<Tile>& nb_info, Coord& nb_l_ijk) const {
		Coord g_ijk = composeGlobalCoord(info.mTileCoord, l_ijk);
		Coord nb_g_ijk = g_ijk + off;

		int level = info.mLevel;
		findVoxel(level, nb_g_ijk, nb_info, nb_l_ijk);

		//from coarse to fine: NONLEAF, NONLEAF, NONLEAF..., LEAF, [GHOST], EMPTY, EMPTY, ...

		if (!nb_info.empty()) {
			//should go down
			HATileInfo<Tile> tmp_info; Coord tmp_l_ijk;
			while (level <= mMaxLevel) {
				Coord child_offset(0, 0, 0);
				//if off is -1, the child offset is 1, because we are looking for the face just lower
				//if off is 0, the child offset is 0, because we are looking for the face just upper
				child_offset[0] = (off[0] == 0) ? 0 : 1;
				child_offset[1] = (off[1] == 0) ? 0 : 1;
				child_offset[2] = (off[2] == 0) ? 0 : 1;
				nb_g_ijk = childCoord(nb_g_ijk, child_offset);
				findVoxel(++level, nb_g_ijk, tmp_info, tmp_l_ijk);
				if (tmp_info.empty()) break;
				//the last recorded nb_info, nb_l_ijk is the correct one
				//it must be either LEAF or GHOST
				nb_info = tmp_info;
				nb_l_ijk = tmp_l_ijk;
			}
		}
		else {
			//should go up
			while (level > 0) {
				nb_g_ijk = parentCoord(nb_g_ijk);
				findVoxel(--level, nb_g_ijk, nb_info, nb_l_ijk);
				//if (nb_info.isLeaf()) break;
				if (nb_info.mType & (LEAF | GHOST)) break;
			}
		}
	}

	__hostdev__ bool findVoxelAndFrac(const Vec& world_pos, uint8_t tile_types, HATileInfo<Tile>& info, Coord& l_ijk, Vec& frac) const {
		for (int i = mMaxLevel; i >= 0; i--) {
			Coord g_ijk;
			worldToVoxelAndFraction(i, world_pos, g_ijk, frac);
			findVoxel(i, g_ijk, info, l_ijk);
			if (info.mType & tile_types) {
				//printf("world_pos: %f %f %f\n", world_pos[0], world_pos[1], world_pos[2]);
				//printf("g_ijk: %d %d %d\n", g_ijk[0], g_ijk[1], g_ijk[2]);
				//printf("frac: %f %f %f\n", frac[0], frac[1], frac[2]);
				//printf("level %d voxel size %f\n", i, voxelSize(i));
				return true;
			}
		}
		info = HATileInfo<Tile>();//empty
		return false;
	}

	__hostdev__ bool findLeafVoxelAndFrac(const Vec& world_pos, HATileInfo<Tile>& info, Coord& l_ijk, Vec& frac) const {
		return findVoxelAndFrac(world_pos, LEAF, info, l_ijk, frac);
		//for (int i = mMaxLevel; i >= 0; i--) {
		//	Coord g_ijk;
		//	worldToVoxelAndFraction(i, world_pos, g_ijk, frac);
		//	findVoxel(i, g_ijk, info, l_ijk);
		//	if (info.isLeaf()) {
		//		//printf("world_pos: %f %f %f\n", world_pos[0], world_pos[1], world_pos[2]);
		//		//printf("g_ijk: %d %d %d\n", g_ijk[0], g_ijk[1], g_ijk[2]);
		//		//printf("frac: %f %f %f\n", frac[0], frac[1], frac[2]);
		//		//printf("level %d voxel size %f\n", i, voxelSize(i));
		//		return true;
		//	}
		//}
		//info = HATileInfo<Tile>();//empty
		//return false;
	}

	__hostdev__ bool findPlusFaceIntpVoxel(const Vec& world_pos, const int axis, const HATileInfo<Tile>& r_info, const Coord& r_ijk, HATileInfo<Tile>& n_info, Coord& n_ijk, Vec& frac)const {
		//the cell we want may differ 0 or 1 level from the cell we are in
		auto test_pos = world_pos;
		auto cell_ctr = cellCenter(r_info, r_ijk);
		test_pos[axis] = cell_ctr[axis] + (0.5 + 0.25) * voxelSize(r_info);

		//printf("world pos: %f %f %f\n", world_pos[0], world_pos[1], world_pos[2]);
		//printf("test_pos: %f %f %f\n", test_pos[0], test_pos[1], test_pos[2]);

		return findLeafVoxelAndFrac(test_pos, n_info, n_ijk, frac);
	}

	//will only itearte over non-empty tiles
	template<class FuncBC>
	__hostdev__ void iterateChildVoxels(const HATileInfo<Tile>& info, const Coord& l_ijk, FuncBC f)const {
		auto g_ijk = composeGlobalCoord(info.mTileCoord, l_ijk);
		for (int s = 0; s < NUMCHILDREN; s++) {
			Coord c_g_ijk = childCoord(g_ijk, s);
			HATileInfo<Tile> c_info; Coord c_l_ijk;
			findVoxel(info.mLevel + 1, c_g_ijk, c_info, c_l_ijk);
			if (!c_info.empty()) f(c_info, c_l_ijk);
		}
	}


	template<class FuncBCII>
	__hostdev__ void iterateSameLevelNeighborVoxels(const HATileInfo<Tile>& info, const Coord& l_ijk, FuncBCII f)const {
		auto& tile = info.tile();
		auto g_ijk = composeGlobalCoord(info.mTileCoord, l_ijk);
		for (int axis : {0, 1, 2}) {
			for (auto& sgn : { -1,1 }) {
				auto nb_g_ijk = g_ijk;
				nb_g_ijk[axis] += sgn;
				HATileInfo<Tile> nb_info; Coord nb_l_ijk;
				findVoxel(info.mLevel, nb_g_ijk, nb_info, nb_l_ijk);
				f(nb_info, nb_l_ijk, axis, sgn);
			}
		}
	}

	template<class FuncBCII>
	__hostdev__ void iterateNeighborLeafCells(const HATileInfo<Tile>& info, const Coord& l_ijk, FuncBCII f)const {
		auto g_ijk = localToGlobalCoord(info, l_ijk);
		for (int axi : {0, 1, 2}) {
			for (int sgn : {-1, 1}) {
				int axj = Base::rotateAxis(axi, 1);
				int axk = Base::rotateAxis(axi, 2);
				auto ng_ijk = g_ijk;
				ng_ijk[axi] += sgn;
				HATileInfo<Tile> ninfo; Coord nl_ijk;
				findVoxel(info.mLevel, ng_ijk, ninfo, nl_ijk);
				if (ninfo.isGhost()) {
					//the neighbor exists, and is coarser
					auto np_ijk = parentCoord(ng_ijk);
					findVoxel(info.mLevel - 1, np_ijk, ninfo, nl_ijk);
					//it's either empty or non-empty
					if (!ninfo.empty()) {
						f(ninfo, nl_ijk, axj, sgn);
					}
				}
				else if (!ninfo.empty()) {
					if (ninfo.isLeaf()) {
						f(ninfo, nl_ijk, axi, sgn);
					}
					else {
						for (int offj : {0, 1}) {
							for (int offk : {0, 1}) {
								Coord off(0, 0, 0);
								off[axi] = (sgn == -1) ? 1 : 0;
								off[axj] = offj;
								off[axk] = offk;
								auto nc_ijk = Base::childCoord(ng_ijk, off);
								findVoxel(info.mLevel + 1, nc_ijk, ninfo, nl_ijk);
								if (!ninfo.empty() && ninfo.isLeaf()) {
									f(ninfo, nl_ijk, axj, sgn);
								}
							}
						}
					}
				}
			}
		}
	}

	template<class FuncBCII>
	__hostdev__ void iterateNeighborLeafFaces(const int axis, const HATileInfo<Tile>& info, const Coord& l_ijk, FuncBCII f)const {
		auto g_ijk = localToGlobalCoord(info, l_ijk);

		for (int xj : {1, 2}) {
			int xk = (xj == 1) ? 2 : 1;

			int axj = Base::rotateAxis(axis, xj);
			int axk = Base::rotateAxis(axis, xk);

			for (auto& sgn : { -1,1 }) {
				auto ng_ijk = g_ijk;
				ng_ijk[axj] += sgn;
				HATileInfo<Tile> ninfo; Coord nl_ijk;
				findVoxel(info.mLevel, ng_ijk, ninfo, nl_ijk);
				if (ninfo.isGhost()) {
					//the neighbor exists, and is coarser
					auto np_ijk = parentCoord(ng_ijk);
					findVoxel(info.mLevel - 1, np_ijk, ninfo, nl_ijk);
					//it's either empty or non-empty
					if (!ninfo.empty()) {
						f(ninfo, nl_ijk, axj, sgn);
					}
				}
				else if(!ninfo.empty()) {
					if (ninfo.isLeaf()) {
						f(ninfo, nl_ijk, axj, sgn);
					}
					else {
						//maybe finer
						for (auto offk : { 0,1 }) {
							Coord off(0, 0, 0);
							off[axj] = (sgn == -1) ? 1 : 0;
							off[axk] = offk;
							auto nc_ijk = Base::childCoord(ng_ijk, off);
							findVoxel(info.mLevel + 1, nc_ijk, ninfo, nl_ijk);
							if (!ninfo.empty()) {
								f(ninfo, nl_ijk, axj, sgn);
							}
						}
					}
				}
				//if it's empty, then it's non-exist
			}
		}
	}
};