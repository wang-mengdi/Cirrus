#pragma once

#include "Common.h"


enum CellType { DIRICHLET = 0b001, INTERIOR = 0b010, NEUMANN = 0b100, SURFACE = 0b1000 };
enum AdaptiveStat { REFINE_FLAG = 0b001, DELETE_FLAG = 0b010, COARSEN_FLAG = 0b100, ACTIVE_FLAG = 0b1000 };

template<class Type>
class PoissonTile {
public:
    static constexpr int x_channel = 0;
    static constexpr int b_channel = 1;
    static constexpr int r_channel = 1;
    static constexpr int p_channel = 2;
    static constexpr int Ap_channel = 3;
    static constexpr int z_channel = 4;
    static constexpr int tmp_channel = 5;
    static constexpr int u_channel = 6;
    static constexpr int v_channel = 7;
    static constexpr int w_channel = 8;
    static constexpr int vor_channel = 9;//vorticity
	static constexpr int dye_channel = 10;//dye density
    static constexpr int num_channels = 11;
    //static constexpr uint32_t c0_channel = 11;
    //static constexpr uint32_t c1_channel = 12;
    //static constexpr uint32_t c2_channel = 13;
    //static constexpr uint32_t c3_channel = 14;

    using T = typename Type;
    using CoordType = typename nanovdb::Coord;
    using Coord = typename CoordType;
    using VecType = typename nanovdb::Vec3<T>;
    using CoordAcc = typename HACoordAccessor<PoissonTile<T>>;
    constexpr static uint32_t LOG2DIM = 3;
    constexpr static uint32_t DIM = 1u << LOG2DIM; // this tile stores (DIM*DIM*DIM) voxels (default 8^3=512)
    static constexpr uint32_t SIZE = 1u << (3 * LOG2DIM); // total number of voxels
    static constexpr uint32_t NODESIZE = (DIM + 1) * (DIM + 1) * (DIM + 1); // total number of nodes
    static constexpr uint32_t CHNLSIZE = 736; // size of a channel, >= NODESIZE

    static constexpr T BACKGROUND_VALUE = 0;


    //somehow we can't move this forward, presumably alignment issues

    //alignas(16) int mStatus;//record refine and coarsen status
    //alignas(16) int mIsInterestArea = false, mIsLockedRefine = false;
    //alignas(16) int mSerialIdx;


    T mData[num_channels][CHNLSIZE];//9^3 =729
    HATileInfo<PoissonTile<T>> mNeighbors[6];//x-,y-,z-,x+,y+,z+
    uint8_t mCellType[SIZE];
    bool mIsInterestArea = false, mIsLockedRefine = false;
    int mStatus;//record refine and coarsen status
    int mSerialIdx;

    

    


    
 //   int mSerialIdx;
 //   bool mIsInterestArea = false, mIsLockedRefine = false;
 //   uint8_t mStatus;//record refine and coarsen status
 //   HATileInfo<PoissonTile<T>> mNeighbors[6];//x-,y-,z-,x+,y+,z+
 //   uint8_t mCellType[SIZE];
	//T mData[num_channels][CHNLSIZE];//9^3 =729

    
	

    PoissonTile() {
        mStatus = 0;
    }
    __hostdev__ void setMask(const uint8_t mask, bool value) {
        if (value) mStatus |= mask;
        else mStatus &= (~mask);
    }

    __hostdev__ T& operator ()(const uint32_t channel, const Coord& l_ijk) {
        return mData[channel][CoordAcc::localCoordToOffset(l_ijk)];
    }
    __hostdev__ T& operator ()(const int channel, const int idx) {
		return mData[channel][idx];

    }
	__hostdev__ T value(const uint32_t channel, const Coord& l_ijk) const {
		return mData[channel][CoordAcc::localCoordToOffset(l_ijk)];
	}
    __hostdev__ T interiorValue(const uint32_t channel, const Coord& l_ijk) const {
        if (isInterior(l_ijk)) return mData[channel][CoordAcc::localCoordToOffset(l_ijk)];
        else return BACKGROUND_VALUE;
    }

    __hostdev__ T& node(const int channel, const Coord& l_ijk) { return mData[channel][l_ijk[0] * (DIM + 1) * (DIM + 1) + l_ijk[1] * (DIM + 1) + l_ijk[2]]; }
    __hostdev__ const T& node(const int channel, const Coord& l_ijk) const { return mData[channel][l_ijk[0] * (DIM + 1) * (DIM + 1) + l_ijk[1] * (DIM + 1) + l_ijk[2]]; }

    __hostdev__ uint8_t& type(const Coord& l_ijk) { return mCellType[CoordAcc::localCoordToOffset(l_ijk)]; }
	__hostdev__ uint8_t& type(const int idx) { return mCellType[idx]; }
    __hostdev__ bool isInterior(const Coord& l_ijk) const { return mCellType[CoordAcc::localCoordToOffset(l_ijk)] & CellType::INTERIOR; }
	__hostdev__ bool isDirichlet(const Coord& l_ijk) const { return mCellType[CoordAcc::localCoordToOffset(l_ijk)] & CellType::DIRICHLET; }
	__hostdev__ bool isNeumann(const Coord& l_ijk) const { return mCellType[CoordAcc::localCoordToOffset(l_ijk)] & CellType::NEUMANN; }

    __hostdev__ T cellInterp(const uint32_t cell_channel, const uint32_t node_channel, const Coord& l_ijk, const VecType& frac) const {
        T node_intp = 0, node_avg = 0;
        for (int offi : {0, 1}) {
            for (int offj : {0, 1}) {
                for (int offk : {0, 1}) {
                    T weight = ((offi == 0) ? (1 - frac[0]) : frac[0])
                        * ((offj == 0) ? (1 - frac[1]) : frac[1])
                        * ((offk == 0) ? (1 - frac[2]) : frac[2]);
                    auto node_l_ijk = l_ijk + Coord(offi, offj, offk);
                    T val = node(node_channel, node_l_ijk);

                    //printf("node[%d,%d,%d]=%f\n", node_l_ijk[0], node_l_ijk[1], node_l_ijk[2], val);

                    node_intp += weight * val;
                    node_avg += val;
                }
            }
        }

        T frac_min = 1.0;
        for (int i : {0, 1, 2}) {
            if (frac[i] < frac_min) frac_min = frac[i];
            if (1 - frac[i] < frac_min) frac_min = 1 - frac[i];
        }
        T delta = value(cell_channel, l_ijk) - node_avg / 8.0;
        return node_intp + 2 * delta * frac_min;
    }

    __hostdev__ T faceInterp(const uint32_t u_channel, const uint32_t node_u_channel, const int axis, const Coord& l_ijk, const VecType& frac) const {
        T node_intp = 0, node_sum = 0;
        int axi = axis;
		int axj = CoordAcc::rotateAxis(axis, 1);
		int axk = CoordAcc::rotateAxis(axis, 2);

        //printf("interpolate axis %d l_ijk %d,%d,%d frac %f,%f,%f\n", axis, l_ijk[0], l_ijk[1], l_ijk[2], frac[0], frac[1], frac[2]);

        for (int offj : {0, 1}) {
            for (int offk : {0, 1}) {
                Coord off_ijk = CoordAcc::rotateCoord(axis, Coord(0, offj, offk));

                T wj = ((offj == 0) ? (1 - frac[axj]) : frac[axj]);
				T wk = ((offk == 0) ? (1 - frac[axk]) : frac[axk]);
				T weight = wj * wk;
                T val = node(node_u_channel + axis, l_ijk + off_ijk);
				node_intp += weight * val;
                node_sum += val;

				//printf("node[%d,%d,%d]=%f\n", l_ijk[0] + off_ijk[0], l_ijk[1] + off_ijk[1], l_ijk[2] + off_ijk[2], val);
                //printf("hello\n");
            }
        }

		T frac_min = 1.0;
        for (int ax : {axj, axk}) {
			if (frac[ax] < frac_min) frac_min = frac[ax];
            if (1 - frac[ax] < frac_min) frac_min = 1 - frac[ax];
        }

            
        T delta = value(u_channel + axis, l_ijk) - node_sum / 4.0;
        //printf("face center value: %f node_avg: %f delta: %f\n", interiorValue(u_channel + axis, l_ijk), node_sum / 4.0, delta);
		return node_intp + 2 * delta * frac_min;
	}

    __hostdev__ PoissonTile childTile(const Coord offset)const {
		PoissonTile child;
		child.mIsInterestArea = mIsInterestArea;
		child.mStatus = mStatus;

        for (int i = 0; i < DIM; i++) {
            for (int j = 0; j < DIM; j++) {
                for (int k = 0; k < DIM; k++) {
                    Coord l_ijk(i, j, k);
                    Coord g_ijk = Coord(offset[0] * DIM, offset[1] * DIM, offset[2] * DIM) + l_ijk;

                    Coord pl_ijk = CoordAcc::parentCoord(g_ijk);
                    for (int c = 0; c < num_channels; c++) {
                        child(c, l_ijk) = value(c, pl_ijk);
                    }
					child.type(l_ijk) = mCellType[CoordAcc::localCoordToOffset(pl_ijk)];
                }
            }
        }

        return child;
    }
};