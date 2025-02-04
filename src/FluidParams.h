#pragma once

#include "GMGSolver.h"
#include "SDFGrid.h"


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//TVORTEX: tornado-like vortex, reference: Physically-based Simulation of Tornadoes
enum TestCase { TVORTEX = 0, FORCE_VORT, FREE_VORT, DUAL_VORT, LEAP_FROG, KARMAN, SMOKESPHERE, NASA, PROP, WP3D, F1CAR, LIZARD, FISH, BAT, FLAMINGO, BLUEANGEL };

class PropellerTransform {
public:
	using Vec = nanovdb::Vec3<float>;

	__hostdev__ PropellerTransform() {}
	__hostdev__ PropellerTransform(
		const Vec& model_center,
		const Vec& world_center,
		float attack_angle,
		float scale_factor,
		float omega,
		float theta0 = 0.0f
	);

	__hostdev__ Vec transformModelToWorld(const Vec& point, float time) const;
	__hostdev__ Vec transformWorldToModel(const Vec& point, float time) const;
	__hostdev__ Vec transformModelVelocityToWorld(const Vec& velocity, float time) const;
	__hostdev__ Vec calculateModelVelocityFromRotation(const Vec& point, float time) const;
	__hostdev__ Vec calculateWorldVelocity(const Vec& point, float time) const;

public:
	Vec model_center_;
	Vec world_center_;
	float attack_angle_;    // Rotation angle around y-z plane
	float scale_factor_;    // Scale factor, world_dim / model_dim
	float omega_;           // Angular velocity
	float theta0_;          // Initial rotation angle at t = 0

	__hostdev__ Vec rotateXY(const Vec& point, float theta) const;
	__hostdev__ Vec rotateYZ(const Vec& point, float alpha) const;
};

class FluidParams {
public:
	//parameters read from json
	TestCase mTestCase;
	int mFlowMapStride;
	int mCoarseLevel;
	int mFineLevel;
	T mRefineThreshold;
	T mParticleLife;
	nanovdb::Vec3R mGravity;
	MaskGridAccessor mMaskGridAccessor;
	SDFGridAccessor mSDFGridAccessor;

	SDFGridAccessor mSDFVelocityAccessors[3];

	//for karman
	T karman_source = 1.0;

	//Vec smokesphere_source = Vec(1.0, 0., 0.);
	Vec smokesphere_source = Vec(0., 0., 0.4);
	Vec smokesphere_center = Vec(0.5, 0.5, 0.3);
	//Vec smokesphere_center = Vec(0.5, 0.5, 0.7);
	T smokesphere_radius = 0.05;

	T nasa_source = 1.0;

	T prop_source = 1.0;
	T f1_source = 1.0;
	T lizard_source = +1.0;
	T fish_source = 1.0;
	T bat_source = 1.0;
	T flamingo_source = 1.0;

	//parameters set by itself
	bool mIsPureNeumann = false;

	FluidParams() {}
	FluidParams(json& j, std::shared_ptr<MaskGrid> mask_grid_ptr, std::shared_ptr<SDFGrid> sdf_float_grid_ptr) {

		std::string test = Json::Value<std::string>(j, "test", "tvortex");
		if (test == "tvortex") mTestCase = TVORTEX;
		else if (test == "force_vort") mTestCase = FORCE_VORT;
		else if (test == "free_vort") mTestCase = FREE_VORT;
		else if (test == "dual_vort") mTestCase = DUAL_VORT;
		else if (test == "leap_frog") mTestCase = LEAP_FROG;
		else if (test == "karman") mTestCase = KARMAN;
		else if (test == "smokesphere") mTestCase = SMOKESPHERE;
		else if (test == "nasa") mTestCase = NASA;
		else if (test == "prop") mTestCase = PROP;
		else if (test == "wp3d") mTestCase = WP3D;
		else if (test == "f1car") mTestCase = F1CAR;
		else if (test == "lizard") mTestCase = LIZARD;
		else if (test == "fish") mTestCase = FISH;
		else if (test == "bat") mTestCase = BAT;
		else if (test == "flamingo") mTestCase = FLAMINGO;
		else if (test == "blueangel") mTestCase = BLUEANGEL;
		else Assert(false, "invalid test {}", test);

		if (mTestCase == LEAP_FROG || mTestCase == KARMAN || mTestCase == SMOKESPHERE || mTestCase == NASA || mTestCase == PROP || mTestCase == WP3D || mTestCase == F1CAR || mTestCase == LIZARD
			|| mTestCase == FISH || mTestCase == BAT || mTestCase == FLAMINGO || mTestCase == BLUEANGEL
			) {
			mIsPureNeumann = true;
		}

		mFlowMapStride = Json::Value<int>(j, "flowmap_stride", 5);
		mCoarseLevel = Json::Value<int>(j, "coarse_level", 0);
		mFineLevel = Json::Value<int>(j, "fine_level", 6);
		mRefineThreshold = Json::Value<T>(j, "refine_threshold", 200);
		mGravity = Vec(0, 0, 0);
		mGravity[2] = Json::Value<double>(j, "gravity", -9.8);
		mParticleLife = Json::Value<T>(j, "particle_life", FLT_MAX);

		if (mask_grid_ptr) {
			mMaskGridAccessor = mask_grid_ptr->GetDeviceAccessor();
		}
		if (sdf_float_grid_ptr) {
			mSDFGridAccessor = sdf_float_grid_ptr->GetDeviceAccessor();
		}
	}

	__device__ static bool queryBoundaryDirection(const HATileAccessor<Tile>& acc, const int level, const Coord& g_ijk, int& boundary_axis, int& boundary_off) {
		boundary_axis = -1;
		boundary_off = 0;

		for (int axis : {0, 1, 2}) {
			for (int off : {-1, 1, 2}) {
				auto ng_ijk = g_ijk; ng_ijk[axis] += off;
				HATileInfo<Tile> ninfo; Coord nl_ijk;
				acc.findVoxel(level, ng_ijk, ninfo, nl_ijk);
				if (ninfo.empty()) {
					boundary_axis = axis;
					boundary_off = off;
					return true;
				}

			}
		}
		return false;
	}

	__device__ static bool queryBoundaryDirection(const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk, int& boundary_axis, int& boundary_off) {
		auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
		return queryBoundaryDirection(acc, info.mLevel, g_ijk, boundary_axis, boundary_off);

		//boundary_axis = -1;
		//boundary_off = 0;

		//auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
		//for (int axis : {0, 1, 2}) {
		//	for (int off : {-1, 1, 2}) {
		//		auto ng_ijk = g_ijk; ng_ijk[axis] += off;
		//		HATileInfo<Tile> ninfo; Coord nl_ijk;
		//		acc.findVoxel(info.mLevel, ng_ijk, ninfo, nl_ijk);
		//		if (ninfo.empty()) {
		//			boundary_axis = axis;
		//			boundary_off = off;
		//			return true;
		//		}

		//	}
		//}
		//return false;

		//acc.iterateSameLevelNeighborVoxels(info, l_ijk, [&]__device__(const HATileInfo<Tile>&n_info, const Coord & nl_ijk, const int axis, const int sgn) {
		//	if (n_info.empty()) {
		//		boundary_axis = axis;
		//		boundary_sgn = sgn;
		//	}
		//});
		//return boundary_axis != -1;
	}

	__device__ static bool queryBoundaryDirection1(const HATileAccessor<Tile>& acc, const int level, const Coord& g_ijk, int& boundary_axis, int& boundary_off) {
		boundary_axis = -1;
		boundary_off = 0;

		for (int axis : {0, 1, 2}) {
			for (int off : {-1, 1}) {
				auto ng_ijk = g_ijk; ng_ijk[axis] += off;
				HATileInfo<Tile> ninfo; Coord nl_ijk;
				acc.findVoxel(level, ng_ijk, ninfo, nl_ijk);
				if (ninfo.empty()) {
					boundary_axis = axis;
					boundary_off = off;
					return true;
				}

			}
		}
		return false;
	}

	__device__ static bool queryEffectiveBoundaryDirection1(const HATileAccessor<Tile>& acc, int chk_level, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk, int& boundary_axis, int& boundary_off) {
		chk_level = min(chk_level, info.mLevel);
		int level_diff = info.mLevel - chk_level;
		
		auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
		

		g_ijk = Coord(g_ijk[0] >> level_diff, g_ijk[1] >> level_diff, g_ijk[2] >> level_diff);

		//printf("level %d chk level %d diff %d g_ijk %d %d %d\n", info.mLevel, chk_level, level_diff, g_ijk[0], g_ijk[1], g_ijk[2]);
		return queryBoundaryDirection1(acc, chk_level, g_ijk, boundary_axis, boundary_off);
	}

	__hostdev__ static Vec NASAWorldToModelPos(const Vec& world_pos, const double time) {
		return world_pos;
		//Vec model_center(0.5, 0.5, 0.5);
		//Vec new_center(0.5, 0.5, 0.5);
		//T new_dim = 0.5;
		////in global space, model scales from 1 to new_dim and center moves from model_center to new_center
		////therefore we have to apply the inverse transform to get model pos
		//Vec model_pos = (world_pos - new_center) / new_dim + model_center;
		//return model_pos;
	}
	__hostdev__ static Vec F1CarWorldToModelPos(const Vec& world_pos, const double time) {
		return world_pos + Vec(0, +0.340, -0.25);
	}
	//__hostdev__ static Vec FishWorldToModelPos(const Vec& world_pos, const double time) {
	//	return world_pos;
	//}
	__hostdev__ T FishQuerySDF(const Vec& world_pos, const double time) const {
		Vec model_ctr = Vec(0.5, 0.5, 0.5);
		Vec world_ctr = Vec(1.5, 0.5, 0.5);
		//x: 2s from 3.5 to 0.5
		constexpr T rate = 1.5;
		world_ctr[0] = 3.5 - rate * time;
		//z: 2s->2pi
		constexpr T A = 0.1 * 0, omega = 2 * M_PI / 2.0;
		world_ctr[2] = 0.5 + A * sin(time * omega);


		Vec vel_dir = Vec(0, 0, 0);
		vel_dir[0] = -rate;
		vel_dir[2] = A * omega * cos(time * omega);
		Vec world_x = -vel_dir.normalize();
		Vec world_y = Vec(0, 1, 0);
		Vec world_z = world_x.cross(world_y);
		
		constexpr T model_dim = 0.794;
		//constexpr T world_dim = 0.6;
		constexpr T world_dim = model_dim;

		//transform world pos to model pos:
		//1. translate world_ctr to origin
		//2. rotate world_x to (1,0,0), world_y to (0,1,0), world_z to (0,0,1)
		//3. scale by model_dim / world_dim
		//4. translate model_ctr to origin
		Vec model_pos = (world_pos - world_ctr);
		//model_pos = Vec(world_x.dot(model_pos), world_y.dot(model_pos), world_z.dot(model_pos));
		model_pos = model_pos / world_dim * model_dim;
		model_pos = model_pos + model_ctr;


		

		//Vec model_pos = (world_pos - world_ctr) + model_ctr;
		//Vec model_pos = (world_pos - world_ctr);
		

		//T model_dim = 0.9;
		//T world_dim = 0.6;
		auto model_phi = mSDFGridAccessor.linearInterpolate(model_pos[0], model_pos[1], model_pos[2]);
		return model_phi;
		//return model_phi / model_dim * world_dim;
	}

	
	__hostdev__ Vec BATWorldToModelPos(const Vec& world_pos, const double time) const {
		return world_pos;
	}
	__hostdev__ Vec BATQuerySolidVelocity(const Vec& world_pos, const double time)const {
		return Vec(0, 0, 0);
		auto model_pos = BATWorldToModelPos(world_pos, time);
		Vec vel;
		for (int axis : {0, 1, 2}) {
			vel[axis] = mSDFVelocityAccessors[axis].linearInterpolate(model_pos[0], model_pos[1], model_pos[2]);
		}
		return vel;
	}
	__hostdev__ T BATQuerySDF(const Vec& world_pos, const double time) const {
		auto model_pos = BATWorldToModelPos(world_pos, time);
		auto model_phi = mSDFGridAccessor.linearInterpolate(model_pos[0], model_pos[1], model_pos[2]);
		return model_phi;
	}

	__hostdev__ Vec FlamingoWorldToModelPos(const Vec& world_pos, const double time) const {
		return world_pos;
	}
	__hostdev__ T FlamingoQuerySDF(const Vec& world_pos, const double time) const {
		auto model_pos = FlamingoWorldToModelPos(world_pos, time);
		auto model_phi = mSDFGridAccessor.linearInterpolate(model_pos[0], model_pos[1], model_pos[2]);
		return model_phi;
	}

	//__hostdev__ Vec BlueAngelWorldToModelPos(const Vec& world_pos, const int idx, const double time) const {
	//	const static Vec diamond0(-0.25, -0.25, -0.25);
	//	const static Vec diamond1(0.25, 0.25, 0.25);
	//	const static T r = 0.25;
	//	const static Vec rel_positions[8] = {
	//		diamond0 + Vec(0,0,-r),
	//		diamond0 + Vec(0,0,r),
	//		diamond0 + Vec(r,0,0),
	//		diamond0 + Vec(-r,0,0),
	//		diamond1 + Vec(0,0,-r),
	//		diamond1 + Vec(0,0,r),
	//		diamond1 + Vec(r,0,0),
	//		diamond1 + Vec(-r,0,0),
	//	};

	//	Vec formation_ctr = Vec(0.5, 0.5, 3.5 - time * 1.5);
	//	Vec plane_world_pos = rel_positions[idx] + formation_ctr;
	//	const static Vec plane_model_pos = Vec(0.5, 0.5, 0.5);

	//	return world_pos - plane_world_pos + plane_model_pos;
	//}

	//__hostdev__ T BlueAngelQuerySDF(const Vec& world_pos, const double time)const {
	//	auto model_pos = world_pos;
	//	T phi = FLT_MAX;
	//	for (int i = 0; i < 8; i++) {
	//		auto model_pos = BlueAngelWorldToModelPos(world_pos, i, time);
	//		auto model_phi = mSDFGridAccessor.linearInterpolate(model_pos[0], model_pos[1], model_pos[2]);
	//		phi = min(phi, model_phi);
	//	}
	//	return phi;
	//}

	//__hostdev__ static Vec worldToModelPosPropeller1(const Vec& world_pos, const int prop_idx, T omega, const double time) {
	//	T world_dim = 0.105, model_dim = 0.9;
	//	T omega0 = 0;
	//	Vec world_center;
	//	if (prop_idx == 0) {
	//		world_center = Vec(0.5, 0.5, 0.5);
	//		omega0 = 0;
	//	}

	//	Vec pos = (world_pos - world_center) / world_dim * model_dim;
	//	T theta = -(time * omega + omega0);
	//	T c = cos(theta);
	//	T s = sin(theta);
	//	Vec model0(c * pos[0] - s * pos[1], s * pos[0] + c * pos[1], pos[2]);
	//}

	//__hostdev__ static Vec PropWorldToModelPos(const Vec& world_pos, const double time) {
	//	T omega = 10.0;

	//	Vec model_center(0.5, 0.5, 0.5);
	//	Vec world_center(0.5, 0.5, 0.5);
	//	T world_dim = 0.105, model_dim = 0.9;
	//	//in global space, model scales from 1 to new_dim and center moves from model_center to new_center
	//	//therefore we have to apply the inverse transform to get model pos
	//	Vec pos = (world_pos - world_center) / world_dim * model_dim;

	//	//rotate -time * omeag in x-y plane
	//	T theta = -time * omega;
	//	T c = cos(theta);
	//	T s = sin(theta);
	//	Vec model0(c * pos[0] - s * pos[1], s * pos[0] + c * pos[1], pos[2]);

	//	Vec model_pos = model0 + model_center;
	//	return model_pos;
	//}

	__hostdev__ static void createPropeller1(PropellerTransform& p0) {
		Vec pc = (Vec(0.728916, 0.547824, 0.328438) + Vec(0.722682, 0.531279, 0.324005)) * 0.5;

		T rad_of_15 = 15.0 * M_PI / 180.0;
		//p0 = PropellerTransform(Vec(0.5, 0.5, 0.5), Vec(0.5, 0.5, 0.5), rad_of_15, 0.105 / 0.9, 10);
		p0 = PropellerTransform(Vec(0.5, 0.5, 0.5), pc, rad_of_15, 0.105 / 0.9, 10);
	}
	__hostdev__ static void createPropeller4(PropellerTransform* props) {
		T rad_of_15 = 15.0 * M_PI / 180.0;
		//T world_over_model_dim = 0.105 / 0.9;
		T world_over_model_dim = 0.105 / 0.9 * 0.8;

		//  prop0:
		//  0.734363, 0.536540, 0.325415
		//	0.717235, 0.542562, 0.327028

		//	prop1 :
		//	0.615386, 0.539832, 0.319593
		//	0.623661, 0.524144, 0.315389

		//	prop2 :
		//	0.372833, 0.536783, 0.318776
		//	0.388120, 0.527194, 0.316206

		//	prop3 :
		//	0.274995, 0.548321, 0.328571
		//	0.273407, 0.530781, 0.323871

		

		Vec pc0 = (Vec(0.734363, 0.536540, 0.325415) + Vec(0.717235, 0.542562, 0.327028)) * 0.5;
		Vec pc1 = (Vec(0.615386, 0.539832, 0.319593) + Vec(0.623661, 0.524144, 0.315389)) * 0.5;
		Vec pc2 = (Vec(0.372833, 0.536783, 0.318776) + Vec(0.388120, 0.527194, 0.316206)) * 0.5;
		Vec pc3 = (Vec(0.274995, 0.548321, 0.328571) + Vec(0.273407, 0.530781, 0.323871)) * 0.5;

		//{
		//	Vec offset = Vec(0, sin(rad_of_15), -cos(rad_of_15)) * 0.01;
		//	pc0 += offset;
		//	pc1 += offset;
		//	pc2 += offset;
		//	pc3 += offset;

		//}


		//auto mirror_over_x_half = []__hostdev__(const Vec& p) {return Vec(1 - p[0], p[1], p[2]); };
		//pc0 = mirror_over_x_half(pc0);
		//pc1 = mirror_over_x_half(pc1);
		//pc2 = mirror_over_x_half(pc1);
		//pc3 = mirror_over_x_half(pc0);

		props[0] = PropellerTransform(Vec(0.5, 0.5, 0.5), pc0, rad_of_15, world_over_model_dim, -10, 0.0);
		props[1] = PropellerTransform(Vec(0.5, 0.5, 0.5), pc1, rad_of_15, world_over_model_dim, -10, M_PI / 4);
	
		props[2] = PropellerTransform(Vec(0.5, 0.5, 0.5), pc2, rad_of_15, world_over_model_dim, -10, M_PI / 4);
		props[3] = PropellerTransform(Vec(0.5, 0.5, 0.5), pc3, rad_of_15, world_over_model_dim, -10, 0.0);
	}

	__hostdev__ T PropQuerySDFWorld(const PropellerTransform& P, const Vec& world_pos, const double time) const {
		auto model_pos = P.transformWorldToModel(world_pos, time);
		auto model_phi = mSDFGridAccessor.linearInterpolate(model_pos[0], model_pos[1], model_pos[2]);
		T world_phi = model_phi * P.scale_factor_;
		return world_phi;
		

		//T world_dim = 0.105, model_dim = 0.9;
		//auto model_pos = PropWorldToModelPos(world_pos, time);
		//auto phi = mSDFGridAccessor.linearInterpolate(model_pos[0], model_pos[1], model_pos[2]);
		////return phi;
		//return phi / model_dim * world_dim;
	}

	__hostdev__ Vec propellerBoundaryVelocity1(PropellerTransform & p0, const Vec& pos, const double time)const {
		auto phi = PropQuerySDFWorld(p0, pos, time);
		if (phi <= mSDFGridAccessor.gen_isovalue) {
			return p0.calculateWorldVelocity(pos, time);
		}
		else {
			return Vec(0, 0, prop_source);
		}
	}

	__hostdev__ bool propellerBoundaryVelocity4(PropellerTransform* prs, const Vec& pos, const double time, Vec& vel)const {
		for (int i = 0; i < 4; ++i) {
			auto phi = PropQuerySDFWorld(prs[i], pos, time);
			if (phi <= mSDFGridAccessor.gen_isovalue) {
				vel = prs[i].calculateWorldVelocity(pos, time);
				return true;
			}
		}
		return false;
	}

	__hostdev__ int initialLevelTarget(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info) const {
		if (mTestCase == PROP) {
			auto world_box = acc.tileBBox(info);
			auto world_min = world_box.min();
			auto world_max = world_box.max();

			//PropellerTransform p0; createPropeller1(p0);
			PropellerTransform prs[4]; createPropeller4(prs);

			//auto model_min = PropWorldToModelPos(world_min, 0);
			//auto model_max = PropWorldToModelPos(world_max, 0);

			//for (auto& p0 : {prs[0],prs[1]})
			for(auto& p0: prs)
			{
				auto model_min = p0.transformWorldToModel(world_min, 0);
				auto model_max = p0.transformWorldToModel(world_max, 0);
				nanovdb::BBox<Vec> model_box(model_min, model_max);

				//{
				//	Vec prop_model_ctr(0.5, 0.5, 0.5);
				//	if (model_box.isInside(prop_model_ctr)) {
				//		return mFineLevel;
				//	}
				//}

				{
					//model pos bounding box for propeller
					Vec mb_min(0., 0., 0.47003019);
					Vec mb_max(1., 1., 0.52996981);
					nanovdb::BBox<Vec> mb_box(mb_min, mb_max);

					auto intersect_base = model_box.intersect(mb_box);
					nanovdb::BBox<Vec> intersect_bbox(intersect_base.min(), intersect_base.max());

					if (!intersect_bbox.empty()) return mFineLevel;

					//if (model_box.intersect(mb_box)) return 0;
				}
			}

			return mCoarseLevel;
		}
		if (mTestCase == F1CAR) {
			auto world_box = acc.tileBBox(info);
			auto world_min = world_box.min();
			auto world_max = world_box.max();

			auto model_min = F1CarWorldToModelPos(world_min, 0);
			auto model_max = F1CarWorldToModelPos(world_max, 0);
			nanovdb::BBox<Vec> model_box(model_min, model_max);

			{
				Vec mb_min(0.33300686, 0.40301267, 0.05);
				Vec mb_max(0.66699314, 0.59698733, 0.95);
				nanovdb::BBox<Vec> mb_box(mb_min, mb_max);


				auto intersect_base = model_box.intersect(mb_box);
				nanovdb::BBox<Vec> intersect_bbox(intersect_base.min(), intersect_base.max());

				if (!intersect_bbox.empty()) return mFineLevel;
			}

			return mCoarseLevel;
		}
		else if (mTestCase == BAT) {
			auto world_box = acc.tileBBox(info);
			auto world_min = world_box.min();
			auto world_max = world_box.max();

			auto model_min = BATWorldToModelPos(world_min, 0);
			auto model_max = BATWorldToModelPos(world_max, 0);
			nanovdb::BBox<Vec> model_box(model_min, model_max);

			{
				Vec mb_min(0.37531415, 0.43527624, 0.33551854);
				Vec mb_max(0.62796707, 0.7775978, 0.66308886);
				nanovdb::BBox<Vec> mb_box(mb_min, mb_max);


				auto intersect_base = model_box.intersect(mb_box);
				nanovdb::BBox<Vec> intersect_bbox(intersect_base.min(), intersect_base.max());

				if (!intersect_bbox.empty()) return mFineLevel;
			}

			return mCoarseLevel;
		}
		else if (mTestCase == FLAMINGO) {
			auto world_box = acc.tileBBox(info);
			auto world_min = world_box.min();
			auto world_max = world_box.max();

			auto model_min = BATWorldToModelPos(world_min, 0);
			auto model_max = BATWorldToModelPos(world_max, 0);
			nanovdb::BBox<Vec> model_box(model_min, model_max);

			{
				Vec mb_min(0.255108,   0.219413 ,  2.69286001);
				Vec mb_max(0.89778898, 0.74070299, 3.60383499);
				nanovdb::BBox<Vec> mb_box(mb_min, mb_max);


				auto intersect_base = model_box.intersect(mb_box);
				nanovdb::BBox<Vec> intersect_bbox(intersect_base.min(), intersect_base.max());

				if (!intersect_bbox.empty()) return mFineLevel;
			}

			return mCoarseLevel;
		}
		else {
			return mCoarseLevel;
		}
	}

	__hostdev__ bool isInParticleGenerationRegion(const T current_time, const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const Coord& l_ijk)const {
		if (mTestCase == KARMAN) {
			auto pos = acc.cellCenter(info, l_ijk);
			


			//return true;
			const Vec center(0.25, 0.5, 0.5);
			T radius = 0.175;

			//return (0.5-radius <= pos[2] && pos[2] <= 0.5+radius);

			//radius = 0.35;


			if (0.25 <= pos[1] && pos[1] <= 0.75) {
				pos[1] = center[1];
				if ((pos - center).length() <= radius) {
					return true;
				}
			}
			return false;
		}
		else if (mTestCase == SMOKESPHERE) {
			return true;
			auto pos = acc.cellCenter(info, l_ijk);
			T gen_radius = smokesphere_radius + 0.025;
			if ((pos - smokesphere_center).length() <= gen_radius) {
				return true;
			}
			return false;
		}
		else if (mTestCase == NASA) {
			auto pos = acc.cellCenter(info, l_ijk);
			auto model_pos = NASAWorldToModelPos(pos, current_time);

			return mMaskGridAccessor.GetMaskIso(model_pos);
		}
		else if (mTestCase == PROP) {
			{
				auto pos = acc.cellCenter(info, l_ijk);
				//check if it's in propeller gen region
				PropellerTransform pr[4]; createPropeller4(pr);
				for (int i = 0; i < 4; ++i) {
					auto phi = PropQuerySDFWorld(pr[i], pos, current_time);
					if (phi <= mSDFGridAccessor.gen_isovalue) return true;
				}
			}
			return false;

			//PropellerTransform p0; createPropeller1(p0);

			//auto pos = acc.cellCenter(info, l_ijk);
			////auto phi = PropQuerySDF(pos, current_time);
			//auto phi = PropQuerySDFWorld(p0, pos, current_time);

			//return phi <= mSDFGridAccessor.gen_isovalue;

			//return mSDFGridAccessor.GetMaskIso(model_pos);
			//return mMaskGridAccessor.GetMaskIso(model_pos);
		}
		else if (mTestCase == WP3D) {
			auto pos = acc.cellCenter(info, l_ijk);

			{
				//check if it's in fuselage gen region
				auto model_pos = NASAWorldToModelPos(pos, current_time);
				if (mMaskGridAccessor.GetMaskIso(model_pos)) return true;
				//return mMaskGridAccessor.GetMaskIso(model_pos);
			}
			//return false;

			{
				//check if it's in propeller gen region
				PropellerTransform pr[4]; createPropeller4(pr);
				for (int i = 0; i < 4; ++i) {
					auto phi = PropQuerySDFWorld(pr[i], pos, current_time);
					if (phi <= mSDFGridAccessor.gen_isovalue) return true;
				}
			}
			return false;
		}
		else if (mTestCase == F1CAR) {
			auto pos = acc.cellCenter(info, l_ijk);
			auto model_pos = F1CarWorldToModelPos(pos, current_time);

			auto phi = mSDFGridAccessor.linearInterpolate(model_pos[0], model_pos[1], model_pos[2]);
			return phi <= mSDFGridAccessor.gen_isovalue;
			//return mMaskGridAccessor.GetMaskIso(model_pos);
		}
		else if (mTestCase == LIZARD) {
			auto pos = acc.cellCenter(info, l_ijk);
			{
				if (mMaskGridAccessor.GetMaskIso(pos)) return true;
			}
			{
				auto phi = mSDFGridAccessor.linearInterpolate(pos[0], pos[1], pos[2]);
				if (phi <= mSDFGridAccessor.gen_isovalue) return true;
			}
			return false;
		}
		else if (mTestCase == FISH) {
			auto pos = acc.cellCenter(info, l_ijk);
			auto phi = FishQuerySDF(pos, current_time);
			//auto model_pos = FishWorldToModelPos(pos, current_time);

			//auto phi = mSDFGridAccessor.linearInterpolate(model_pos[0], model_pos[1], model_pos[2]);
			return phi <= mSDFGridAccessor.gen_isovalue;
			//return mMaskGridAccessor.GetMaskIso(model_pos);
		}
		else if (mTestCase == BAT) {
			auto pos = acc.cellCenter(info, l_ijk);
			auto phi = BATQuerySDF(pos, current_time);
			return phi <= mSDFGridAccessor.gen_isovalue;
			//return mMaskGridAccessor.GetMaskIso(model_pos);
		}
		else if (mTestCase == FLAMINGO) {
			auto pos = acc.cellCenter(info, l_ijk);
			auto phi = FlamingoQuerySDF(pos, current_time);
			return phi <= mSDFGridAccessor.gen_isovalue;
		}
		return false;
	}

	__device__ uint8_t cellType(const T current_time, const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk, int& boundary_axis, int& boundary_off) const;


	//Physically-based Simulation of Tornadoes, Xiangyang Ding
	__device__ Vec boundaryVelocityDing(const Vec& pos, const int axis, const int sgn) const {
		if (axis == 2) {//bottom/top boundary
			if (sgn == -1) {
				//zero for bottom
				return Vec(0, 0, 0);
			}
			else {
				T r = sqrt(0.05);
				auto x = pos[0];
				auto y = pos[1];

				auto rx2 = ((x - 0.5) / r) * ((x - 0.5) / r);
				auto ry2 = ((y - 0.5) / r) * ((y - 0.5) / r);
				T w = 2 * exp(-(rx2 + ry2));
				return Vec(0, 0, w);
			}
		}
		else {
			auto z = pos[2];
			//auto shear_scale = 2 * (1.0 - z);//bottom magnitude 2, top 0
			//auto shear_scale = (z >= 0.5) ? 1.0 : z * 2;
			auto shear_scale = 2;
			Vec v = Vec(1, 1, 0) * shear_scale;
			if ((axis == 0 && sgn == 1) || (axis == 1 && sgn == 1)) v[0] *= -1;
			if ((axis == 0 && sgn == -1) || (axis == 1 && sgn == 1)) v[1] *= -1;
			return v;
		}
	}

	__device__ Vec boundaryVelocityForceVortex(const Vec& pos, const int axis, const int sgn)const {
		//return boundaryVelocityDing(pos, axis, sgn);
		if (axis == 2) {//bottom/top boundary
			return Vec(0, 0, 0);
		}
		else {
			auto z = pos[2];
			//auto shear_scale = 2 * (1.0 - z) * 2;//bottom magnitude 2, top 0
			//auto shear_scale = 2 * z * 2;//bottom magnitude 0, top 2
			auto shear_scale = 2 * 1 * 2;
			Vec r = pos - Vec(0.5, 0.5, 0);
			r[2] = 0;
			Vec omega(0, 0, shear_scale);
			return omega.cross(r);
		}
	}

	//boundary axis, sgn indicates which face it is on relative to the whole computational domain
	//axis indicates the component to be set
	__device__ void setSlipBoundary(const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk, const int boundary_axis, const int boundary_off, const int axis) const {
		auto& tile = info.tile();
		//suppose boundary_axis=1, sgn=-1, l_ijk=(i,0,k), l1_ijk=(i,1,k)
		Coord l1_ijk = l_ijk; l1_ijk[boundary_axis] -= boundary_off;
		if (axis != boundary_axis) {
			tile(Tile::u_channel + axis, l_ijk) = tile(Tile::u_channel + axis, l1_ijk);
		}
		//otherwise there is a velocity node right on the face, should be either inflow/outflow or DIRICHLET
	}
	//non-slip with given velocity vector
	//vel_func(pos, boundary_axis, boundary_sgn)
	template<class FuncVII>
	__device__ void setFlowBoundary(const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk, const int boundary_axis, const int boundary_off, const FuncVII& vel_func, const int axis) const {
		auto& tile = info.tile();
		//suppose boundary_axis=1, sgn=-1, l_ijk=(i,0,k), l1_ijk=(i,1,k)
		//Coord l1_ijk = l_ijk; l1_ijk[boundary_axis] -= 1 * boundary_sgn;

		if (axis == boundary_axis) {
			if (boundary_off == -1) {
				//set face centers 0,1

				//we will not cross the tile so it's ok to use local coord
				auto l1_ijk = l_ijk; l1_ijk[boundary_axis] += 1;
				auto pos = acc.faceCenter(axis, info, l1_ijk);
				auto vel = vel_func(pos, boundary_axis, boundary_off);
				tile(Tile::u_channel + axis, l1_ijk) = vel[axis];

				Coord l2_ijk = l_ijk; l2_ijk[boundary_axis] += 2;
				tile(Tile::u_channel + axis, l_ijk) = 2 * vel[axis] - tile(Tile::u_channel + axis, l2_ijk);
			}
			else if (boundary_off == 1) {
				//auto pos = acc.faceCenter(axis, info, l_ijk);
				//auto vel = vel_func(pos, boundary_axis, boundary_sgn);
				//tile(Tile::u_channel + axis, l_ijk) = vel[axis];
			}
			else if (boundary_off == 2) {
				//set cells n-1, n-2
				//this cell is n-2

				auto pos = acc.faceCenter(axis, info, l_ijk);
				auto vel = vel_func(pos, boundary_axis, boundary_off);
				tile(Tile::u_channel + axis, l_ijk) = vel[axis];

				//cell n-3
				Coord l3_ijk = l_ijk; l3_ijk[boundary_axis] -= 1;
				Coord l1_ijk = l_ijk; l1_ijk[boundary_axis] += 1;
				tile(Tile::u_channel + axis, l1_ijk) = 2 * vel[axis] - tile(Tile::u_channel + axis, l3_ijk);
			}
		}
		else {
			Coord l1_ijk = l_ijk; l1_ijk[boundary_axis] -= boundary_off;
			auto pos0 = acc.faceCenter(axis, info, l_ijk);
			auto pos1 = acc.faceCenter(axis, info, l1_ijk);
			auto pos = (pos0 + pos1) * 0.5;
			auto vel = vel_func(pos, boundary_axis, boundary_off);
			tile(Tile::u_channel + axis, l_ijk) = 2 * vel[axis] - tile(Tile::u_channel + axis, l1_ijk);
		}
	}



	//__device__ Vec freeVortexLambOseenVelocity()

	__device__ void setBoundaryCondition(const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk, const double current_time) const {
		auto& tile = info.tile();
		auto boundary_vel_func = [this](const Vec& pos, const int axis, const int sgn) {
			if (this->mTestCase == TVORTEX) {
				return this->boundaryVelocityDing(pos, axis, sgn);
			}
			else if (this->mTestCase == FORCE_VORT) {
				return this->boundaryVelocityForceVortex(pos, axis, sgn);
			}
			};

		if (mTestCase == TVORTEX || mTestCase == FORCE_VORT) {
			int boundary_axis, boundary_sgn;
			queryBoundaryDirection(acc, info, l_ijk, boundary_axis, boundary_sgn);

			//auto g_ijk = acc.localToGlobalCoord(info, l_ijk);
			//printf("neumann cell %d,%d,%d boundary %d,%d\n", g_ijk[0], g_ijk[1], g_ijk[2], boundary_axis, boundary_sgn);

			if (tile.type(l_ijk) & NEUMANN) {
				if (boundary_axis != -1) {
					//set boundary conditions

					if (boundary_axis == 0 || boundary_axis == 1) {
						//side boundaries:
						//inflow for both u,v, free-slip for w
						//free slip means df/dn = 0
						setFlowBoundary(acc, info, l_ijk, boundary_axis, boundary_sgn, boundary_vel_func, 0);
						setFlowBoundary(acc, info, l_ijk, boundary_axis, boundary_sgn, boundary_vel_func, 1);
						setSlipBoundary(acc, info, l_ijk, boundary_axis, boundary_sgn, 2);
					}
					else if (boundary_axis == 2) {
						if (boundary_sgn == -1) {
							//bottom:
							//no-slip for u,v,w
							setFlowBoundary(acc, info, l_ijk, boundary_axis, boundary_sgn, boundary_vel_func, 0);
							setFlowBoundary(acc, info, l_ijk, boundary_axis, boundary_sgn, boundary_vel_func, 1);
							setFlowBoundary(acc, info, l_ijk, boundary_axis, boundary_sgn, boundary_vel_func, 2);
						}
						else {
							//top:
							//free-slip for u,v, outflow for w
							setSlipBoundary(acc, info, l_ijk, boundary_axis, boundary_sgn, 0);
							setSlipBoundary(acc, info, l_ijk, boundary_axis, boundary_sgn, 1);
							setFlowBoundary(acc, info, l_ijk, boundary_axis, boundary_sgn, boundary_vel_func, 2);
						}
					}

				}
			}
			else if (tile.type(l_ijk) & DIRICHLET) {
				setSlipBoundary(acc, info, l_ijk, boundary_axis, boundary_sgn, 0);
				setSlipBoundary(acc, info, l_ijk, boundary_axis, boundary_sgn, 1);
				setSlipBoundary(acc, info, l_ijk, boundary_axis, boundary_sgn, 2);
			}
		}
		else if (mTestCase == FREE_VORT || mTestCase == DUAL_VORT) {
			auto zero_vel_func = []__device__(const Vec & pos, const int axis, const int sgn) {
				return Vec(0, 0, 0);
			};
			int boundary_axis, boundary_sgn;
			if (queryBoundaryDirection(acc, info, l_ijk, boundary_axis, boundary_sgn)) {
				for (int axis : {0, 1, 2}) {
					if (axis != boundary_axis) {
						setSlipBoundary(acc, info, l_ijk, boundary_axis, boundary_sgn, axis);
					}
					//else if (tile.type(l_ijk) == NEUMANN) {
					else {
						setFlowBoundary(acc, info, l_ijk, boundary_axis, boundary_sgn, zero_vel_func, axis);
					}
				}
			}
		}
		else if (mTestCase == LEAP_FROG) {
			auto zero_vel_func = []__device__(const Vec & pos, const int axis, const int sgn) {
				return Vec(0, 0, 0);
			};
			int boundary_axis, boundary_sgn;
			if (queryBoundaryDirection(acc, info, l_ijk, boundary_axis, boundary_sgn)) {
				for (int axis : {0, 1, 2}) {
					if (axis != boundary_axis) {
						setSlipBoundary(acc, info, l_ijk, boundary_axis, boundary_sgn, axis);
					}
					else if (tile.type(l_ijk) == NEUMANN) {
						setFlowBoundary(acc, info, l_ijk, boundary_axis, boundary_sgn, zero_vel_func, axis);
					}
				}
			}
		}
		else if (mTestCase == KARMAN) {
			auto source_vel = karman_source;
			auto inflow_vel_func = [=]__device__(const Vec & pos, const int axis, const int sgn) {
				return Vec(source_vel, 0, 0);
				//if (axis == 0) return Vec(source_vel, 0, 0);
				//else return Vec(0, 0, 0);
			};
			//auto zero_vel_func = []__device__(const Vec & pos, const int axis, const int sgn) {
			//	return Vec(0, 0, 0);
			//};
			int boundary_axis, boundary_sgn;
			if (queryBoundaryDirection(acc, info, l_ijk, boundary_axis, boundary_sgn)) {
				for (int axis : {0, 1, 2}) {
					if (axis != boundary_axis) {
						setSlipBoundary(acc, info, l_ijk, boundary_axis, boundary_sgn, axis);
					}
					else if (tile.type(l_ijk) == NEUMANN) {
						setFlowBoundary(acc, info, l_ijk, boundary_axis, boundary_sgn, inflow_vel_func, axis);
					}
				}
			}
		}
		else if (mTestCase == SMOKESPHERE) {
			//auto source_vel = smokesphere_source;
			//auto inflow_vel_func = [=]__device__(const Vec & pos, const int axis, const int sgn) {
			//	return Vec(0, 0, source_vel);
			//	//if ((pos - Vec(0.5, 0.5, 0)).length() <= smokesphere_radius) {
			//	//	return Vec(0, 0,source_vel);
			//	//}
			//	//else {
			//	//	return Vec(0, 0, 0);
			//	//}
			//};
			//int boundary_axis, boundary_sgn;
			//if (queryBoundaryDirection(acc, info, l_ijk, boundary_axis, boundary_sgn)) {
			//	for (int axis : {0, 1, 2}) {
			//		if (axis != boundary_axis) {
			//			setSlipBoundary(acc, info, l_ijk, boundary_axis, boundary_sgn, axis);
			//		}
			//		else if (tile.type(l_ijk) == NEUMANN) {
			//			setFlowBoundary(acc, info, l_ijk, boundary_axis, boundary_sgn, inflow_vel_func, axis);
			//		}
			//	}
			//}


			//Vec vel(0, 0, -source_vel);
			//Vec vel(source_vel, 0, 0);
			//Vec omega(0, 5, 0);
			Vec vel = smokesphere_source;
			if (tile.type(l_ijk) & NEUMANN) {
				int boundary_axis, boundary_sgn;
				if (queryEffectiveBoundaryDirection1(acc, mCoarseLevel, info, l_ijk, boundary_axis, boundary_sgn)) {
					for (int axis : {0, 1, 2}) {
						//setFlowBoundary(acc, info, l_ijk, boundary_axis, boundary_sgn, inflow_vel_func, axis);

						//auto fpos = acc.faceCenter(axis, info, l_ijk);
						//auto r = fpos - Vec(0.5, 0.5, 0.5);
						//r[1] = 0;
						//auto vel = omega.cross(r);

						tile(Tile::u_channel + axis, l_ijk) = vel[axis];
					}
				}
			}
			else if (tile.type(l_ijk) & INTERIOR) {
				for (int axis : {0, 1, 2}) {
					auto nl_ijk = l_ijk; nl_ijk[axis] -= 1;
					int boundary_axis, boundary_sgn;
					if (queryEffectiveBoundaryDirection1(acc, mCoarseLevel, info, nl_ijk, boundary_axis, boundary_sgn)) {
						if (boundary_sgn == -1) {

							//auto fpos = acc.faceCenter(axis, info, l_ijk);
							//auto r = fpos - Vec(0.5, 0.5, 0.5);
							//r[1] = 0;
							//auto vel = omega.cross(r);

							tile(Tile::u_channel + axis, l_ijk) = vel[axis];
						}
					}
					//tile(Tile::u_channel + axis, l_ijk) = vel[axis];
				}
			}
		}
		else if (mTestCase == NASA) {
			Vec vel(0, 0, nasa_source);
			int chk_level = 1;
			if (tile.type(l_ijk) & (NEUMANN | DIRICHLET)) {
				int boundary_axis, boundary_sgn;
				if (queryEffectiveBoundaryDirection1(acc, chk_level, info, l_ijk, boundary_axis, boundary_sgn)) {
					for (int axis : {0, 1, 2}) {
						//setFlowBoundary(acc, info, l_ijk, boundary_axis, boundary_sgn, inflow_vel_func, axis);
						tile(Tile::u_channel + axis, l_ijk) = vel[axis];
					}
				}
			}
			else if (tile.type(l_ijk) & INTERIOR) {
				for (int axis : {0, 1, 2}) {
					auto nl_ijk = l_ijk; nl_ijk[axis] -= 1;
					int boundary_axis, boundary_sgn;
					if (queryEffectiveBoundaryDirection1(acc, chk_level, info, nl_ijk, boundary_axis, boundary_sgn)) {
						if (boundary_sgn == -1) {
							tile(Tile::u_channel + axis, l_ijk) = vel[axis];
						}
					}
					//tile(Tile::u_channel + axis, l_ijk) = vel[axis];
				}
			}
		}
		else if (mTestCase == PROP) {
			Vec wall_vel(0, 0, prop_source);
			//PropellerTransform p0; createPropeller1(p0);
			PropellerTransform prs[4]; createPropeller4(prs);


			int chk_level = 1;
			if (tile.type(l_ijk) & (NEUMANN | DIRICHLET)) {
				//case 1: wall boundary
				int boundary_axis, boundary_sgn;
				if (queryEffectiveBoundaryDirection1(acc, chk_level, info, l_ijk, boundary_axis, boundary_sgn)) {
					for (int axis : {0, 1, 2}) {
						//setFlowBoundary(acc, info, l_ijk, boundary_axis, boundary_sgn, inflow_vel_func, axis);
						tile(Tile::u_channel + axis, l_ijk) = wall_vel[axis];
					}
				}
				else {
					//either propeller or fuselage
					for (int axis : {0, 1, 2}) {
						Vec vel(0, 0, 0);
						auto fpos = acc.faceCenter(axis, info, l_ijk);

						//case 2: propeller boundary
						//if it's in propeller region, set velocity to propeller velocity + source
						for (int i = 0; i < 4; ++i) {
							if (propellerBoundaryVelocity4(prs, fpos, current_time, vel)) {
								vel[2] = prop_source;
								break;
							}
						}
						//if it's fuselage, then vel is 0
						tile(Tile::u_channel + axis, l_ijk) = vel[axis];
					}
				}
			}
			else if (tile.type(l_ijk) & INTERIOR) {
				Vec wall_vel(0, 0, prop_source);
				for (int axis : {0, 1, 2}) {
					auto nl_ijk = l_ijk; nl_ijk[axis] -= 1;
					int boundary_axis, boundary_sgn;
					//solid neighbor
					if (queryEffectiveBoundaryDirection1(acc, chk_level, info, nl_ijk, boundary_axis, boundary_sgn)) {
						if (boundary_sgn == -1) {
							tile(Tile::u_channel + axis, l_ijk) = wall_vel[axis];
						}
					}
					else {
						auto ng_ijk = acc.localToGlobalCoord(info, l_ijk); ng_ijk[axis] -= 1;
						HATileInfo<Tile> ninfo; Coord nl_ijk;
						if (acc.findVoxel(info.mLevel, ng_ijk, ninfo, nl_ijk)) {
							auto& ntile = ninfo.tile();
							if (ntile.type(nl_ijk) & NEUMANN) {
								Vec vel(0, 0, 0);
								auto fpos = acc.faceCenter(axis, info, l_ijk);
								for (int i = 0; i < 4; ++i) {
									if (propellerBoundaryVelocity4(prs, fpos, current_time, vel)) {
										vel[2] = prop_source;
										break;
									}
								}
								tile(Tile::u_channel + axis, l_ijk) = vel[axis];
							}
						}
					}

					//tile(Tile::u_channel + axis, l_ijk) = vel[axis];
				}
			}
		}
		else if (mTestCase == WP3D) {
			Vec wall_vel(0, 0, nasa_source);
			PropellerTransform prs[4]; createPropeller4(prs);

			int chk_level = 1;
			if (tile.type(l_ijk) & (NEUMANN | DIRICHLET)) {
				//case 1: wall boundary
				int boundary_axis, boundary_sgn;
				if (queryEffectiveBoundaryDirection1(acc, chk_level, info, l_ijk, boundary_axis, boundary_sgn)) {
					for (int axis : {0, 1, 2}) {
						//setFlowBoundary(acc, info, l_ijk, boundary_axis, boundary_sgn, inflow_vel_func, axis);
						tile(Tile::u_channel + axis, l_ijk) = wall_vel[axis];
					}
				}
				else {
					//either propeller or fuselage
					for (int axis : {0, 1, 2}) {
						Vec vel(0, 0, 0);
						auto fpos = acc.faceCenter(axis, info, l_ijk);

						//case 2: propeller boundary
						//if it's in propeller region, set velocity to propeller velocity + source
						for (int i = 0; i < 4; ++i) {
							if (propellerBoundaryVelocity4(prs, fpos, current_time, vel)) {
								//vel[2] = nasa_source;
								break;
							}
						}
						//if it's fuselage, then vel is 0
						tile(Tile::u_channel + axis, l_ijk) = vel[axis];
					}
				}
			}
			else if (tile.type(l_ijk) & INTERIOR) {
				for (int axis : {0, 1, 2}) {
					auto nl_ijk = l_ijk; nl_ijk[axis] -= 1;
					int boundary_axis, boundary_sgn;
					//solid neighbor
					if (queryEffectiveBoundaryDirection1(acc, chk_level, info, nl_ijk, boundary_axis, boundary_sgn)) {
						if (boundary_sgn == -1) {
							tile(Tile::u_channel + axis, l_ijk) = wall_vel[axis];
						}
					}
					else {
						auto ng_ijk = acc.localToGlobalCoord(info, l_ijk); ng_ijk[axis] -= 1;
						HATileInfo<Tile> ninfo; Coord nl_ijk;
						if (acc.findVoxel(info.mLevel, ng_ijk, ninfo, nl_ijk)) {
							auto& ntile = ninfo.tile();
							if (ntile.type(nl_ijk) & NEUMANN) {
								Vec vel(0, 0, 0);
								auto fpos = acc.faceCenter(axis, info, l_ijk);
								for (int i = 0; i < 4; ++i) {
									if (propellerBoundaryVelocity4(prs, fpos, current_time, vel)) {
										//vel[2] = nasa_source;
										break;
									}
								}
								tile(Tile::u_channel + axis, l_ijk) = vel[axis];
							}
						}
					}

					//tile(Tile::u_channel + axis, l_ijk) = vel[axis];
				}
			}
		}
		else if (mTestCase == F1CAR) {
			Vec vel(0, 0, f1_source);
			int chk_level = 1;
			if (tile.type(l_ijk) & (NEUMANN | DIRICHLET)) {
				int boundary_axis, boundary_sgn;
				if (queryEffectiveBoundaryDirection1(acc, chk_level, info, l_ijk, boundary_axis, boundary_sgn)) {
					for (int axis : {0, 1, 2}) {
						//setFlowBoundary(acc, info, l_ijk, boundary_axis, boundary_sgn, inflow_vel_func, axis);
						tile(Tile::u_channel + axis, l_ijk) = vel[axis];
					}
				}
			}
			else if (tile.type(l_ijk) & INTERIOR) {
				for (int axis : {0, 1, 2}) {
					auto nl_ijk = l_ijk; nl_ijk[axis] -= 1;
					int boundary_axis, boundary_sgn;
					if (queryEffectiveBoundaryDirection1(acc, chk_level, info, nl_ijk, boundary_axis, boundary_sgn)) {
						if (boundary_sgn == -1) {
							tile(Tile::u_channel + axis, l_ijk) = vel[axis];
						}
					}
					//tile(Tile::u_channel + axis, l_ijk) = vel[axis];
				}
			}
		}
		else if (mTestCase == LIZARD) {
			Vec vel(lizard_source, 0, 0);
			//Vec vel(0, lizard_source, 0);
			int chk_level = 2;
			if (tile.type(l_ijk) & (NEUMANN | DIRICHLET)) {
				int boundary_axis, boundary_sgn;
				if (queryEffectiveBoundaryDirection1(acc, chk_level, info, l_ijk, boundary_axis, boundary_sgn)) {
					for (int axis : {0, 1, 2}) {
						//setFlowBoundary(acc, info, l_ijk, boundary_axis, boundary_sgn, inflow_vel_func, axis);
						tile(Tile::u_channel + axis, l_ijk) = vel[axis];
					}
				}
			}
			else if (tile.type(l_ijk) & INTERIOR) {
				for (int axis : {0, 1, 2}) {
					auto nl_ijk = l_ijk; nl_ijk[axis] -= 1;
					int boundary_axis, boundary_sgn;
					if (queryEffectiveBoundaryDirection1(acc, chk_level, info, nl_ijk, boundary_axis, boundary_sgn)) {
						if (boundary_sgn == -1) {
							tile(Tile::u_channel + axis, l_ijk) = vel[axis];
						}
					}
					//tile(Tile::u_channel + axis, l_ijk) = vel[axis];
				}
			}
		}
		else if (mTestCase == FISH) {
			Vec vel(fish_source, 0, 0);
			int chk_level = 1;
			if (tile.type(l_ijk) & (NEUMANN | DIRICHLET)) {
				int boundary_axis, boundary_sgn;
				if (queryEffectiveBoundaryDirection1(acc, chk_level, info, l_ijk, boundary_axis, boundary_sgn)) {
					for (int axis : {0, 1, 2}) {
						//setFlowBoundary(acc, info, l_ijk, boundary_axis, boundary_sgn, inflow_vel_func, axis);
						tile(Tile::u_channel + axis, l_ijk) = vel[axis];
					}
				}
			}
			else if (tile.type(l_ijk) & INTERIOR) {
				for (int axis : {0, 1, 2}) {
					auto nl_ijk = l_ijk; nl_ijk[axis] -= 1;
					int boundary_axis, boundary_sgn;
					if (queryEffectiveBoundaryDirection1(acc, chk_level, info, nl_ijk, boundary_axis, boundary_sgn)) {
						if (boundary_sgn == -1) {
							tile(Tile::u_channel + axis, l_ijk) = vel[axis];
						}
					}
					//tile(Tile::u_channel + axis, l_ijk) = vel[axis];
				}
			}
		}
		else if (mTestCase == BAT) {
			Vec wall_vel(bat_source, 0, 0);
			int chk_level = 1;
			if (tile.type(l_ijk) & (NEUMANN | DIRICHLET)) {
				int boundary_axis, boundary_sgn;
				if (queryEffectiveBoundaryDirection1(acc, chk_level, info, l_ijk, boundary_axis, boundary_sgn)) {
					for (int axis : {0, 1, 2}) {
						//setFlowBoundary(acc, info, l_ijk, boundary_axis, boundary_sgn, inflow_vel_func, axis);
						tile(Tile::u_channel + axis, l_ijk) = wall_vel[axis];
					}
				}
				else {
					//bat body
					for (int axis : {0, 1, 2}) {
						auto fpos = acc.faceCenter(axis, info, l_ijk);
						Vec vel = BATQuerySolidVelocity(fpos, current_time);
						//printf("vel %f %f %f\n", vel[0], vel[1], vel[2]);
						tile(Tile::u_channel + axis, l_ijk) = vel[axis];
					}
				}
			}
			else if (tile.type(l_ijk) & INTERIOR) {
				for (int axis : {0, 1, 2}) {
					auto nl_ijk = l_ijk; nl_ijk[axis] -= 1;
					int boundary_axis, boundary_sgn;
					if (queryEffectiveBoundaryDirection1(acc, chk_level, info, nl_ijk, boundary_axis, boundary_sgn)) {
						if (boundary_sgn == -1) {
							tile(Tile::u_channel + axis, l_ijk) = wall_vel[axis];
						}
					}
					else {
						auto ng_ijk = acc.localToGlobalCoord(info, l_ijk); ng_ijk[axis] -= 1;
						HATileInfo<Tile> ninfo; Coord nl_ijk;
						if (acc.findVoxel(info.mLevel, ng_ijk, ninfo, nl_ijk)) {
							auto& ntile = ninfo.tile();
							if (ntile.type(nl_ijk) & NEUMANN) {
								auto fpos = acc.faceCenter(axis, info, l_ijk);
								Vec vel = BATQuerySolidVelocity(fpos, current_time);
								tile(Tile::u_channel + axis, l_ijk) = vel[axis];
							}
						}
					}
					//tile(Tile::u_channel + axis, l_ijk) = vel[axis];
				}
			}
		}
		else if (mTestCase == FLAMINGO) {
			Vec vel(flamingo_source, 0, 0);
			int chk_level = 1;
			if (tile.type(l_ijk) & (NEUMANN | DIRICHLET)) {
				int boundary_axis, boundary_sgn;
				if (queryEffectiveBoundaryDirection1(acc, chk_level, info, l_ijk, boundary_axis, boundary_sgn)) {
					for (int axis : {0, 1, 2}) {
						//setFlowBoundary(acc, info, l_ijk, boundary_axis, boundary_sgn, inflow_vel_func, axis);
						tile(Tile::u_channel + axis, l_ijk) = vel[axis];
					}
				}
			}
			else if (tile.type(l_ijk) & INTERIOR) {
				for (int axis : {0, 1, 2}) {
					auto nl_ijk = l_ijk; nl_ijk[axis] -= 1;
					int boundary_axis, boundary_sgn;
					if (queryEffectiveBoundaryDirection1(acc, chk_level, info, nl_ijk, boundary_axis, boundary_sgn)) {
						if (boundary_sgn == -1) {
							tile(Tile::u_channel + axis, l_ijk) = vel[axis];
						}
					}
					//tile(Tile::u_channel + axis, l_ijk) = vel[axis];
				}
			}
		}
	}


	__device__ void enforceDyeDensityBoundaryCondition(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk) const {
		if (mTestCase == TVORTEX || mTestCase == FORCE_VORT) {
			auto get_bbox = [](const Vec center, T halfside) {
				return nanovdb::BBox<Vec>(center - Vec(halfside, halfside, halfside), center + Vec(halfside, halfside, halfside));
				};
			T halfside = 0.05 / 2.0;
			//4 bottom edge centers
			//auto box1 = get_bbox(Vec(halfside, 0.5, 0.1), halfside);
			//auto box2 = get_bbox(Vec(0.5, halfside, 0.1), halfside);
			//auto box3 = get_bbox(Vec(0.5, 1 - halfside, 0.1), halfside);
			//auto box4 = get_bbox(Vec(1 - halfside, 0.5, 0.1), halfside);

			//4 corners
			auto box1 = get_bbox(Vec(halfside, halfside, 0.1), halfside);
			auto box2 = get_bbox(Vec(1 - halfside, halfside, 0.1), halfside);
			auto box3 = get_bbox(Vec(1 - halfside, 1 - halfside, 0.1), halfside);
			auto box4 = get_bbox(Vec(halfside, 1 - halfside, 0.1), halfside);

			//3 points from center
			//auto box1 = get_bbox(Vec(1-halfside, 0.5, 0.1), halfside);
			//auto box2 = get_bbox(Vec(0.75, 0.5, 0.1), halfside);
			//auto box3 = get_bbox(Vec(0.5, 0.5, 0.1), halfside);


			auto pos = acc.cellCenter(info, l_ijk);
			auto& tile = info.tile();
			if (box1.isInside(pos) || box2.isInside(pos) || box3.isInside(pos) || box4.isInside(pos)) {
				//if (box1.isInside(pos) || box2.isInside(pos) || box3.isInside(pos)) {
					//dye density is 1
				tile(Tile::dye_channel, l_ijk) = 1;
			}
		}
	}

	//gamma: vortex strength
	//radius: radius of the vortex ring
	//delta: thickness of the vortex ring
	//center: center of the vortex ring
	//unit_x: unit vector of vortex in the x direction
	//unit_y: unit vector of vortex in the y direction
	//num_samples: number of sample points on the vortex ring
	//reference: https://github.com/zjw49246/particle-flow-maps/blob/main/3D/init_conditions.py
	__device__ void addVortexRingInitialVelocityAndSmoke(const Vec& pos, double gamma, float radius, float delta, const Vec& center, const Vec& unit_x, const Vec& unit_y, int num_samples, Vec& velocity, T& smoke) const {


		// Curve length per sample point
		float curve_length = (2 * M_PI * radius) / num_samples;

		// Loop through each sample point on the vortex ring
		for (int l = 0; l < num_samples; ++l) {
			float theta = l / float(num_samples) * 2 * M_PI;
			Vec p_sampled = radius * (cos(theta) * unit_x + sin(theta) * unit_y) + center;

			Vec p_diff = pos - p_sampled;
			float r = p_diff.length();

			Vec w_vector = gamma * (-sin(theta) * unit_x + cos(theta) * unit_y);
			float decay_factor = exp(-pow(r / delta, 3));

			// Biot-Savart law contribution
			velocity += curve_length * (-1 / (4 * M_PI * r * r * r)) * (1 - decay_factor) * p_diff.cross(w_vector);

			// Smoke density update based on decay factor
			smoke += (curve_length * decay_factor);
		}
	}

	//set type, velocity, smoke
	__device__ void setInitialCondition(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk)const;
};