#include "FluidParams.h"

__hostdev__ PropellerTransform::PropellerTransform(
	const Vec& model_center,
	const Vec& world_center,
	float attack_angle,
	float scale_factor,
	float omega,
	float theta0
)
	: model_center_(model_center),
	world_center_(world_center),
	attack_angle_(attack_angle),
	scale_factor_(scale_factor),
	omega_(omega),
	theta0_(theta0) {}

// 在 x-y 平面旋转
__hostdev__ PropellerTransform::Vec PropellerTransform::rotateXY(const Vec& point, float theta) const {
	float cosTheta = std::cos(theta);
	float sinTheta = std::sin(theta);
	return Vec(
		point[0] * cosTheta - point[1] * sinTheta,
		point[0] * sinTheta + point[1] * cosTheta,
		point[2]
	);
}

// 在 y-z 平面旋转
__hostdev__ PropellerTransform::Vec PropellerTransform::rotateYZ(const Vec& point, float alpha) const {
	float cosAlpha = std::cos(alpha);
	float sinAlpha = std::sin(alpha);
	return Vec(
		point[0],
		point[1] * cosAlpha - point[2] * sinAlpha,
		point[1] * sinAlpha + point[2] * cosAlpha
	);
}

// 正向变换：从 model -> world
__hostdev__ PropellerTransform::Vec PropellerTransform::transformModelToWorld(const Vec& point, float time) const {
	// 计算旋转角度 theta
	float theta = theta0_ + time * omega_;

	// 平移到原点
	Vec transformed = point - model_center_;

	// 旋转变换
	transformed = rotateXY(transformed, theta);
	transformed = rotateYZ(transformed, attack_angle_);

	// 缩放
	transformed *= scale_factor_;

	// 平移到世界坐标
	transformed += world_center_;

	return transformed;
}

// 逆向变换：从 world -> model
__hostdev__ PropellerTransform::Vec PropellerTransform::transformWorldToModel(const Vec& point, float time) const {
	// 计算旋转角度 theta
	float theta = theta0_ + time * omega_;

	// 平移回局部坐标
	Vec transformed = point - world_center_;

	// 逆缩放
	transformed /= scale_factor_;

	// 逆旋转变换
	transformed = rotateYZ(transformed, -attack_angle_);
	transformed = rotateXY(transformed, -theta);

	// 平移回模型坐标
	transformed += model_center_;

	return transformed;
}

__hostdev__ PropellerTransform::Vec PropellerTransform::transformModelVelocityToWorld(const Vec& velocity, float time) const {
	// 计算旋转角度 theta
	float theta = theta0_ + time * omega_;

	// 计算旋转速度的影响
	// 对应的速度变换包括旋转矩阵的影响
	Vec rotated_velocity = rotateXY(velocity, theta);
	rotated_velocity = rotateYZ(rotated_velocity, attack_angle_);

	// 缩放速度
	rotated_velocity *= scale_factor_;

	return rotated_velocity;
}

// 根据旋转计算model space中的速度
__hostdev__ PropellerTransform::Vec PropellerTransform::calculateModelVelocityFromRotation(const Vec& point, float time) const {
	// 计算点相对于模型中心的相对位置矢量
	Vec relative_position = point - model_center_;

	// 计算速度：ω × r (角速度与相对位置的叉乘)
	float velocity_x = omega_ * relative_position[1];  // 旋转对y轴的影响
	float velocity_y = -omega_ * relative_position[0]; // 旋转对x轴的影响
	float velocity_z = 0.0f;                           // 假设旋转在xy平面，不影响z轴

	return Vec(velocity_x, velocity_y, velocity_z);
}

// 计算world space的velocity
__hostdev__ PropellerTransform::Vec PropellerTransform::calculateWorldVelocity(const Vec& point, float time) const {
	// 步骤 1: 将世界空间的点变换到模型空间
	Vec point_in_model = transformWorldToModel(point, time);

	// 步骤 2: 计算模型空间中的速度
	Vec velocity_in_model = calculateModelVelocityFromRotation(point_in_model, time);

	// 步骤 3: 将速度从模型空间变换回世界空间
	Vec velocity_in_world = transformModelVelocityToWorld(velocity_in_model, time);

	return velocity_in_world;
}

__device__ uint8_t FluidParams::cellType(const T current_time, const HATileAccessor<Tile>& acc, const HATileInfo<Tile>& info, const nanovdb::Coord& l_ijk, int& boundary_axis, int& boundary_off) const {
	if (mTestCase == TVORTEX || mTestCase == FORCE_VORT || mTestCase == FREE_VORT || mTestCase == DUAL_VORT) {
		//top air
		//side walls
		bool is_neumann = false;
		bool is_dirichlet = false;
		acc.iterateSameLevelNeighborVoxels(info, l_ijk, [&]__device__(const HATileInfo<Tile>&n_info, const Coord & n_l_ijk, const int axis, const int sgn) {
			if (n_info.empty()) {
				boundary_axis = axis;
				boundary_off = sgn;

				if (axis == 2 && sgn == 1) is_dirichlet = true;
				else is_neumann = true;
				//is_neumann = true;
			}
		});

		if (is_neumann) return NEUMANN;
		else if (is_dirichlet) return DIRICHLET;
		else return INTERIOR;
	}
	else if (mTestCase == LEAP_FROG) {
		//x-, x+ air
		//other walls
		bool is_neumann = false;
		bool is_dirichlet = false;

		//int boundary_axis, boundary_off;
		if (queryBoundaryDirection(acc, info, l_ijk, boundary_axis, boundary_off)) {
			is_neumann = true;
		}
		//acc.iterateSameLevelNeighborVoxels(info, l_ijk, [&]__device__(const HATileInfo<Tile>&n_info, const Coord & n_l_ijk, const int axis, const int sgn) {
		//	if (n_info.empty()) {
		//		//if (axis == 0) is_dirichlet = true;
		//		//else is_neumann = true;
		//		is_neumann = true;
		//	}
		//});

		if (is_neumann) return NEUMANN;
		else if (is_dirichlet) return DIRICHLET;
		else return INTERIOR;
	}
	else if (mTestCase == KARMAN) {
		//x+ air
		//other walls
		bool is_neumann = false;
		bool is_dirichlet = false;



		//int boundary_axis, boundary_off;
		if (queryBoundaryDirection(acc, info, l_ijk, boundary_axis, boundary_off)) {
			is_neumann = true;
		}




		//acc.iterateSameLevelNeighborVoxels(info, l_ijk, [&]__device__(const HATileInfo<Tile>&n_info, const Coord & n_l_ijk, const int axis, const int sgn) {
		//	if (n_info.empty()) {
		//		is_neumann = true;
		//		//if (axis == 0 && sgn == -1) is_neumann = true;
		//		//else is_dirichlet = true;
		//		//if (axis == 0 && sgn == 1) is_dirichlet = true;
		//		//else is_neumann = true;
		//	}
		//});

		const Vec center(0.25, 0.5, 0.5);
		const T radius = 0.15;
		auto pos = acc.cellCenter(info, l_ijk);

		//if ((pos - center).length() <= radius) {
		//	return NEUMANN;
		//}

		if (0.25 <= pos[1] && pos[1] <= 0.75) {
			pos[1] = center[1];
			if ((pos - center).length() <= radius) {
				return NEUMANN;
			}
		}


		if (is_neumann) return NEUMANN;
		else if (is_dirichlet) return DIRICHLET;
		else return INTERIOR;
	}
	else if (mTestCase == SMOKESPHERE) {
		//z+ air
		//other walls
		bool is_neumann = false;
		bool is_dirichlet = false;

		//int boundary_axis, boundary_off;
		if (queryEffectiveBoundaryDirection1(acc, mCoarseLevel, info, l_ijk, boundary_axis, boundary_off)) {



			is_neumann = true;
		}

		//int boundary_axis, boundary_off;
		//if (queryBoundaryDirection(acc, info, l_ijk, boundary_axis, boundary_off)) {
		//	//if (!(boundary_axis == 2 && boundary_off > 0)) {
		//	//	is_neumann = true;
		//	//}
		//	//else {
		//	//	is_dirichlet = true;
		//	//}
		//	is_neumann = true;
		//}

		if (is_neumann) return NEUMANN;
		else if (is_dirichlet) return DIRICHLET;

		auto pos = acc.cellCenter(info, l_ijk);
		if ((pos - smokesphere_center).length() <= smokesphere_radius) {
			return NEUMANN;
		}
		return INTERIOR;
	}
	else if (mTestCase == NASA) {
		//z+ air
		//other walls
		int chk_level = 1;
		bool is_neumann = false;
		bool is_dirichlet = false;

		if (queryEffectiveBoundaryDirection1(acc, chk_level, info, l_ijk, boundary_axis, boundary_off)) {
			is_neumann = true;
			//if (boundary_axis == 0 && boundary_off > 0) is_dirichlet = true;
			//else is_neumann = true;
		}
		if (is_neumann) return NEUMANN;
		else if (is_dirichlet) return DIRICHLET;

		auto pos = acc.cellCenter(info, l_ijk);
		auto model_pos = NASAWorldToModelPos(pos, current_time);
		if (mMaskGridAccessor.GetMask0(model_pos)) return NEUMANN;
		return INTERIOR;
	}
	else if (mTestCase == PROP) {
		//z+ air
		//other walls
		int chk_level = 1;
		bool is_neumann = false;
		bool is_dirichlet = false;

		if (queryEffectiveBoundaryDirection1(acc, chk_level, info, l_ijk, boundary_axis, boundary_off)) {
			is_neumann = true;
			//if (boundary_axis == 0 && boundary_off > 0) is_dirichlet = true;
			//else is_neumann = true;
		}
		if (is_neumann) return NEUMANN;
		else if (is_dirichlet) return DIRICHLET;
		auto pos = acc.cellCenter(info, l_ijk);
		{
			//check propellers
			PropellerTransform prs[4]; createPropeller4(prs);
			for (int i = 0; i < 4; i++) {
				auto phi = PropQuerySDFWorld(prs[i], pos, current_time);
				if (phi <= mSDFGridAccessor.solid_isovalue) return NEUMANN;
			}
		}

		//PropellerTransform p0; createPropeller1(p0);
		
		//auto phi = PropQuerySDF(pos, current_time);
		//auto phi = PropQuerySDFWorld(p0, pos, current_time);
		//if (phi <= mSDFGridAccessor.solid_isovalue) return NEUMANN;

		//if (mMaskGridAccessor.GetMask0(model_pos)) return NEUMANN;
		return INTERIOR;
	}
	else if (mTestCase == WP3D) {
		//z+ air
		//other walls
		int chk_level = 1;
		bool is_neumann = false;
		bool is_dirichlet = false;

		if (queryEffectiveBoundaryDirection1(acc, chk_level, info, l_ijk, boundary_axis, boundary_off)) {
			is_neumann = true;
			//if (boundary_axis == 0 && boundary_off > 0) is_dirichlet = true;
			//else is_neumann = true;
		}
		if (is_neumann) return NEUMANN;
		else if (is_dirichlet) return DIRICHLET;

		auto pos = acc.cellCenter(info, l_ijk);
		{
			//check fuselage
			auto model_pos = NASAWorldToModelPos(pos, current_time);
			if (mMaskGridAccessor.GetMask0(model_pos)) return NEUMANN;
		}

		{
			//check propellers
			PropellerTransform prs[4]; createPropeller4(prs);
			for (int i = 0; i < 4; i++) {
				auto phi = PropQuerySDFWorld(prs[i], pos, current_time);
				if (phi <= mSDFGridAccessor.solid_isovalue) return NEUMANN;
			}
		}


		return INTERIOR;
	}
	else if (mTestCase == F1CAR) {
		//z+ air
		//other walls
		int chk_level = 1;
		bool is_neumann = false;
		bool is_dirichlet = false;

		if (queryEffectiveBoundaryDirection1(acc, chk_level, info, l_ijk, boundary_axis, boundary_off)) {
			is_neumann = true;
			//if (boundary_axis == 0 && boundary_off > 0) is_dirichlet = true;
			//else is_neumann = true;
		}
		if (is_neumann) return NEUMANN;
		else if (is_dirichlet) return DIRICHLET;

		auto pos = acc.cellCenter(info, l_ijk);
		auto model_pos = F1CarWorldToModelPos(pos, current_time);
		auto phi = mSDFGridAccessor.linearInterpolate(model_pos[0], model_pos[1], model_pos[2]);
		if (phi <= mSDFGridAccessor.solid_isovalue) return NEUMANN;
		//if (mMaskGridAccessor.GetMask0(model_pos)) return NEUMANN;
		return INTERIOR;
	}
	else if (mTestCase == LIZARD) {
		//z+ air
//other walls
		int chk_level = 2;
		bool is_neumann = false;
		bool is_dirichlet = false;

		if (queryEffectiveBoundaryDirection1(acc, chk_level, info, l_ijk, boundary_axis, boundary_off)) {
			is_neumann = true;
			//if (boundary_axis == 0 && boundary_off > 0) is_dirichlet = true;
			//else is_neumann = true;
		}
		if (is_neumann) return NEUMANN;
		else if (is_dirichlet) return DIRICHLET;

		auto pos = acc.cellCenter(info, l_ijk);
		{
			if (mMaskGridAccessor.GetMask0(pos)) return NEUMANN;
		}
		{
			auto phi = mSDFGridAccessor.linearInterpolate(pos[0], pos[1], pos[2]);
			if (phi <= mSDFGridAccessor.solid_isovalue) return NEUMANN;
		}
		return INTERIOR;
	}
	else if (mTestCase == FISH) {
		//z+ air
		//other walls
		int chk_level = 1;
		bool is_neumann = false;
		bool is_dirichlet = false;

		if (queryEffectiveBoundaryDirection1(acc, chk_level, info, l_ijk, boundary_axis, boundary_off)) {
			is_neumann = true;
			//if (boundary_axis == 0 && boundary_off > 0) is_dirichlet = true;
			//else is_neumann = true;
		}
		if (is_neumann) return NEUMANN;
		else if (is_dirichlet) return DIRICHLET;

		auto pos = acc.cellCenter(info, l_ijk);
		//auto model_pos = FishWorldToModelPos(pos, current_time);
		//auto phi = mSDFGridAccessor.linearInterpolate(model_pos[0], model_pos[1], model_pos[2]);
		auto phi = FishQuerySDF(pos, current_time);
		if (phi <= mSDFGridAccessor.solid_isovalue) return NEUMANN;
		//if (mMaskGridAccessor.GetMask0(model_pos)) return NEUMANN;
		return INTERIOR;
	}
	else if (mTestCase == BAT) {
		//z+ air
		//other walls
		int chk_level = 1;
		bool is_neumann = false;
		bool is_dirichlet = false;

		if (queryEffectiveBoundaryDirection1(acc, chk_level, info, l_ijk, boundary_axis, boundary_off)) {
			is_neumann = true;
			//if (boundary_axis == 0 && boundary_off > 0) is_dirichlet = true;
			//else is_neumann = true;
		}
		if (is_neumann) return NEUMANN;
		else if (is_dirichlet) return DIRICHLET;

		auto pos = acc.cellCenter(info, l_ijk);
		auto phi = BATQuerySDF(pos, current_time);
		if (phi <= mSDFGridAccessor.solid_isovalue) return NEUMANN;
		//if (mMaskGridAccessor.GetMask0(model_pos)) return NEUMANN;
		return INTERIOR;
	}
	else if (mTestCase == FLAMINGO) {
		//z+ air
		//other walls
		int chk_level = 1;
		bool is_neumann = false;
		bool is_dirichlet = false;

		if (queryEffectiveBoundaryDirection1(acc, chk_level, info, l_ijk, boundary_axis, boundary_off)) {
			is_neumann = true;
			//if (boundary_axis == 0 && boundary_off > 0) is_dirichlet = true;
			//else is_neumann = true;
		}
		if (is_neumann) return NEUMANN;
		else if (is_dirichlet) return DIRICHLET;

		auto pos = acc.cellCenter(info, l_ijk);
		auto phi = FlamingoQuerySDF(pos, current_time);
		if (phi <= mSDFGridAccessor.solid_isovalue) return NEUMANN;
		//if (mMaskGridAccessor.GetMask0(model_pos)) return NEUMANN;
		return INTERIOR;
		}
	else {
		return DIRICHLET;
	}
}

__device__ void FluidParams::setInitialCondition(HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const Coord& l_ijk) const {
	double current_time = 0.0;

	if (mTestCase == LEAP_FROG) {
		double radius = 0.21;
		double x_gap = 0.625 * radius;
		double x_start = 0.16;
		double delta = 0.08 * radius;
		double gamma = radius * 0.1;
		int num_samples = 500;

		Vec unit_x(0, 0, -1);
		Vec unit_y(0, 1, 0);

		Vec velocity(0, 0, 0);
		T smoke = 0;


		auto pos = acc.cellCenter(info, l_ijk);

		addVortexRingInitialVelocityAndSmoke(pos, gamma, radius, delta, Vec(x_start, 0.5, 0.5), unit_x, unit_y, num_samples, velocity, smoke);
		addVortexRingInitialVelocityAndSmoke(pos, gamma, radius, delta, Vec(x_start + x_gap, 0.5, 0.5), unit_x, unit_y, num_samples, velocity, smoke);
		smoke = (smoke > 0.002f) ? 1.0 : 0.0;

		int boundary_axis, boundary_off;
		auto& tile = info.tile();
		tile.type(l_ijk) = cellType(current_time, acc, info, l_ijk, boundary_axis, boundary_off);
		for (int axis : {0, 1, 2}) {
			tile(Tile::u_channel + axis, l_ijk) = velocity[axis];
		}
		tile(Tile::dye_channel, l_ijk) = smoke;
	}
	else if (mTestCase == KARMAN) {
		Vec initial_vel = Vec(karman_source, 0, 0);

		Tile& tile = info.tile();
		for (int axis : {0, 1, 2}) {
			tile(Tile::u_channel + axis, l_ijk) = initial_vel[axis];
		}
		int boundary_axis, boundary_off;
		tile.type(l_ijk) = cellType(current_time, acc, info, l_ijk, boundary_axis, boundary_off);
		tile(Tile::dye_channel, l_ijk) = 0;
	}
	else if (mTestCase == SMOKESPHERE) {
		Vec initial_vel = smokesphere_source;
		//Vec initial_vel = Vec(0, 0, smokesphere_source);
		//Vec initial_vel = Vec(0, 0, 0);

		//Vec pos = acc.cellCenter(info, l_ijk);
		//pos[2] = smokesphere_center[2];
		//if ((pos - smokesphere_center).length() <= smokesphere_radius) {
		//	initial_vel = Vec(0, 0, smokesphere_source);
		//}


		Tile& tile = info.tile();
		for (int axis : {0, 1, 2}) {
			tile(Tile::u_channel + axis, l_ijk) = initial_vel[axis];
		}
		int boundary_axis, boundary_off;
		tile.type(l_ijk) = cellType(current_time, acc, info, l_ijk, boundary_axis, boundary_off);
		tile(Tile::dye_channel, l_ijk) = 0;
	}
	else if (mTestCase == NASA || mTestCase==WP3D) {
		Vec initial_vel = Vec(0, 0, nasa_source);
		Tile& tile = info.tile();
		for (int axis : {0, 1, 2}) {
			tile(Tile::u_channel + axis, l_ijk) = initial_vel[axis];
		}
		int boundary_axis, boundary_off;
		tile.type(l_ijk) = cellType(current_time, acc, info, l_ijk, boundary_axis, boundary_off);
		tile(Tile::dye_channel, l_ijk) = 0;
	}
	else if (mTestCase == PROP) {
		Vec initial_vel = Vec(0, 0, prop_source);
		Tile& tile = info.tile();
		for (int axis : {0, 1, 2}) {
			tile(Tile::u_channel + axis, l_ijk) = initial_vel[axis];
		}
		int boundary_axis, boundary_off;
		tile.type(l_ijk) = cellType(current_time, acc, info, l_ijk, boundary_axis, boundary_off);
		tile(Tile::dye_channel, l_ijk) = 0;
	}
	else if (mTestCase == F1CAR) {
		Vec initial_vel = Vec(0, 0, f1_source);
		Tile& tile = info.tile();
		for (int axis : {0, 1, 2}) {
			tile(Tile::u_channel + axis, l_ijk) = initial_vel[axis];
		}
		int boundary_axis, boundary_off;
		tile.type(l_ijk) = cellType(current_time, acc, info, l_ijk, boundary_axis, boundary_off);
		tile(Tile::dye_channel, l_ijk) = 0;
	}
	else if (mTestCase == LIZARD) {
		Vec initial_vel = Vec(lizard_source, 0, 0);
		//Vec initial_vel(0, lizard_source, 0);
		Tile& tile = info.tile();
		for (int axis : {0, 1, 2}) {
			tile(Tile::u_channel + axis, l_ijk) = initial_vel[axis];
		}
		int boundary_axis, boundary_off;
		tile.type(l_ijk) = cellType(current_time, acc, info, l_ijk, boundary_axis, boundary_off);
		tile(Tile::dye_channel, l_ijk) = 0;
	}
	else if (mTestCase == FISH) {
		Vec initial_vel = Vec(fish_source, 0, 0);
		Tile& tile = info.tile();
		for (int axis : {0, 1, 2}) {
			tile(Tile::u_channel + axis, l_ijk) = initial_vel[axis];
		}
		int boundary_axis, boundary_off;
		tile.type(l_ijk) = cellType(current_time, acc, info, l_ijk, boundary_axis, boundary_off);
		tile(Tile::dye_channel, l_ijk) = 0;
	}
	else if (mTestCase == BAT) {
		Vec initial_vel = Vec(bat_source, 0, 0);
		Tile& tile = info.tile();
		for (int axis : {0, 1, 2}) {
			tile(Tile::u_channel + axis, l_ijk) = initial_vel[axis];
		}
		int boundary_axis, boundary_off;
		tile.type(l_ijk) = cellType(current_time, acc, info, l_ijk, boundary_axis, boundary_off);
		tile(Tile::dye_channel, l_ijk) = 0;
	}
	else if (mTestCase == FLAMINGO) {
		Vec initial_vel = Vec(flamingo_source, 0, 0);
		Tile& tile = info.tile();
		for (int axis : {0, 1, 2}) {
			tile(Tile::u_channel + axis, l_ijk) = initial_vel[axis];
		}
		int boundary_axis, boundary_off;
		tile.type(l_ijk) = cellType(current_time, acc, info, l_ijk, boundary_axis, boundary_off);
		tile(Tile::dye_channel, l_ijk) = 0;
	}
	else {
		Tile& tile = info.tile();
		for (int axis : {0, 1, 2}) {
			tile(Tile::u_channel + axis, l_ijk) = 0;
		}
		int boundary_axis, boundary_off;
		tile.type(l_ijk) = cellType(current_time, acc, info, l_ijk, boundary_axis, boundary_off);
		tile(Tile::dye_channel, l_ijk) = 0;
	}
}
