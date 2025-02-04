#include "FlowMapTests.h"
#include "PoissonSolver.h"
#include "PoissonIOFunc.h"
#include "FMParticles.h"
#include "Common.h"
#include "GPUTimer.h"
#include "Random.h"
#include <polyscope/polyscope.h>
#include <polyscope/point_cloud.h>
#include <polyscope/curve_network.h>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

namespace FlowMapTests {

    //case0: regular 128^3 lattice
    //case1: fine on left and coarse on right
    //case2: two sources
    __device__ int FlowMapTestsLevelTarget(const HATileAccessor<Tile>& acc, HATileInfo<Tile>& info, const int grid_case) {
        if (grid_case == 0) {
            //128^3
            return 4;
        }
        else if (grid_case == 1) {
            auto bbox = acc.tileBBox(info);
            int desired_level = 0;
            if (bbox.min()[0] <= 0.25) return 4;//slow converging, if 0.25 not converging
            else return 3;
        }
        else if (grid_case == 2) {
            int desired_level = 0;
            auto bbox = acc.tileBBox(info);
            const Vec pointSrc1(0.51, 0.49, 0.54);
            const Vec pointSrc2(0.93, 0.08, 0.91);
            if (bbox.isInside(pointSrc2)) desired_level = 6;
            if (bbox.isInside(pointSrc1)) desired_level = 7;
            //if (bbox.isInside(pointSrc2)) desired_level = 3;
            //if (bbox.isInside(pointSrc1)) desired_level = 4;
            return desired_level;
        }
        else if (grid_case == 3) {
            //refine at (0.35,0.35,0.35)
            //it's for testing the 3D deformation
            int desired_level = 0;
            auto bbox = acc.tileBBox(info);
            const Vec pointSrc1(0.35, 0.35, 0.35);
            const Vec pointSrc2(0.8, 0.2, 0.6);
            if (bbox.isInside(pointSrc2)) desired_level = 5;
            if (bbox.isInside(pointSrc1)) desired_level = 6;
            //if (bbox.isInside(pointSrc2)) desired_level = 3;
            //if (bbox.isInside(pointSrc1)) desired_level = 4;
            return desired_level;
        }
        else if (grid_case == 4) {
            //8^3
            //to test most basic case
            return 3;
        }
        else if (grid_case == 5) {
            //bottom part denser
            auto bbox = acc.tileBBox(info);
            if (bbox.min()[1] <= 0.25) return 4;//slow converging, if 0.25 not converging
            //if (bbox.max()[1] >= 0.75) return 3;//slow converging, if 0.25 not converging
            else return 2;
        }
        else if (grid_case == 6) {
            auto bbox = acc.tileBBox(info);
            if (bbox.min()[1] <= 0.5 && 0.5 <= bbox.max()[1]) return 6;
            else return 2;
        }
        else if (grid_case == 7) {
            //try to test nfm advection with 3d deformation

            int desired_level = 0;
            auto bbox = acc.tileBBox(info);
            double eps = 1e-6;
            const Vec pointSrc1(0.5 - eps, 0.5 - eps, 0.5 - eps);
            if (bbox.isInside(pointSrc1)) desired_level = 6;

            return desired_level;
        }
        else if (grid_case == 8) {
            return 3;
        }
        else if (grid_case == 9) {
            return 5;
        }
    }

    class Deformation3D {
    public:
        //3D deformation test from: Unstructured un-split geometrical Volume-of-Fluid methods – A review
        double T0 = 3;
        double pi = CommonConstants::pi;
        __hostdev__ Vec operator()(const Vec& pos, const double t) const {
            double x = pos[0], y = pos[1], z = pos[2];
            double u = 2 * sin(pi * x) * sin(pi * x) * sin(2 * pi * y) * sin(2 * pi * z) * cos(pi * t / T0);
            double v = -sin(2 * pi * x) * sin(pi * y) * sin(pi * y) * sin(2 * pi * z) * cos(pi * t / T0);
            double w = -sin(2 * pi * x) * sin(2 * pi * y) * sin(pi * z) * sin(pi * z) * cos(pi * t / T0);
            return Vec(u, v, w);
        };
        __hostdev__ Eigen::Matrix3<T> gradu(const Vec& pos, const double t)const {
            double x = pos[0], y = pos[1], z = pos[2];
            double cos_pi_t_T0 = cos(pi * t / T0);
            double sin_pi_t_T0 = sin(pi * t / T0);

            Eigen::Matrix3<T> jacobian;

            // Partial derivatives for u with respect to x, y, z
            jacobian(0, 0) = 4 * pi * sin(pi * x) * cos(pi * x) * sin(2 * pi * y) * sin(2 * pi * z) * cos_pi_t_T0;
            jacobian(0, 1) = 4 * pi * sin(pi * x) * sin(pi * x) * cos(2 * pi * y) * sin(2 * pi * z) * cos_pi_t_T0;
            jacobian(0, 2) = 4 * pi * sin(pi * x) * sin(pi * x) * sin(2 * pi * y) * cos(2 * pi * z) * cos_pi_t_T0;

            // Partial derivatives for v with respect to x, y, z
            jacobian(1, 0) = -2 * pi * cos(2 * pi * x) * sin(pi * y) * sin(pi * y) * sin(2 * pi * z) * cos_pi_t_T0;
            jacobian(1, 1) = -2 * pi * sin(2 * pi * x) * sin(pi * y) * cos(pi * y) * sin(2 * pi * z) * cos_pi_t_T0;
            jacobian(1, 2) = -2 * pi * sin(2 * pi * x) * sin(pi * y) * sin(pi * y) * cos(2 * pi * z) * cos_pi_t_T0;

            // Partial derivatives for w with respect to x, y, z
            jacobian(2, 0) = -2 * pi * cos(2 * pi * x) * sin(2 * pi * y) * sin(pi * z) * sin(pi * z) * cos_pi_t_T0;
            jacobian(2, 1) = -4 * pi * sin(2 * pi * x) * cos(2 * pi * y) * sin(pi * z) * sin(pi * z) * cos_pi_t_T0;
            jacobian(2, 2) = -2 * pi * sin(2 * pi * x) * sin(2 * pi * y) * cos(pi * z) * sin(pi * z) * cos_pi_t_T0;

            return jacobian;
        }
        __hostdev__ void velocityAndJacobian(const Vec& pos, const double t, Vec& vel, Eigen::Matrix3<T>& jacobian) const {
            vel = operator()(pos, t);
            jacobian = gradu(pos, t);
        }

        thrust::host_vector<Vec> samplePoints(const int vn = 100000)const {
            Vec center(0.35, 0.35, 0.35); double radius = 0.15;
            RandomGenerator rng;
            thrust::host_vector<Vec> verts_h;
            for (int i = 0; i < vn; ++i) {
                double theta = rng.uniform(0, 2 * pi); // azimuthal angle in [0, 2*pi]
                double phi = rng.uniform(0, pi);     // polar angle in [0, pi]

                double x = radius * sin(phi) * cos(theta);
                double y = radius * sin(phi) * sin(theta);
                double z = radius * cos(phi);

                auto point = center + Vec(x, y, z);
                verts_h.push_back(point);
            }

            return verts_h;
        }

        thrust::host_vector<Particle> sampleParticles(const T time, const int vn = 100000) const {
            RandomGenerator rng;


            thrust::host_vector<Particle> particles;
            Vec center(0.35, 0.35, 0.35); double radius = 0.15;
            for (int i = 0; i < vn; i++) {
                double theta = rng.uniform(0, 2 * pi); // azimuthal angle in [0, 2*pi]
                double phi = rng.uniform(0, pi);     // polar angle in [0, pi]

                double x = radius * sin(phi) * cos(theta);
                double y = radius * sin(phi) * sin(theta);
                double z = radius * cos(phi);

                auto pos = center + Vec(x, y, z);

                Particle p;
                p.start_time = time;
                p.impulse = operator()(pos, time);
                p.pos = pos;
                p.gradm = Eigen::Matrix3<T>::Zero();
                p.matT = Eigen::Matrix3<T>::Identity();
                particles.push_back(p);
            }
            return particles;
        }
    };

    class Deformation2D {
    public:
        double T0 = 8;
        double pi = CommonConstants::pi;
        __hostdev__ Vec operator()(const Vec& pos, const double t)const {
            //return Vec(1, 0, 0);
            double t_over_T = t / T0;
            double x = pos[0], y = pos[1];
            double u = -2 * cos(pi * t_over_T) * cos(pi * y) * pow(sin(pi * x), 2) * sin(pi * y);
            double v = 2 * cos(pi * t_over_T) * cos(pi * x) * sin(pi * x) * pow(sin(pi * y), 2);
            return Vec(u, v, 0);
        }

        static Vec UnitVector(const double rad) {
            return Vec(cos(rad), sin(rad), 0);
        }

        thrust::host_vector<Vec> samplePoints(const int vn = 100000) const {
            Vec center(0.5, 0.75, 0.5); double radius = 0.15;
            thrust::host_vector<Vec> verts_h;
            auto push = [&](thrust::host_vector<Vec>& v, const Vec& p) {v.push_back(p); };
            for (int i = 0; i < vn; i++) push(verts_h, center + UnitVector(2 * pi / vn * i) * radius);
            return verts_h;
        }

        thrust::host_vector<Particle> sampleParticles(const T time, const int vn = 100000) const {
            thrust::host_vector<Particle> particles;

            Vec center(0.5, 0.75, 0.5); double radius = 0.15;
            for (int i = 0; i < vn; i++) {
                Vec pos = center + UnitVector(2 * pi / vn * i) * radius;
                Particle p;
                p.start_time = time;
                p.impulse = (*this)(pos, time);
                p.pos = pos;
                p.gradm = Eigen::Matrix3<T>::Zero();
                p.matT = Eigen::Matrix3<T>::Identity();
                particles.push_back(p);
            }
            return particles;
        }
    };



    void TestParticleAdvectionWithNFMInterpolation(const int grid_case) {
        double pi = CommonConstants::pi;

        uint32_t scale = 8;
        float h = 1.0 / scale;

        //0:8, 1:16, 2:32, 3:64, 4:128, 5:256, 6:512, 7:1024
        HADeviceGrid<Tile> grid(h, { 16,16,16,16,16,16,16,16,16,16 });

        grid.setTileHost(0, nanovdb::Coord(0, 0, 0), Tile(), LEAF);
        grid.compressHost();
        grid.syncHostAndDevice();
        SpawnGhostTiles(grid);

        IterativeRefine(grid, [=]__device__(const HATileAccessor<Tile>&acc, HATileInfo<Tile>&info) { return FlowMapTestsLevelTarget(acc, info, grid_case); });
        grid.launchVoxelFunc(
            [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
            auto& tile = info.tile();
            tile.type(l_ijk) = INTERIOR;
        }, -1, LEAF, LAUNCH_SUBTREE
        );
        CalcCellTypesFromLeafs(grid);

        //we use the 3D deformation test from: Unstructured un-split geometrical Volume-of-Fluid methods – A review

        Deformation3D vel_func;
        auto particles_h = vel_func.sampleParticles(0.0);//time 0

        auto base_dir = fs::current_path() / "data" / fmt::format("particles_nfm_advection_test{}", grid_case);
        fs::create_directories(base_dir);
        auto holder = grid.getHostTileHolderForLeafs();
        IOFunc::OutputTilesAsVTU(holder, base_dir / "tiles.vtu");

        int u_channel = Tile::u_channel;
        int node_u_channel = 0;

        double cfl = 2.0;//max vel=1
        double dt = h / (1 << grid.mMaxLevel) * cfl;
        int N = ceil(vel_func.T0 / dt);
        double t = 0;
        thrust::device_vector<Particle> particles_d = particles_h;
        for (int i = 0; i <= N; i++) {
            Info("frame {}/{}", i, N);
            grid.launchVoxelFuncOnAllTiles(
                [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
                auto& tile = info.tile();
                for (int axis = 0; axis < 3; axis++) {
                    auto face_ctr = acc.faceCenter(axis, info, l_ijk);
                    tile(u_channel + axis, l_ijk) = vel_func(face_ctr, t)[axis];
                }
            }, LEAF, 4);

            InterpolateVelocitiesAtAllTiles(grid, u_channel, node_u_channel);

            auto acc = grid.deviceAccessor();
            auto particles_d_ptr = thrust::raw_pointer_cast(particles_d.data());
            int fine_level = grid.mMaxLevel;
            int coarse_level = 0;

            LaunchIndexFunc([=]__device__(int idx) {
                auto& p = particles_d_ptr[idx];
                RK4ForwardPositionAndT(acc, fine_level, coarse_level, dt, u_channel, node_u_channel, p.pos, p.matT);

            }, particles_d.size());

            //IOFunc::Output
            auto particles_h_ptr = std::make_shared<thrust::host_vector<Particle>>(particles_d);
            IOFunc::OutputParticleSystemAsVTU(particles_h_ptr, base_dir / fmt::format("particles_{:04d}.vtu", i));


            t += dt;

        }
    }

    double LinfErrorBetweenPointCloud(const thrust::host_vector<Vec>& pc1, const thrust::host_vector<Vec>& pc2) {
        Assert(pc1.size() == pc2.size(), "Two point clouds should have the same size.");

        double max_error = -1;
        int max_idx = -1;

        for (int i = 0; i < pc1.size(); ++i) {
            double error = (pc1[i] - pc2[i]).length(); // Compute distance (norm)
            if (error > max_error || max_idx == -1) {
                max_error = error;
                max_idx = i;
            }
        }

        Info("Max error at index {} with value {}, pc1 {} pc2 {}", max_idx, max_error, pc1[max_idx], pc2[max_idx]);

        return max_error;
    }

    double LinfErrorBetweenTargetMatrixForbenius2(const thrust::host_vector<Eigen::Matrix3<T>>& mats, const Eigen::Matrix3<T>& target) {
        double max_error = 0.0;
        int max_idx = -1;

        for (size_t i = 0; i < mats.size(); ++i) {
            double error = (mats[i] - target).norm(); // Frobenius norm in Eigen
            if (error > max_error) {
                max_error = error;
                max_idx = i;
            }
        }
        //Info("Max error at index {} with value {}", max_idx, max_error);

        return max_error;
    }

    std::vector<double> CalcLinfErrorBetweenMatrixForbenius2(const thrust::host_vector<Eigen::Matrix3<T>>& mats1, const thrust::host_vector<Eigen::Matrix3<T>>& mats2) {
        Assert(mats1.size() == mats2.size(), "Two matrix arrays should have the same size.");
		std::vector<double> errors(mats1.size(), 0.0);

        for (size_t i = 0; i < mats1.size(); ++i) {
            double err = (mats1[i] - mats2[i]).norm(); // Frobenius norm in Eigen
			errors[i] = err;
        }

        return errors;
    }

    double LinfErrorBetweenMatrixForbenius2(const thrust::host_vector<Eigen::Matrix3<T>>& mats1, const thrust::host_vector<Eigen::Matrix3<T>>& mats2) {
        Assert(mats1.size() == mats2.size(), "Two matrix arrays should have the same size.");
        double max_error = 0.0;
        int max_idx = -1;

        for (size_t i = 0; i < mats1.size(); ++i) {
            double error = (mats1[i] - mats2[i]).norm(); // Frobenius norm in Eigen
            if (error > max_error) {
                max_error = error;
                max_idx = i;
            }
        }

        Info("Max error at index {} with value {}", max_idx, max_error);

        return max_error;
    }


    template<class FuncVT>
    __hostdev__ Vec RK4ForwardAdvectWithAnalyticalVelocity(FuncVT vel_func, const Vec& pos, const double time, const double dt) {
        double c1 = 1.0 / 6.0 * dt, c2 = 1.0 / 3.0 * dt, c3 = 1.0 / 3.0 * dt, c4 = 1.0 / 6.0 * dt;
        Vec vel1 = vel_func(pos, time);
        Vec pos1 = pos + vel1 * 0.5 * dt;
        Vec vel2 = vel_func(pos1, time + 0.5 * dt);
        Vec pos2 = pos + vel2 * 0.5 * dt;
        Vec vel3 = vel_func(pos2, time + 0.5 * dt);
        Vec pos3 = pos + vel3 * dt;
        Vec vel4 = vel_func(pos3, time + dt);
        return pos + c1 * vel1 + c2 * vel2 + c3 * vel3 + c4 * vel4;
    }

    void TestFlowMapAdvection(const int grid_case) {
        Info("Test flow map advection with grid case {}", grid_case);
        int test_flowmap_stride = 5; // x dt

        std::vector<std::shared_ptr<HADeviceGrid<Tile>>> grid_ptrs;
        std::vector<double> time_steps;

        float h = 1.0 / Tile::DIM;
        for (int i = 0; i <= test_flowmap_stride; i++) {
            //0:8, 1:16, 2:32, 3:64, 4:128, 5:256, 6:512, 7:1024
            auto ptr = std::make_shared<HADeviceGrid<Tile>>(h, std::initializer_list<uint32_t>{ 16, 16, 16, 16, 16, 16, 16, 16, 16, 16 });
            grid_ptrs.push_back(ptr);

            auto& grid = *ptr;
            grid.setTileHost(0, nanovdb::Coord(0, 0, 0), Tile(), LEAF);
            grid.compressHost(false);
            grid.syncHostAndDevice();
            SpawnGhostTiles(grid, false);

            IterativeRefine(grid, [=]__device__(const HATileAccessor<Tile>&acc, HATileInfo<Tile>&info) { return FlowMapTestsLevelTarget(acc, info, grid_case); }, false);
            grid.launchVoxelFunc(
                [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
                auto& tile = info.tile();
                tile.type(l_ijk) = INTERIOR;
            }, -1, LEAF, LAUNCH_SUBTREE
            );
            CalcCellTypesFromLeafs(grid);
        }
        thrust::host_vector<HATileAccessor<Tile>> accs_h;
        for (int i = 0; i <= test_flowmap_stride; i++) {
            accs_h.push_back(grid_ptrs[i]->deviceAccessor());
        }
        thrust::device_vector<HATileAccessor<Tile>> accs_d = accs_h;
        auto accs_d_ptr = thrust::raw_pointer_cast(accs_d.data());

        Deformation3D vel_func;


        auto base_dir = fs::current_path() / "data" / fmt::format("flowmap_advection_test{}", grid_case);
        fs::create_directories(base_dir);


        double cfl = 1.0;//max vel=1
        double dt = 1.0 / 1024 * cfl;
        //double dt = h / (1 << (grid_ptrs[0]->mMaxLevel)) * cfl;
        //double dt = h / (1 << 4) * cfl;//assume that finest level is 4
        int u_channel = Tile::u_channel;
        int node_u_channel = 0;

        //fill velocity fields
        for (int i = 0; i <= test_flowmap_stride; i++) {
            double time = i * dt;

            auto& grid = *grid_ptrs[i];
            grid.launchVoxelFunc(
                [=] __device__(HATileAccessor<Tile>&acc, HATileInfo<Tile>&info, const Coord & l_ijk) {
                auto& tile = info.tile();
                for (int axis = 0; axis < 3; axis++) {
                    auto face_ctr = acc.faceCenter(axis, info, l_ijk);
                    tile(u_channel + axis, l_ijk) = vel_func(face_ctr, time)[axis];
                }
            }, -1, LEAF, LAUNCH_SUBTREE
            );
			InterpolateVelocitiesAtAllTiles(grid, u_channel, node_u_channel);

            time_steps.push_back(dt);
        }

        auto rk4_phi_analytical = [=]__hostdev__(Vec pos, int step_begin, int step_end) {
            for (int i = step_begin; i < step_end; i++) {
                pos = RK4ForwardAdvectWithAnalyticalVelocity(vel_func, pos, i * dt, dt);
            }
            return pos;
        };
        auto rk4_psi_analytical = [=]__hostdev__(Vec pos, int step_end, int step_begin) {
            for (int i = step_end; i > step_begin; i--) {
                pos = RK4ForwardAdvectWithAnalyticalVelocity(vel_func, pos, i * dt, -dt);
            }
            return pos;
        };


        RandomGenerator rng;
        //thrust::host_vector<Vec> points(100000);
        thrust::host_vector<Vec> points(1*1024*1024);
        double lo = 1.0 / 16, hi = 1 - lo;
        //double lo = 1.0 / 8, hi = 1 - lo;
        for (auto& p : points) {
            p = Vec(rng.uniform(lo, hi), rng.uniform(lo, hi), rng.uniform(lo, hi));
        }
        //points[0] = Vec(0.37287432, 0.2518963, 0.7224257);
        //points.resize(1);

        //points[0] = points[867923];
        //points[0] = Vec(0.4559023, 0.2541084, 0.74300975);
        //points.resize(1);

        thrust::device_vector<Vec> points_d = points;


        {
            //test if F@T=I
            thrust::host_vector<Vec> psi_analytical(points.size());
            thrust::host_vector<Vec> phi_of_psi_analytical(points.size());
            tbb::parallel_for(0, static_cast<int>(points.size()), [&](int i) {
                psi_analytical[i] = rk4_psi_analytical(points[i], test_flowmap_stride, 0);
                phi_of_psi_analytical[i] = rk4_phi_analytical(psi_analytical[i], 0, test_flowmap_stride);
                }
            );

            thrust::device_vector<Vec> psi_flowmap_d = points_d;
            thrust::device_vector<Vec> phi_of_psi_flowmap_d = points_d;

            //F_forward is F
            //T_forward is T
            //F_back should be T
            //T_back should be F
            thrust::device_vector<Eigen::Matrix3<T>> F_flowmap_back_d(points.size());
            thrust::device_vector<Eigen::Matrix3<T>> T_flowmap_forward_d(points.size());

            int fine_level = grid_ptrs[0]->mMaxLevel;
            int coarse_level = 0;
            auto points_d_ptr = thrust::raw_pointer_cast(points_d.data());
			auto psi_flowmap_d_ptr = thrust::raw_pointer_cast(psi_flowmap_d.data());
            auto phi_of_psi_flowmap_d_ptr = thrust::raw_pointer_cast(phi_of_psi_flowmap_d.data());
            auto F_flowmap_back_d_ptr = thrust::raw_pointer_cast(F_flowmap_back_d.data());
            auto T_flowmap_forward_d_ptr = thrust::raw_pointer_cast(T_flowmap_forward_d.data());
            LaunchIndexFunc([=]__device__(int i) {
                Vec psi = points_d_ptr[i]; Eigen::Matrix3<T> F_back = Eigen::Matrix3<T>::Identity();

                HATileInfo<Tile> info; Coord l_ijk; Vec frac;
                accs_d_ptr[test_flowmap_stride - 1].findLeafVoxelAndFrac(psi, info, l_ijk, frac);
                if (info.empty()) return;
                int reference_level = info.mLevel;

                for (int i = test_flowmap_stride - 1; i >= 0; i--) {
                    RK4ForwardPositionAndF(accs_d_ptr[i], fine_level, coarse_level, -dt, u_channel, node_u_channel, psi, F_back);

                }
                F_flowmap_back_d_ptr[i] = F_back;
				psi_flowmap_d_ptr[i] = psi;

                Vec phi1 = psi; Eigen::Matrix3<T> T_forward = Eigen::Matrix3<T>::Identity();
                for (int i = 0; i < test_flowmap_stride; i++) {
                    RK4ForwardPositionAndT(accs_d_ptr[i], fine_level, coarse_level, dt, u_channel, node_u_channel, phi1, T_forward);
                }
                phi_of_psi_flowmap_d_ptr[i] = phi1;
                T_flowmap_forward_d_ptr[i] = T_forward;

            }, points.size(), 128
                );


            Info("Test flowmap advection on {} points", points.size());
            Info("Linf error between analytical phi(psi) and original points: {:.3e}", LinfErrorBetweenPointCloud(phi_of_psi_analytical, points));
            Info("Linf error between flowmap advected phi(psi) and original points: {:.3e}", LinfErrorBetweenPointCloud(phi_of_psi_flowmap_d, points));
            Info("Linf error between T_forward and F_back: {:.3e}", LinfErrorBetweenMatrixForbenius2(T_flowmap_forward_d, F_flowmap_back_d));
            fmt::print("\n");

            //{
            //    // 将 psi_flowmap_d 从设备复制到主机
            //    thrust::host_vector<Vec> psi_flowmap_h = psi_flowmap_d;

            //    // 创建线段的顶点和索引
            //    std::vector<std::array<size_t, 2>> edges; // 每条线段由两个点的索引组成

            //    for (size_t i = 0; i < points.size(); ++i) {
            //        edges.push_back({ i, points.size() + i }); // 连接 points[i] 和 psi_flowmap_h[i]
            //    }

            //    // 合并起点和终点为一个顶点集合
            //    thrust::host_vector<Vec> all_points = points;
            //    all_points.insert(all_points.end(), psi_flowmap_h.begin(), psi_flowmap_h.end());

            //    // 初始化 Polyscope
            //    polyscope::init();

            //    std::vector<double> errs = CalcLinfErrorBetweenMatrixForbenius2(T_flowmap_forward_d, F_flowmap_back_d);
            //    // 可视化起点
            //    auto pc = polyscope::registerPointCloud("points", points);
            //    pc->addScalarQuantity("error", errs);

            //    // 注册运动轨迹
            //    auto cn = polyscope::registerCurveNetwork("motion trajectories", all_points, edges);

            //    IOFunc::AddTilesToPolyscopeVolumetricMesh(*grid_ptrs[0], LEAF, "leaf tiles");

            //    // 显示 Polyscope
            //    polyscope::show();
            //}

			//std::vector<double> errs = CalcLinfErrorBetweenMatrixForbenius2(T_flowmap_forward_d, F_flowmap_back_d);
   //         polyscope::init();
   //         IOFunc::AddTilesToPolyscopeVolumetricMesh(*grid_ptrs[0], LEAF, "leaf tiles");

			//auto pc = polyscope::registerPointCloud("points", points);
			//pc->addScalarQuantity("error", errs);
   //         polyscope::show();
        }
    }
    
}