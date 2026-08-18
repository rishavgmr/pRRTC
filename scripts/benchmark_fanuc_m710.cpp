// Dev-only benchmark driver for pRRTC against our own Collins P2P-benchmark
// scene, instead of MotionBenchMaker, for direct before/after comparison against
// trajectory_planner's benchmark_p2p.cpp (RSW-2740). Mirrors
// scripts/single_mbm.cpp's structure (env construction, settings, the
// pRRTC::solve<Robot> call) but sources the problem from our own files.
//
// Environment collision-checking uses direct SDF lookups (RSW-2740's SDF-integration phase)
// rather than pRRTC's original brute-force cuboid list - see fanuc_m710_benchmark.cuh and
// src/collision/sdf_environment.hh for the device-side half, and pRRTC_benchmark.hh's
// SDFGridHost/uploadSDFEnvironment for the host-side upload this file drives. The raw .bin
// files loaded below are the exact same ones PhysxNode::loadSDFArray/loadFloatSDFArray read at
// runtime - verified to apply no downsampling at load time, so no conversion step is needed
// here either.
//
// After solve() finds a path, runs the same kind of post-solve random-shortcut
// refinement P2P::postProcessing() does (trajectory_planner/src/p2p.cpp), via
// pRRTC::shortcutPath() (src/planning/pRRTC_benchmark.cu) - so the wall time we
// report pays for the same work benchmark_p2p.cpp's timed solvePath() call does,
// not just the raw RRT-Connect search.
//
// Not part of any test suite - build via the benchmark_fanuc_m710 CMake
// target and run by hand, same spirit as benchmark_p2p.cpp.

#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <numeric>
#include <algorithm>
#include <utility>
#include <pwd.h>
#include <unistd.h>

#include "src/collision/environment.hh"
#include "src/planning/Planners.hh"
#include "src/planning/pRRTC_settings.hh"
#include "src/planning/pRRTC_benchmark.hh"

using json = nlohmann::json;
using namespace ppln::collision;

namespace {

constexpr int kNumRuns = 10;

// Matches P2P's params.m_refinement.max_path_refinement_attempts (see
// trajectory_planner's DefaultP2PParams()) - same refinement budget on both sides.
constexpr int kShortcutMaxAttempts = 200;

// Fanucm710's approximate joint count (fanucm710_approx_joint_types[8] in
// fanuc_m710_benchmark.cuh) - can't reference that macro directly since this file is compiled
// as plain C++, not CUDA, and that header is full of __device__/__constant__ code a non-CUDA
// compiler can't parse. Kept in sync by convention, not by the compiler - if the robot's joint
// count ever changes, update both.
constexpr int kFanucm710JointCount = 8;

// Minimum clearance offsets, matching benchmark_p2p.cpp's own CollisionOffsets (test/
// benchmark_p2p.cpp:82-84) exactly - these are the "real Collins deployment values" that
// benchmark hardcodes for this same fixed comparison problem (NOT default_params.yaml's
// collision_offsets, which that file's own comment misattributes these to but which actually
// holds different, smaller values - benchmark_p2p.cpp's literal constants are what the P2P
// side of this comparison actually runs with, so that's what we match here). Workpiece and
// environment offsets get pre-scaled by *100.0 below, same as
// CollisionManager::updateCustomCollisionOffsets does; self-collision stays in meters, matching
// that function leaving m_self_collision_offset_ unscaled.
constexpr float kWorkpieceCollisionOffsetM = 0.04f;
constexpr float kEnvironmentCollisionOffsetM = 0.025f;
constexpr float kSelfCollisionOffsetM = 0.025f;

// Our fanuc_m710 fixture data lives in the platform repo, not pRRTC's own
// resources/ - same absolute-path convention used for the fkcc_gen configs.
const std::string kFixtureDir =
    "/workspaces/platform/src/planners/trajectory_planner/test/data/p2p_benchmark";

// Mirror of gmr_paths::runtimeDataPath() (C++) / the Python mirror in
// playback_p2p_benchmark_path.py: $HOME/.ss_temp[/subdir]. Shared with
// benchmark_p2p.cpp's output location on purpose, so both algorithms' results
// sit side by side for comparison.
std::string runtime_data_path(const std::string& subdir) {
    const char* home_env = std::getenv("HOME");
    std::string home;
    if (home_env) {
        home = home_env;
    } else if (struct passwd* pw = getpwuid(getuid())) {
        home = pw->pw_dir;
    } else {
        home = "/var/tmp/ss_runtime";
    }
    std::string path = home + "/.ss_temp/" + subdir;
    std::system(("mkdir -p " + path).c_str());
    return path;
}

// Reads one reduced SDF .bin file verbatim - same header layout and dtype auto-detection as
// voxel_sdf_to_cuboids.py's read_sdf_grid (numX/numY/numZ as u32, then boundsLower/boundsUpper
// as 3 floats each, then the payload; int16_t if payload size matches numX*numY*numZ*2, float if
// it matches *4). No resampling: PhysxNode::loadSDFArray/loadFloatSDFArray don't do any either,
// so loading verbatim is what keeps this comparable to the real system's own queries.
pRRTC::SDFGridHost load_sdf_grid_host(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("failed to open " + path);
    }

    uint32_t num_x, num_y, num_z;
    f.read(reinterpret_cast<char*>(&num_x), sizeof(uint32_t));
    f.read(reinterpret_cast<char*>(&num_y), sizeof(uint32_t));
    f.read(reinterpret_cast<char*>(&num_z), sizeof(uint32_t));

    float bounds_lower[3], bounds_upper[3];
    f.read(reinterpret_cast<char*>(bounds_lower), sizeof(float) * 3);
    f.read(reinterpret_cast<char*>(bounds_upper), sizeof(float) * 3);

    if (!f) {
        throw std::runtime_error(path + ": failed reading header");
    }

    std::vector<uint8_t> payload((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    const std::size_t n_voxels = static_cast<std::size_t>(num_x) * num_y * num_z;
    bool is_float;
    if (payload.size() == n_voxels * 2) {
        is_float = false;
    } else if (payload.size() == n_voxels * 4) {
        is_float = true;
    } else {
        throw std::runtime_error(path + ": payload is " + std::to_string(payload.size()) +
                                  " bytes, matches neither int16 (" + std::to_string(n_voxels * 2) +
                                  ") nor float32 (" + std::to_string(n_voxels * 4) + ") for " +
                                  std::to_string(n_voxels) + " voxels");
    }

    pRRTC::SDFGridHost grid;
    grid.raw_bytes = std::move(payload);
    grid.is_float = is_float;
    grid.numX = num_x;
    grid.numY = num_y;
    grid.numZ = num_z;
    for (int i = 0; i < 3; i++) {
        grid.boundsLower[i] = bounds_lower[i];
        grid.boundsUpper[i] = bounds_upper[i];
    }
    // Set by the caller (load_sdf_environment()), which knows which object this is -
    // this loader only knows how to parse the .bin file, not what margin the object needs.
    grid.offset_scaled = 0.0f;
    return grid;
}

// Loads the three environment objects (same raw .bin files the real system queries against - see
// scene_config_m710.json's "env" list) into host memory and builds the per-link collision mask.
// Object order here must match the mask's column order.
//
// Deliberately returns the loaded data rather than uploading it directly: pRRTC::solve() calls
// cudaDeviceReset() at the end of every single call (inherited, upstream behavior, present in
// the original pRRTC.cu too - not something this SDF work introduced), which destroys the whole
// CUDA context and wipes every __constant__ global, including whatever uploadSDFEnvironment()
// last wrote. Verified directly: a __constant__ int written via cudaMemcpyToSymbol reads back
// as 0 after cudaDeviceReset(), with no other change. That means a single upload before the run
// loop only actually protects run 0's solve() phase - run 0's own shortcutPath() call (which
// happens after solve() has already reset the context) and every run from 1 onward see
// fanucm710_num_sdf_grids == 0, silently disabling all environment/workpiece collision checking
// (self-collision, which doesn't depend on this state, still works) - exactly the "robot passes
// straight through the workpiece" symptom this was tracked down from. The fix is to re-upload
// from these already-parsed host buffers before every run, not to re-read the .bin files each
// time (main() calls uploadSDFEnvironment(grids, link_env_mask) once per iteration, right before
// starting that iteration's timer, so the fast re-upload from memory doesn't get counted as
// planning time either).
std::pair<std::vector<pRRTC::SDFGridHost>, std::vector<std::vector<bool>>> load_sdf_environment() {
    const std::vector<std::pair<std::string, std::string>> objects = {
        {"Unified_collision_model", kFixtureDir + "/env_meshes/3l_env_35_collins.bin"},
        {"rail_box", kFixtureDir + "/env_meshes/rail_box_35.bin"},
        {"workpiece", kFixtureDir + "/workpiece.bin"},
    };

    std::vector<pRRTC::SDFGridHost> grids;
    for (const auto& object : objects) {
        auto grid = load_sdf_grid_host(object.second);
        // Workpiece gets its own (larger) margin; Unified_collision_model and rail_box both
        // draw from the single shared environment margin - matches envPhysXCollisionQuery/
        // envPhysXDistanceQuery applying m_env_collision_offset_ uniformly across every object
        // in the env loop, with no per-object distinction.
        grid.offset_scaled = (object.first == "workpiece")
            ? kWorkpieceCollisionOffsetM * 100.0f
            : kEnvironmentCollisionOffsetM * 100.0f;
        std::cout << object.first << ": " << grid.numX << "x" << grid.numY << "x" << grid.numZ << " voxels, "
                   << (grid.is_float ? "float" : "int16") << ", " << grid.raw_bytes.size() / (1024 * 1024)
                   << " MB, offset=" << grid.offset_scaled << " (scaled)\n";
        grids.push_back(std::move(grid));
    }

    // Transcribed from scene_config_m710.json's collisionMask (one entry today: base_link is
    // exempt from Unified_collision_model and rail_box) - the per-link exemption pRRTC's old
    // cuboid Environment had no equivalent for, which is why rail_box_cuboids.json needed a
    // manual geometric trim earlier in this investigation. Joint index 1, not 0, is base_link -
    // verified directly (fanucm710_approx_sphere_to_joint only ever contains 1..7, never 0):
    // joint 0 is the geometry-less fixed root (fanucm710_approx_joint_id_to_dof[0] == FIXED) with
    // no spheres of its own, while joint 1 is the first DOF-bearing link (the one that moves
    // when the rail slides) - i.e. the physical base_link carriage riding the rail, which is
    // exactly the link colliding with Unified_collision_model/rail_box independent of position.
    std::vector<std::vector<bool>> link_env_mask(kFanucm710JointCount, std::vector<bool>(objects.size(), false));
    link_env_mask[1][0] = true;  // base_link vs Unified_collision_model
    link_env_mask[1][1] = true;  // base_link vs rail_box

    return {std::move(grids), std::move(link_env_mask)};
}

// Same values as trajectory_planner's benchmark_p2p.cpp StartConfig()/GoalConfig(),
// [rail, joint1..joint6] - the fixed-point comparison problem for RSW-2740.
template <typename Configuration>
Configuration start_config() {
    return {0.57f, 0.18661060362323356f, 0.8490154146326416f, 0.523441695965619f,
            -1.4602122653885363f, -0.7901978188404326f, 0.0f};
}

template <typename Configuration>
Configuration goal_config() {
    return {2.928f, -0.05089380098815477f, 0.9122399667861361f, 0.557667602597228f,
            0.7137698508956003f, -0.7901978188404326f, 0.9852034561657597f};
}

template <typename Configuration>
nlohmann::json config_to_json(const Configuration& cfg) {
    return std::vector<float>(cfg.begin(), cfg.end());
}

// Same cost metric as benchmark_p2p.cpp's PathCost(): sum of consecutive-waypoint
// L2 distances. Recomputed after shortcutting since shortcutPath() changes the
// waypoints (and hence the cost) that pRRTC::solve() originally reported.
template <typename Configuration>
double path_cost(const std::vector<Configuration>& path) {
    double cost = 0.0;
    for (std::size_t i = 1; i < path.size(); ++i) {
        double d = 0.0;
        for (std::size_t j = 0; j < path[i].size(); ++j) {
            double diff = path[i][j] - path[i - 1][j];
            d += diff * diff;
        }
        cost += std::sqrt(d);
    }
    return cost;
}

}  // namespace

int main() {
    using Robot = ppln::robots::Fanucm710;
    using Configuration = typename Robot::Configuration;

    // Auto-flush every std::cout insertion - without this, output sits in a
    // fully-buffered stream (the default once stdout isn't a terminal, e.g.
    // piped/redirected) and nothing appears until the buffer fills or the
    // process exits, which looks indistinguishable from a hang.
    std::cout << std::unitbuf;

    std::cout << "Loading SDF environment...\n";
    auto sdf_environment = load_sdf_environment();
    const std::vector<pRRTC::SDFGridHost>& sdf_grids = sdf_environment.first;
    const std::vector<std::vector<bool>>& sdf_link_mask = sdf_environment.second;

    // solve()'s public signature still takes an Environment<float>& - fixed by the shared
    // Planners.hh declaration, not ours to change - but its contents are no longer read; env
    // collision-checking now goes through the SDF grids uploaded above. Passed through
    // unchanged (empty) purely to satisfy that signature.
    Environment<float> env{};

    pRRTC_settings settings;
    settings.num_new_configs = 512;
    settings.max_iters = 5000;
    settings.granularity = 16;  // must match Fanucm710's BATCH_SIZE (16, from our fkcc_gen configs)
    settings.range = 0.5;
    settings.balance = 2;
    settings.tree_ratio = 1.0;
    settings.dynamic_domain = true;
    settings.dd_radius = 4.0;
    settings.dd_min_radius = 1.0;
    settings.dd_alpha = 0.0001;

    Configuration start = start_config<Configuration>();
    std::vector<Configuration> goals = {goal_config<Configuration>()};

    int num_success = 0;
    std::vector<float> costs;
    std::vector<double> wall_times_s;
    std::vector<double> kernel_times_s;
    nlohmann::json runs = nlohmann::json::array();

    for (int i = 0; i < kNumRuns; ++i) {
        std::cout << "Starting run " << i << " (max_iters=" << settings.max_iters << ")...\n";
        // Re-upload before every run, not just once - see load_sdf_environment()'s comment for
        // why: solve()'s own cudaDeviceReset() wipes this __constant__ state at the end of every
        // call, so without re-uploading here, every run after the first (and even run 0's own
        // shortcutPath phase) would silently check against zero environment objects. Done from
        // already-loaded host memory (no disk re-read) and before starting the timer, so this
        // doesn't get counted as planning time.
        pRRTC::uploadSDFEnvironment(sdf_grids, sdf_link_mask, kSelfCollisionOffsetM);

        auto t0 = std::chrono::steady_clock::now();
        auto result = pRRTC::solve<Robot>(start, goals, env, settings);

        double cost_before_shortcut = 0.0;
        double cost_after_shortcut = 0.0;
        double reupload_s = 0.0;
        if (result.solved) {
            cost_before_shortcut = path_cost(result.path);
            // solve() (just returned above) ends with its own cudaDeviceReset() call, which
            // wipes the same __constant__ state uploadSDFEnvironment() just populated - same
            // bug as the run-loop re-upload above, just at this second call site. Without
            // re-uploading here, shortcutPath's edge validation would silently see zero
            // environment objects (env/workpiece collision checking disabled) and an unset
            // self-collision offset, for every run, not just run 0. Re-uploaded from the same
            // already-loaded host memory, so this doesn't re-read the .bin files either. Timed
            // and subtracted from planning_time_s below - like the pre-solve upload above, this
            // is purely a benchmark-only artifact of solve()'s cudaDeviceReset(), not a cost P2P
            // itself pays, so it shouldn't count as planning time.
            auto tu0 = std::chrono::steady_clock::now();
            pRRTC::uploadSDFEnvironment(sdf_grids, sdf_link_mask, kSelfCollisionOffsetM);
            auto tu1 = std::chrono::steady_clock::now();
            reupload_s = std::chrono::duration<double>(tu1 - tu0).count();
            result.path =
                pRRTC::shortcutPath<Robot>(result.path, settings.range, settings.granularity, kShortcutMaxAttempts);
            cost_after_shortcut = path_cost(result.path);
        }
        auto t1 = std::chrono::steady_clock::now();
        const double planning_time_s = std::chrono::duration<double>(t1 - t0).count() - reupload_s;
        std::cout << "  run " << i << " returned after " << planning_time_s
                   << "s (host-side, solve+shortcut)\n";

        nlohmann::json run;
        run["iteration"] = i;
        run["success"] = result.solved;
        // Wall time now covers solve() + the shortcut refinement loop, matching how
        // benchmark_p2p.cpp times the whole solvePath() call (refinement included).
        run["planning_time_s"] = planning_time_s;
        run["solve_wall_time_s"] = result.wall_ns / 1e9;
        run["kernel_time_s"] = result.kernel_ns / 1e9;
        run["copy_time_s"] = result.copy_ns / 1e9;
        run["iters"] = result.iters;
        run["start_tree_size"] = result.start_tree_size;
        run["goal_tree_size"] = result.goal_tree_size;

        if (result.solved) {
            ++num_success;
            costs.push_back(static_cast<float>(cost_after_shortcut));
            wall_times_s.push_back(planning_time_s);
            kernel_times_s.push_back(result.kernel_ns / 1e9);
            run["path_cost"] = cost_after_shortcut;
            run["path_cost_before_shortcut"] = cost_before_shortcut;
            nlohmann::json path = nlohmann::json::array();
            for (auto& cfg : result.path) {
                path.push_back(config_to_json(cfg));
            }
            run["path"] = path;
        }
        runs.push_back(std::move(run));

        std::cout << "run " << i << ": solved=" << result.solved << " cost=" << cost_after_shortcut
                   << " (before shortcut=" << cost_before_shortcut << ") wall_ms=" << planning_time_s * 1e3
                   << " start_tree_size=" << result.start_tree_size << " goal_tree_size=" << result.goal_tree_size
                   << "\n";
    }

    nlohmann::json output;
    output["algorithm"] = "pRRTC";
    output["robot"] = "fanuc_m710";
    output["environment_representation"] = "sdf";
    output["start_config"] = config_to_json(start);
    output["goal_config"] = config_to_json(goals[0]);
    output["settings"] = {
        {"num_new_configs", settings.num_new_configs},
        {"granularity", settings.granularity},
        {"range", settings.range},
        {"balance", settings.balance},
        {"shortcut_max_attempts", kShortcutMaxAttempts},
    };
    output["runs"] = runs;
    output["success_rate"] = static_cast<double>(num_success) / kNumRuns;
    if (!costs.empty()) {
        output["path_cost_mean"] = std::accumulate(costs.begin(), costs.end(), 0.0) / costs.size();
        output["path_cost_min"] = *std::min_element(costs.begin(), costs.end());
        output["path_cost_max"] = *std::max_element(costs.begin(), costs.end());
        output["planning_time_s_mean"] =
            std::accumulate(wall_times_s.begin(), wall_times_s.end(), 0.0) / wall_times_s.size();
        output["kernel_time_s_mean"] =
            std::accumulate(kernel_times_s.begin(), kernel_times_s.end(), 0.0) / kernel_times_s.size();
    }

    const auto now_s = std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
    const std::string output_dir = runtime_data_path("p2p_benchmark_paths");
    const std::string output_path = output_dir + "/prrtc_benchmark_" + std::to_string(now_s) + ".json";
    std::ofstream(output_path) << output.dump(2);
    std::cout << "Wrote " << output_path << "\n";

    return 0;
}
