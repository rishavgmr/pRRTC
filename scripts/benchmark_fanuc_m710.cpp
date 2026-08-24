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
#include <Eigen/Dense>
#include <array>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <numeric>
#include <algorithm>
#include <utility>
#include <regex>
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

// Scene/env/workpiece/tool data lives under a per-test-case directory (RSW-2740), separate from
// the robot's own fixtures directly under kFixtureDir/robots/fanuc_m710/ (URDF, sphere models,
// srdf, cricket configs), which don't vary per case. Runtime-selectable via PRRTC_BENCHMARK_CASE
// (defaults to "collins"), mirroring benchmark_p2p.cpp's P2P_BENCHMARK_CASE - nothing else in
// this file needs to change to run a different scene/tool/robot-override combination.
std::string case_name() {
    const char* env = std::getenv("PRRTC_BENCHMARK_CASE");
    return env ? std::string(env) : std::string("collins");
}
const std::string kCaseName = case_name();
const std::string kCaseDir = kFixtureDir + "/cases/" + kCaseName;

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

// Extracts a mesh path's bare filename with no extension, e.g.
// "package://.../env_meshes/rail_box_1000.STL" -> "rail_box_1000" - used to derive each env
// object's .bin path from the .STL path scene_config_m710.json actually stores (RSW-2740), since
// the physics .bin and visual/collision .STL for a given object always share a basename.
std::string mesh_basename_no_ext(const std::string& path) {
    const std::size_t slash = path.find_last_of('/');
    const std::string base = (slash == std::string::npos) ? path : path.substr(slash + 1);
    const std::size_t dot = base.find_last_of('.');
    return (dot == std::string::npos) ? base : base.substr(0, dot);
}

// robot_link names scene_config_m710.json's collisionMask can reference, mapped to pRRTC's joint
// index (RSW-2740) - only base_link is known to appear in any case's collisionMask today; throws
// rather than silently no-op-ing if a case ever exempts a different link, so a missing mapping
// gets noticed instead of quietly ignored.
int robot_link_to_joint_index(const std::string& robot_link) {
    if (robot_link == "base_link") {
        // Joint 1, not 0: joint 0 is the geometry-less fixed root
        // (fanucm710_approx_joint_id_to_dof[0] == FIXED, no spheres of its own); joint 1 is the
        // first DOF-bearing link (rides the rail) - i.e. the physical base_link carriage,
        // verified directly against fanucm710_approx_sphere_to_joint earlier in this
        // investigation.
        return 1;
    }
    throw std::runtime_error(
        "robot_link_to_joint_index: unrecognized robot_link \"" + robot_link +
        "\" in collisionMask - add a mapping for it");
}

// World-to-local inverse rigid transform for one env object (RSW-2740) - see SDFGrid's own
// comment (src/collision/sdf_environment.hh) for why this is needed at all: the SDF's
// bounds/voxels are in the object's own local frame, but robot sphere positions are computed in
// world frame. Mirrors CollisionManager's own inverse-transform construction
// (collision_manager.cpp's m_env_inv_T3_map_), just computed here instead since pRRTC's plain
// SDFGrid has no PhysX transform type of its own.
struct InverseTransform {
    float rotation[3][3];
    float translation[3];
};

InverseTransform compute_inverse_transform(const nlohmann::json& position, const nlohmann::json& orientation) {
    const Eigen::Vector3d t(position[0].get<double>(), position[1].get<double>(), position[2].get<double>());
    const Eigen::Quaterniond q(
        orientation[3].get<double>(), orientation[0].get<double>(), orientation[1].get<double>(), orientation[2].get<double>());
    const Eigen::Matrix3d r_inv = q.toRotationMatrix().transpose();
    const Eigen::Vector3d t_inv = -r_inv * t;

    InverseTransform result;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            result.rotation[r][c] = static_cast<float>(r_inv(r, c));
        }
        result.translation[r] = static_cast<float>(t_inv(r));
    }
    return result;
}

void set_grid_transform(pRRTC::SDFGridHost& grid, const InverseTransform& inv) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            grid.inv_rotation[r][c] = inv.rotation[r][c];
        }
        grid.inv_translation[r] = inv.translation[r];
    }
}

void set_grid_transform_identity(pRRTC::SDFGridHost& grid) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            grid.inv_rotation[r][c] = (r == c) ? 1.0f : 0.0f;
        }
        grid.inv_translation[r] = 0.0f;
    }
}

// Loads this case's environment objects, driven entirely by its own scene_config_m710.json (RSW-
// 2740) rather than a hardcoded object list, so different cases (different object names/counts/
// placements) don't require touching this function - see mesh_basename_no_ext/
// compute_inverse_transform above for the two things that varying per case actually requires.
// RESTRICTED_ZONE is a reserved scene_config key, deliberately excluded from collision checking
// on both sides of this comparison (matches benchmark_p2p.cpp's own
// addIgnoredObject("RESTRICTED_ZONE") - see the RSW-2740 project memory on why).
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
    nlohmann::json scene_json;
    std::ifstream(kCaseDir + "/scene_config_m710.json") >> scene_json;

    std::vector<pRRTC::SDFGridHost> grids;
    std::vector<std::string> names;  // parallel to grids, for the collisionMask name-matching below

    for (const auto& env_obj : scene_json["env"]) {
        const std::string name = env_obj["name"].get<std::string>();
        if (name == "RESTRICTED_ZONE") {
            continue;
        }

        const std::string bin_path =
            kCaseDir + "/env_meshes/" + mesh_basename_no_ext(env_obj["path"].get<std::string>()) + ".bin";
        auto grid = load_sdf_grid_host(bin_path);
        // Every env object draws from the single shared environment margin - matches
        // envPhysXCollisionQuery/envPhysXDistanceQuery applying m_env_collision_offset_
        // uniformly across every object in the env loop, with no per-object distinction.
        grid.offset_scaled = kEnvironmentCollisionOffsetM * 100.0f;
        set_grid_transform(grid, compute_inverse_transform(env_obj["position"], env_obj["orientation"]));

        std::cout << name << ": " << grid.numX << "x" << grid.numY << "x" << grid.numZ << " voxels, "
                   << (grid.is_float ? "float" : "int16") << ", " << grid.raw_bytes.size() / (1024 * 1024)
                   << " MB, offset=" << grid.offset_scaled << " (scaled)\n";

        names.push_back(name);
        grids.push_back(std::move(grid));
    }

    // Workpiece is a separate, always-present file (not part of scene_config's "env" list) - its
    // own (larger) margin, and identity placement, matching benchmark_p2p.cpp's own
    // addPhysXWorkpiece(workpiece_bin, Eigen::Matrix4d::Identity()) (workpiece positioning is a
    // property of the planning problem/TCP, not a fixed scene transform).
    {
        auto grid = load_sdf_grid_host(kCaseDir + "/workpiece.bin");
        grid.offset_scaled = kWorkpieceCollisionOffsetM * 100.0f;
        set_grid_transform_identity(grid);
        std::cout << "workpiece: " << grid.numX << "x" << grid.numY << "x" << grid.numZ << " voxels, "
                   << (grid.is_float ? "float" : "int16") << ", " << grid.raw_bytes.size() / (1024 * 1024)
                   << " MB, offset=" << grid.offset_scaled << " (scaled)\n";
        names.push_back("workpiece");
        grids.push_back(std::move(grid));
    }

    // Per-(link, object) collision mask, built by name from scene_config's collisionMask rather
    // than hardcoded indices (RSW-2740) - different cases exempt different links/objects (e.g.
    // Collins exempts base_link from an object literally named "rail_box", while
    // pierce_primer's equivalent rail object is named "Rail_model" instead), so this has to be
    // resolved dynamically against whatever actually got loaded above, not assumed fixed.
    std::vector<std::vector<bool>> link_env_mask(kFanucm710JointCount, std::vector<bool>(grids.size(), false));
    for (const auto& mask_entry : scene_json["robots"][0]["chains"][0]["collisionMask"]) {
        const int joint = robot_link_to_joint_index(mask_entry["robot_link"].get<std::string>());
        for (const auto& masked_name_json : mask_entry["masked_env"]) {
            const std::string masked_name = masked_name_json.get<std::string>();
            const auto it = std::find(names.begin(), names.end(), masked_name);
            if (it == names.end()) {
                throw std::runtime_error("collisionMask references unknown env object \"" + masked_name + "\"");
            }
            link_env_mask[joint][std::distance(names.begin(), it)] = true;
        }
    }

    return {std::move(grids), std::move(link_env_mask)};
}

// A sphere as scanned directly out of a sphere-model yaml, still in whatever local frame that
// yaml's authored in - callers apply their own frame conversion on top (see
// load_tool_spheres_yaml/load_base_link_spheres_rotated below, which need different ones).
struct RawSphere {
    float x, y, z, radius;
};

// Scans `content` for every "center: [x, y, z]" / "radius: r" pair, in either of the two
// quoting styles this project's sphere yamls actually use in practice - unquoted keys
// (tool_spheres/*.yaml, e.g. CollisionManager::addPhysXTool's comment: "- tool0: - center: [...]
// radius: ...") and quoted keys (robot_sphere_models/*.yaml's "\"center\": [...]"). Deliberately
// not a real YAML parser: this project's benchmark scripts don't otherwise link one, and scanning
// for center/radius pairs is robust enough for this fixed, simple format regardless of comments,
// quoting, or exactly how a link name nests (RSW-2740).
std::vector<RawSphere> scan_spheres_raw(const std::string& content) {
    static const std::regex sphere_re(
        R"("?center"?:\s*\[\s*([-0-9.eE]+)\s*,\s*([-0-9.eE]+)\s*,\s*([-0-9.eE]+)\s*\]\s*\n\s*"?radius"?:\s*([-0-9.eE]+))");

    std::vector<RawSphere> spheres;
    for (std::sregex_iterator it(content.begin(), content.end(), sphere_re), end; it != end; ++it) {
        const auto& m = *it;
        spheres.push_back({std::stof(m[1]), std::stof(m[2]), std::stof(m[3]), std::stof(m[4])});
    }
    return spheres;
}

// Extracts the text of one top-level "- <key>:" list entry from a sphere-model yaml's
// collision_spheres section (RSW-2740), e.g. "base_link" out of fine.yaml/approx.yaml - runs
// until the next top-level entry (a line indented exactly two spaces, "\n  - ...", which a
// sphere's own more-deeply-indented "center:"/"radius:" lines never match) or end of file.
std::string extract_yaml_block(const std::string& content, const std::string& key) {
    const std::string start_marker = "- " + key + ":";
    const std::size_t start = content.find(start_marker);
    if (start == std::string::npos) {
        throw std::runtime_error("extract_yaml_block: key \"" + key + "\" not found");
    }
    const std::string rest = content.substr(start + start_marker.size());
    static const std::regex sibling_re(R"(\n  - )");
    std::smatch m;
    if (std::regex_search(rest, m, sibling_re)) {
        return rest.substr(0, m.position(0));
    }
    return rest;
}

std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("failed to open " + path);
    }
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// Reads a tool sphere yaml (RSW-2740, e.g. cases/collins/tool_spheres/fine.yaml). Centers come
// back already converted from the tool's raw, URDF-authored local frame into joint7/link_6's
// local frame (the (x, -y, -z) 180-degree rotation about X, confirmed against fine_spheres.urdf's
// originally-baked tool0 spheres during the TCP_5_1 swap, RSW-2740) - the same frame
// fanucm710_spheres_array's other entries already use, so callers can hand these straight to
// pRRTC::uploadToolSpheres() with no further conversion.
std::vector<pRRTC::ToolSphereHost> load_tool_spheres_yaml(const std::string& path) {
    const auto raw = scan_spheres_raw(read_file(path));
    if (raw.empty()) {
        throw std::runtime_error(path + ": found no center/radius sphere entries");
    }
    std::vector<pRRTC::ToolSphereHost> spheres;
    spheres.reserve(raw.size());
    for (const auto& s : raw) {
        spheres.push_back({s.x, -s.y, -s.z, s.radius});
    }
    return spheres;
}

// Reads base_link's sphere centers out of this case's own robot_sphere_models/{fine,approx}.yaml
// (the same case-specific files trajectory_planner's benchmark_p2p.cpp reads directly) and
// rotates them by frames1_rotation - see uploadRobotOverrides()'s header comment
// (pRRTC_benchmark.hh) for why pRRTC needs base_link's centers in a different frame than P2P
// does, and compute_frames1_rotation() below for where frames1_rotation itself comes from.
std::vector<pRRTC::ToolSphereHost> load_base_link_spheres_rotated(
    const std::string& path, const pRRTC::Mat3Host& frames1_rotation) {
    const auto raw = scan_spheres_raw(extract_yaml_block(read_file(path), "base_link"));
    if (raw.empty()) {
        throw std::runtime_error(path + ": found no base_link center/radius sphere entries");
    }
    std::vector<pRRTC::ToolSphereHost> spheres;
    spheres.reserve(raw.size());
    for (const auto& s : raw) {
        const float x = frames1_rotation.m[0][0] * s.x + frames1_rotation.m[0][1] * s.y + frames1_rotation.m[0][2] * s.z;
        const float y = frames1_rotation.m[1][0] * s.x + frames1_rotation.m[1][1] * s.y + frames1_rotation.m[1][2] * s.z;
        const float z = frames1_rotation.m[2][0] * s.x + frames1_rotation.m[2][1] * s.y + frames1_rotation.m[2][2] * s.z;
        spheres.push_back({x, y, z, s.radius});
    }
    return spheres;
}

// Builds frames1_rotation (RSW-2740): this case's chains[0].frames[1] quaternion, as a plain
// rotation matrix - see uploadRobotOverrides()'s header comment (pRRTC_benchmark.hh) for what
// this feeds into. Identity for a case (e.g. Collins) whose frames[1] is itself identity.
// frames[1]'s translation is checked (not just ignored) since a non-zero one would need handling
// this function doesn't implement - every case seen so far (Collins, pierce_primer) has it zero.
pRRTC::Mat3Host compute_frames1_rotation() {
    nlohmann::json scene_json;
    std::ifstream(kCaseDir + "/scene_config_m710.json") >> scene_json;
    const auto& frames1 = scene_json["robots"][0]["chains"][0]["frames"][1];

    for (int i = 0; i < 3; ++i) {
        if (frames1[i].get<double>() != 0.0) {
            throw std::runtime_error("compute_frames1_rotation: frames[1] has a non-zero translation - not handled");
        }
    }

    const Eigen::Quaterniond q(
        frames1[6].get<double>(), frames1[3].get<double>(), frames1[4].get<double>(), frames1[5].get<double>());
    const Eigen::Matrix3d r = q.toRotationMatrix();

    pRRTC::Mat3Host result;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            result.m[row][col] = static_cast<float>(r(row, col));
        }
    }
    return result;
}

// Builds the 7 (lower, upper) sampling bounds pRRTC::uploadJointLimits() needs (RSW-2740),
// mirroring exactly how Manipulator::modifyChains() assembles the same 7-value vectors
// (manipulator.cpp:578-620): dof 0 (the rail) comes from this case's scene_config_m710.json
// axisLowerLimits/axisUpperLimits, read at whichever index axisOptions marks active (not assumed
// to be index 1, even though that's what both cases seen so far use); dofs 1-6 come directly
// from jointLowerLimits/jointUpperLimits, in order. Using the same source Manipulator itself
// reads - not robot.urdf's raw hardware joint limits - keeps pRRTC's notion of "a valid sample"
// consistent with what P2P already enforces for this case, not just what the arm is mechanically
// capable of.
std::pair<std::array<float, 7>, std::array<float, 7>> compute_joint_limits() {
    nlohmann::json scene_json;
    std::ifstream(kCaseDir + "/scene_config_m710.json") >> scene_json;
    const auto& chain = scene_json["robots"][0]["chains"][0];

    const auto axis_options = chain["axisOptions"].get<std::vector<bool>>();
    int active_axis = -1;
    for (std::size_t i = 0; i < axis_options.size(); ++i) {
        if (axis_options[i]) {
            active_axis = static_cast<int>(i);
            break;
        }
    }
    if (active_axis < 0) {
        throw std::runtime_error("compute_joint_limits: no active entry in axisOptions");
    }

    std::array<float, 7> lower{}, upper{};
    lower[0] = chain["axisLowerLimits"][active_axis].get<float>();
    upper[0] = chain["axisUpperLimits"][active_axis].get<float>();

    const auto joint_lower = chain["jointLowerLimits"].get<std::vector<float>>();
    const auto joint_upper = chain["jointUpperLimits"].get<std::vector<float>>();
    if (joint_lower.size() != 6 || joint_upper.size() != 6) {
        throw std::runtime_error("compute_joint_limits: expected exactly 6 jointLowerLimits/jointUpperLimits entries");
    }
    for (int i = 0; i < 6; ++i) {
        lower[i + 1] = joint_lower[i];
        upper[i + 1] = joint_upper[i];
    }

    return {lower, upper};
}

// Reads this case's fixed-point comparison problem (RSW-2740), [rail, joint1..joint6], from its
// problem.json - previously two separate hardcoded literals (here and in trajectory_planner's
// benchmark_p2p.cpp), now a single per-case file both benchmarks read, so they can't silently
// drift apart.
template <typename Configuration>
Configuration config_from_json(const nlohmann::json& arr) {
    Configuration cfg{};
    for (std::size_t i = 0; i < arr.size() && i < cfg.size(); ++i) {
        cfg[i] = arr[i].get<float>();
    }
    return cfg;
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

    std::cout << "Loading tool spheres...\n";
    const std::vector<pRRTC::ToolSphereHost> fine_tool_spheres =
        load_tool_spheres_yaml(kCaseDir + "/tool_spheres/fine.yaml");
    const std::vector<pRRTC::ToolSphereHost> approx_tool_spheres =
        load_tool_spheres_yaml(kCaseDir + "/tool_spheres/approx.yaml");
    std::cout << "  fine: " << fine_tool_spheres.size() << " spheres, approx: "
               << approx_tool_spheres.size() << " spheres\n";

    std::cout << "Loading robot overrides (base_link spheres + joint 2 mounting rotation)...\n";
    const pRRTC::Mat3Host frames1_rotation = compute_frames1_rotation();
    const std::vector<pRRTC::ToolSphereHost> base_link_fine_spheres =
        load_base_link_spheres_rotated(kCaseDir + "/robot_sphere_models/fine.yaml", frames1_rotation);
    const std::vector<pRRTC::ToolSphereHost> base_link_approx_spheres =
        load_base_link_spheres_rotated(kCaseDir + "/robot_sphere_models/approx.yaml", frames1_rotation);

    std::cout << "Loading joint sampling bounds (rail + joint_1..6, from scene_config)...\n";
    const auto [joint_limit_lower, joint_limit_upper] = compute_joint_limits();

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

    nlohmann::json problem_json;
    std::ifstream(kCaseDir + "/problem.json") >> problem_json;
    Configuration start = config_from_json<Configuration>(problem_json["start"]);
    std::vector<Configuration> goals = {config_from_json<Configuration>(problem_json["goal"])};

    int num_success = 0;
    std::vector<float> costs;
    std::vector<double> wall_times_s;
    std::vector<double> kernel_times_s;
    nlohmann::json runs = nlohmann::json::array();

    // Uploaded ONCE, not per-run (RSW-2740): this __constant__ state used to get wiped by
    // solve()'s own cudaDeviceReset() at the end of every call, which is why every run (and
    // every post-solve shortcutPath()) used to have to re-upload it from these same host-side
    // buffers. Now that solve() no longer resets the device between calls, nothing wipes this
    // out from under us, so a single upload before the whole run loop is sufficient - this
    // removes ~25-30ms/run of otherwise-unnecessary PCIe transfer (the SDF grids alone are
    // ~430MB), previously excluded from the reported planning_time_s only by explicit
    // subtraction, not because the cost wasn't real.
    pRRTC::uploadSDFEnvironment(sdf_grids, sdf_link_mask, kSelfCollisionOffsetM);
    pRRTC::uploadToolSpheres(fine_tool_spheres, approx_tool_spheres);
    pRRTC::uploadRobotOverrides(base_link_fine_spheres, base_link_approx_spheres, frames1_rotation);
    pRRTC::uploadJointLimits(joint_limit_lower, joint_limit_upper);

    for (int i = 0; i < kNumRuns; ++i) {
        std::cout << "Starting run " << i << " (max_iters=" << settings.max_iters << ")...\n";

        auto t0 = std::chrono::steady_clock::now();
        auto result = pRRTC::solve<Robot>(start, goals, env, settings);

        double cost_before_shortcut = 0.0;
        double cost_after_shortcut = 0.0;
        if (result.solved) {
            cost_before_shortcut = path_cost(result.path);
            result.path =
                pRRTC::shortcutPath<Robot>(result.path, settings.range, settings.granularity, kShortcutMaxAttempts);
            cost_after_shortcut = path_cost(result.path);
        }
        auto t1 = std::chrono::steady_clock::now();
        const double planning_time_s = std::chrono::duration<double>(t1 - t0).count();
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
    output["case"] = kCaseName;
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
