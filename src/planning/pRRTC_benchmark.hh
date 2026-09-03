#pragma once

// Benchmark-only API (RSW-2740): declares shortcutPath() so scripts/benchmark_fanuc_m710.cpp
// (a plain .cpp, can't call __global__ kernels directly) can call into
// pRRTC_benchmark.cu, which defines it. Deliberately separate from Planners.hh - this is
// not part of upstream pRRTC's API, just a dev tool for comparing against P2P's refinement step.
//
// Also declares the SDF-based environment upload (uploadSDFEnvironment/SDFGridHost), added when
// env collision-checking moved from a brute-force cuboid list to direct SDF lookups (same
// RSW-2740 investigation) - see fanuc_m710_benchmark.cuh and src/collision/sdf_environment.hh
// for the device-side half of this. Kept to plain types here (no SDFGrid/Fanuc-specific
// constants in this header) so the .cpp driver doesn't need to see any of that - it just reads
// raw bytes out of a .bin file and hands them over.

#include <array>
#include <cstdint>
#include <vector>

#include "Planners.hh"

namespace pRRTC {

// One entry per environment object (e.g. Unified_collision_model, rail_box, workpiece). Order
// must be consistent with how link_env_mask's columns are indexed in the call to
// uploadSDFEnvironment() below - same object order in both.
struct SDFGridHost {
    std::vector<uint8_t> raw_bytes;  // raw payload bytes, read straight from the .bin file
    bool is_float;                    // dtype of raw_bytes: int16_t if false, float if true
    unsigned int numX, numY, numZ;
    float boundsLower[3], boundsUpper[3];
    // Minimum clearance this object's collision check must respect, x100-scaled (cm) - same
    // convention as CollisionManager's m_env_collision_offset_/m_wp_collision_offset_. 0 if this
    // object has no required margin.
    float offset_scaled;
    // Inverse of this object's world placement (scene_config's position/orientation) - RSW-2740.
    // The caller computes this once, on the host (build the forward world transform from
    // position/quaternion, invert it), so neither this header nor uploadSDFEnvironment() need to
    // know anything about quaternions. Row-major 3x3 inv_rotation + inv_translation; identity
    // (rotation=I, translation=0) for an object placed at the world origin with no rotation. See
    // src/collision/sdf_environment.hh's SDFGrid for how the device side applies this.
    float inv_rotation[3][3];
    float inv_translation[3];
};

// Uploads `grids` (one per environment object) and the per-(link, object) collision mask to the
// device, once, before solve()/shortcutPath() run - both now read the environment through this
// global state rather than a cuboid Environment<float> parameter. `link_env_mask[joint][g]` ==
// true means that joint's spheres skip checking against grids[g] entirely (mirrors
// scene_config's collisionMask). `self_collision_offset_m` is the minimum surface separation
// (in meters, unscaled - matches CollisionManager leaving m_self_collision_offset_ in meters)
// self-collision checks must maintain between robot spheres.
void uploadSDFEnvironment(
    const std::vector<SDFGridHost> &grids,
    const std::vector<std::vector<bool>> &link_env_mask,
    float self_collision_offset_m
);

// A single tool collision sphere, in the tool's own local frame (the same joint7/link_6-local
// frame fanuc_m710_benchmark.cuh's tool-sphere reservation uses - see that file's comment on the
// (x, -y, -z) convention a URDF-authored tool0 frame needs converting through).
struct ToolSphereHost {
    float x, y, z, radius;
};

// Uploads the current test case's tool spheres (RSW-2740) into the fixed capacity
// fanuc_m710_benchmark.cuh reserves for them (FANUCM710_MAX_TOOL_SPHERES fine-tier,
// FANUCM710_APPROX_MAX_TOOL_SPHERES approx-tier) - a pure data upload, so switching tools between
// test cases never requires touching that file or re-running cricket. `fine`/`approx` may have
// fewer spheres than the reserved capacity (any tool is fine as long as it fits); slots beyond
// what's provided are left at the inert placeholder those arrays already default to, so nothing
// needs to track how many slots are "really" active at check time - every existing collision
// loop and self_cc_ranges bound stays exactly as-is. Must be re-uploaded before every
// solve()/shortcutPath() call, same reason and same timing constraints as uploadSDFEnvironment()
// (see its own comment for why: solve()'s own cudaDeviceReset() wipes this state at the end of
// every call). Throws if either list exceeds its reserved capacity.
void uploadToolSpheres(
    const std::vector<ToolSphereHost> &fine,
    const std::vector<ToolSphereHost> &approx
);

// A 3x3 rotation matrix, row-major (RSW-2740) - describes a case's "frames[1]" mounting rotation
// (scene_config_m710.json's chains[0].frames[1] quaternion), as a plain matrix so this header
// doesn't need to know anything about quaternions. Identity for a case with no mounting rotation
// (e.g. Collins itself).
struct Mat3Host {
    float m[3][3];
};

// Overrides base_link's sphere data and the one fixed transform that carries a case's
// "frames[1]" mounting rotation onward into link_1+ (RSW-2740) - both needed because pRRTC's own
// generated FK (cricket) uses a different convention from P2P/Manipulator's KDL-based chain for
// where that rotation lives:
//
// P2P's chain gives base_link its own segment that receives frames[1] directly (confirmed via
// Manipulator::modifyChains()'s splice). pRRTC's cricket-generated FK instead keeps joint 1 (the
// rail) permanently unrotated and absorbs frames[1]'s rotation into joint 2's own fixed transform
// (verified directly: joint 1's fixed transform is identity in every case; joint 2's rotation
// submatrix is where a non-identity frames[1] actually shows up). That means a case's
// robot_sphere_models/{fine,approx}.yaml - already case-specific, already correct for P2P's own
// consumption - needs base_link's centers *additionally* rotated by frames1_rotation before
// pRRTC can use them directly: pRRTC's joint 1 never rotates, so whatever local offset it's given
// *is* the sphere's position relative to the rail, whereas P2P arrives at that same position by
// rotating base_link's own (frames[1]-rotated) frame around a local offset. Link_1 onward need no
// such sphere-data change (frame-invariant, confirmed empirically) - only joint 2's fixed
// transform needs frames1_rotation composed onto it: new_rotation = frames1_rotation *
// canonical_rotation, new_translation = frames1_rotation * canonical_translation, where
// "canonical" is read back from whatever's currently compiled into
// fanucm710_fixed_transforms/fanucm710_approx_fixed_transforms (via cudaMemcpyFromSymbol) rather
// than duplicated as a second hardcoded constant, so there's only one place that ever needs to
// change if the robot's own joint-2 geometry is ever re-baked.
//
// `base_link_fine`/`base_link_approx` overwrite base_link's own reserved capacity in
// fanucm710_spheres_array/fanucm710_approx_spheres_array (24/16 entries at the front of each
// array - see fanuc_m710_benchmark.cuh's FANUCM710_APPROX_MAX_BASE_LINK_SPHERES) - fewer spheres
// than that capacity are padded with the same inert placeholder uploadToolSpheres() uses (e.g.
// pierce_primer's real fine-tier base_link has 22, not 24); throws if either list exceeds its
// capacity, since growing it for real would need the array itself expanded (cricket re-run for
// fine tier, or the same by-hand shift approx tier's 4->16 bump used). Must be re-uploaded before every
// solve()/shortcutPath() call, same reason as uploadSDFEnvironment()/uploadToolSpheres() (solve()'s
// own cudaDeviceReset() wipes this state, but also restores fanucm710_fixed_transforms's
// compile-time values, so reading "canonical" back
// via cudaMemcpyFromSymbol on every call is always the true original, never a previous call's
// composed result).
void uploadRobotOverrides(
    const std::vector<ToolSphereHost> &base_link_fine,
    const std::vector<ToolSphereHost> &base_link_approx,
    const Mat3Host &frames1_rotation
);

void uploadSettings(const pRRTC_settings &settings);

// Uploads the 7 (lower, upper) bounds that Fanucm710::scale_cfg() maps every random Halton
// sample into before EXTEND ever uses it (RSW-2740) - i.e. the region solve()'s random sampling
// can ever place a node in, regardless of iterations or range. These are case-specific (mirrors
// Manipulator::modifyChains(), manipulator.cpp:578-620: dof 0 is scene_config's own
// axisLowerLimits/axisUpperLimits for whichever axis is active, dofs 1-6 are its
// jointLowerLimits/jointUpperLimits directly), not a fixed property of the robot - see
// Robots.hh's fanucm710_dof_s_m/fanucm710_dof_s_a for why those used to be compile-time
// constexpr (baked from Collins' own case) and now aren't. Must be re-uploaded before every
// solve() call, same reset-timing reason as uploadRobotOverrides() (not needed before
// shortcutPath() specifically - it never calls scale_cfg - but uploaded there too for
// consistency with the other per-case overrides).
void uploadJointLimits(const std::array<float, 7> &lower, const std::array<float, 7> &upper);

// Mirrors P2P::postProcessing() (trajectory_planner/src/p2p.cpp): repeatedly picks two random
// waypoints in `path`, checks whether connecting them with a direct straight line is
// collision-free, and keeps the shortcut if it reduces total path cost (sum of consecutive
// waypoint L2 distances, matching benchmark_p2p.cpp's PathCost()). Runs `max_attempts` times
// (P2P's max_path_refinement_attempts is 200). Direct connections are validated in `range`-sized
// chunks at `granularity` steps each - the same per-chunk resolution solve()'s own tree-growth
// edge validation uses - since a shortcut can span many original tree edges at once. Reads the
// environment via the same global SDF state uploadSDFEnvironment() populates.
template <typename Robot>
std::vector<typename Robot::Configuration> shortcutPath(
    const std::vector<typename Robot::Configuration> &path,
    float range,
    int granularity,
    int max_attempts
);

}  // namespace pRRTC
