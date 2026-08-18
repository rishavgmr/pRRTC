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
