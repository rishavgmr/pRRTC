#pragma once

// Benchmark-only (RSW-2740): direct-lookup SDF collision checking, ported from our own
// PhysxNode::PxSDFSampleImpl (trajectory_planner/src/physx_collisions.cpp:179-227) - the exact
// clamp/floor/trilinear-blend math the real system uses to query a dense SDF grid, replacing
// pRRTC's O(num_cuboids) brute-force sphere_environment_in_collision loop with an O(1)
// (8-corner trilinear lookup) query per environment object.
//
// Matches the real system's storage convention exactly: int16_t for env/rail_box-style objects,
// float for the workpiece (dtype carried per-grid via `is_float`, auto-detected at load time -
// see loadSDFArray vs loadFloatSDFArray), and the same x100-scaled distance-value convention
// (grid values and returned distances are in that scaled unit; spatial bounds/spacing stay in
// plain meters) - see CollisionManager::updateCustomCollisionOffsets's *100.0 and
// convertAndSaveSDF's sdf[i]*100.0f for the two ends of that convention in the real system.
//
// `spacing` is derived per-grid from (boundsUpper-boundsLower)/(dim-1) at load time rather than
// relying on a separately-configured global value the way PhysxNode::m_sdf_spacing_ does - same
// derivation voxel_sdf_to_cuboids.py's read_sdf_grid already uses, and self-contained (each grid
// carries everything needed to interpret itself, no dependency on "whatever the current global
// setting happens to be").

#include <cuda_runtime.h>
#include <cfloat>

namespace ppln::collision {

struct SDFGrid {
    const void* data;  // int16_t* or float*, dtype given by is_float
    bool is_float;
    unsigned int numX, numY, numZ;
    float3 boundsLower, boundsUpper;
    float spacing;
    // Minimum clearance this object's collision check must respect, x100-scaled to match
    // sdf_lookup()'s returned distance convention - mirrors CollisionManager's per-object
    // env/workpiece offset (m_env_collision_offset_ / m_wp_collision_offset_), which the real
    // system also pre-scales by *100.0 in updateCustomCollisionOffsets. 0 for a grid with no
    // required margin.
    float offset_scaled;
    // World-to-local rigid transform (RSW-2740) - boundsLower/boundsUpper/the voxel data are all
    // expressed in this object's own local frame (wherever its mesh was authored), but robot
    // sphere positions are computed in world frame. inv_rotation_row{0,1,2}/inv_translation are
    // the *inverse* of the object's world placement (scene_config's position/orientation),
    // precomputed once on the host (see benchmark_fanuc_m710.cpp's load_sdf_environment()) -
    // sdf_lookup() applies local = invR*world + invT as its very first step, mirroring
    // CollisionManager's own `sphere_ball_transform = invTransform * sphere_ball_transform`
    // (collision_manager.cpp's envPhysXCollisionQuery). Identity (row0=(1,0,0), row1=(0,1,0),
    // row2=(0,0,1), inv_translation=(0,0,0)) for an object placed at the world origin with no
    // rotation - a pure no-op in that case, which is why this was never needed while every test
    // case's real (non-excluded) env objects happened to sit at the origin.
    float3 inv_rotation_row0, inv_rotation_row1, inv_rotation_row2;
    float3 inv_translation;
};

// Same flat-index convention as PhysX's PxSDFIdx (foundation/PxMathUtils.h) and our own
// voxel_sdf_to_cuboids.py's read_sdf_grid: X fastest, Z slowest.
__device__ __forceinline__ unsigned int sdf_flat_idx(
    unsigned int i, unsigned int j, unsigned int k, unsigned int dimX, unsigned int dimY) {
    return i + j * dimX + k * dimX * dimY;
}

// Which dtype a grid uses never varies across threads or across a kernel's lifetime (fixed per
// environment object, uploaded once) - same non-divergent category as the joint-type dispatch
// in fk()/fk_approx() or the collision-mask check: every thread reading this grid's data, for
// every candidate, for the whole run, sees the same is_float value.
__device__ __forceinline__ float sdf_read_voxel(const SDFGrid& grid, unsigned int idx) {
    if (grid.is_float) {
        return ((const float*) grid.data)[idx];
    }
    return (float) ((const short*) grid.data)[idx];
}

// Direct port of PhysxNode::PxSDFSampleImpl. `tolerance` mirrors the real system's own call
// sites, which always pass PX_MAX_F32 (i.e. never reject purely for being outside the box - just
// pay the linear penalty below); pass something smaller only if out-of-box points should be
// treated as "unknown, ignore" instead of "far, but still estimable."
__device__ __forceinline__ float sdf_lookup(const SDFGrid& grid, float3 world_pos, float tolerance) {
    // World -> this object's local frame (RSW-2740) - see SDFGrid's own comment on why. A no-op
    // for an identity-placed object (row0=(1,0,0) etc., inv_translation=(0,0,0)).
    float3 pos = make_float3(
        grid.inv_rotation_row0.x * world_pos.x + grid.inv_rotation_row0.y * world_pos.y + grid.inv_rotation_row0.z * world_pos.z + grid.inv_translation.x,
        grid.inv_rotation_row1.x * world_pos.x + grid.inv_rotation_row1.y * world_pos.y + grid.inv_rotation_row1.z * world_pos.z + grid.inv_translation.y,
        grid.inv_rotation_row2.x * world_pos.x + grid.inv_rotation_row2.y * world_pos.y + grid.inv_rotation_row2.z * world_pos.z + grid.inv_translation.z);

    float3 clamped;
    clamped.x = fminf(fmaxf(pos.x, grid.boundsLower.x), grid.boundsUpper.x);
    clamped.y = fminf(fmaxf(pos.y, grid.boundsLower.y), grid.boundsUpper.y);
    clamped.z = fminf(fmaxf(pos.z, grid.boundsLower.z), grid.boundsUpper.z);

    float3 diff = make_float3(pos.x - clamped.x, pos.y - clamped.y, pos.z - clamped.z);
    float diff_sq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
    if (diff_sq > tolerance * tolerance) {
        return FLT_MAX;
    }

    float invDx = 1.0f / grid.spacing;
    float fx = (clamped.x - grid.boundsLower.x) * invDx;
    float fy = (clamped.y - grid.boundsLower.y) * invDx;
    float fz = (clamped.z - grid.boundsLower.z) * invDx;

    unsigned int i = (unsigned int) fx;
    unsigned int j = (unsigned int) fy;
    unsigned int k = (unsigned int) fz;

    fx -= (float) i;
    fy -= (float) j;
    fz -= (float) k;

    // Boundary clamp so (i+1, j+1, k+1) never runs off the grid - matches PxSDFSampleImpl
    // exactly (the corresponding clampedGridPt mutation there is dead code w.r.t. the returned
    // value, since clampedGridPt is never read again afterward - omitted here on purpose).
    if (i >= grid.numX - 1) {
        i = grid.numX - 2;
        fx = 1.0f;
    }
    if (j >= grid.numY - 1) {
        j = grid.numY - 2;
        fy = 1.0f;
    }
    if (k >= grid.numZ - 1) {
        k = grid.numZ - 2;
        fz = 1.0f;
    }

    float s000 = sdf_read_voxel(grid, sdf_flat_idx(i, j, k, grid.numX, grid.numY));
    float s100 = sdf_read_voxel(grid, sdf_flat_idx(i + 1, j, k, grid.numX, grid.numY));
    float s010 = sdf_read_voxel(grid, sdf_flat_idx(i, j + 1, k, grid.numX, grid.numY));
    float s110 = sdf_read_voxel(grid, sdf_flat_idx(i + 1, j + 1, k, grid.numX, grid.numY));
    float s001 = sdf_read_voxel(grid, sdf_flat_idx(i, j, k + 1, grid.numX, grid.numY));
    float s101 = sdf_read_voxel(grid, sdf_flat_idx(i + 1, j, k + 1, grid.numX, grid.numY));
    float s011 = sdf_read_voxel(grid, sdf_flat_idx(i, j + 1, k + 1, grid.numX, grid.numY));
    float s111 = sdf_read_voxel(grid, sdf_flat_idx(i + 1, j + 1, k + 1, grid.numX, grid.numY));

    // Standard trilinear blend over the 8 corners - not necessarily bit-identical to PxTriLerp's
    // internal term ordering (that implementation isn't in our own source), but the same
    // interpolation, same corners, same weights. Per the RSW-2740 decision, matching value/unit
    // conventions is what matters here, not bit-for-bit rounding.
    float c00 = s000 * (1.0f - fx) + s100 * fx;
    float c10 = s010 * (1.0f - fx) + s110 * fx;
    float c01 = s001 * (1.0f - fx) + s101 * fx;
    float c11 = s011 * (1.0f - fx) + s111 * fx;
    float c0 = c00 * (1.0f - fy) + c10 * fy;
    float c1 = c01 * (1.0f - fy) + c11 * fy;
    float dist = c0 * (1.0f - fz) + c1 * fz;

    // If the query point was outside the box, add the (x100-scaled) clamp distance as a penalty -
    // same convention as PxSDFSampleImpl's `dist += diff.magnitude() * 100`.
    dist += sqrtf(diff_sq) * 100.0f;

    return dist;
}

}  // namespace ppln::collision
