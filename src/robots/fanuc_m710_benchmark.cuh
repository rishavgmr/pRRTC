#include "src/collision/sdf_environment.hh"

namespace ppln::collision {





#define FANUCM710_APPROX_SPHERE_COUNT 39
#define FANUCM710_APPROX_MAX_TOOL_SPHERES 10
// Reserved base_link capacity (RSW-2740) - was exactly 4 (Collins's own approx base_link count,
// with no slack) until pierce_primer's approx-tier base_link needed more for a visually-tuned
// fit; bumped to 16 for headroom. See uploadRobotOverrides()'s matching kBaseLinkApproxCapacity
// (pRRTC_benchmark.cu) - the two must stay in sync by convention, same as every other
// count/capacity pairing in this file.
#define FANUCM710_APPROX_MAX_BASE_LINK_SPHERES 16
#define FANUCM710_APPROX_JOINT_COUNT 8
#define FANUCM710_APPROX_SELF_CC_RANGE_COUNT 28
#define FIXED -1
#define X_PRISM 0
#define Y_PRISM 1
#define Z_PRISM 2
#define X_ROT 3
#define Y_ROT 4
#define Z_ROT 5
#define BATCH_SIZE 16

// Minimum surface separation self-collision must maintain between robot spheres, in meters
// (unscaled - matches CollisionManager leaving m_self_collision_offset_ in meters, since its
// own self-collision check works on true GJK surface separation, also in meters). Uploaded
// once via uploadSDFEnvironment(), same mechanism as the SDF grids/mask further below.
__device__ __constant__ float fanucm710_self_collision_offset;

// Joints 6 and 7 are the last two axes of the M710's spherical wrist, which intersect near a
// common point by mechanical design - verified empirically (RSW-2740) that the sphere pairs
// pRRTC's self-collision range tables retain between them sit at ~0 clearance (approximate tier,
// sphere 15 vs 17/18) or ~2.1-2.2cm clearance (detailed tier, sphere 59 vs 77) for essentially
// every valid configuration, both below the required 2.5cm self-collision margin, with no
// relation to actual collision risk. Mirrors CollisionManager::selfPhysxCollisionQuery's own
// exclusion (collision_manager.cpp:592), which requires an even wider separation than the
// standard adjacent-link exclusion for the last 3 links specifically, for the same reason.
__device__ __forceinline__ bool fanucm710_wrist_pair_excluded(int joint_a, int joint_b) {
    int lo = joint_a < joint_b ? joint_a : joint_b;
    int hi = joint_a < joint_b ? joint_b : joint_a;
    return lo == 6 && hi == 7;
}

__device__ __constant__ float4 fanucm710_approx_spheres_array[39] = {
    // Reserves FANUCM710_APPROX_MAX_BASE_LINK_SPHERES (16) slots for the current test case's
    // approx-tier base_link geometry (RSW-2740) - unlike link_1 onward below (frame-invariant,
    // baked in once), base_link's centers are case-specific (see uploadRobotOverrides()'s own
    // comment for why) and always fully overwritten by that function before any solve()/
    // shortcutPath() call runs, for every case including Collins - so these compile-time values
    // are never actually read; same inert-placeholder convention as the tool-sphere reservation
    // below regardless.
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 0.07f, 0.128f, -0.049f, 0.3117f },
    { 0.023f, -0.136f, -0.049f, 0.28573f },
    { 0.007f, -0.528f, 0.183f, 0.1628f },
    { -0.011f, -0.778f, 0.179f, 0.16798f },
    { 0.007f, -1.006f, 0.179f, 0.14822f },
    { -0.0057f, -0.276f, 0.142f, 0.1921f },
    { 0.049f, -0.017f, 0.019f, 0.32657f },
    { 0.0f, 0.0f, -0.627f, 0.14706f },
    { 0.0f, 0.001f, -1.109f, 0.12745f },
    { 0.0f, 0.0f, -0.37f, 0.14706f },
    { 0.002f, -0.003f, -0.872f, 0.14278f },
    { -0.003f, -0.017f, 0.006f, 0.128f },
    { 0.006f, 0.007f, 0.065f, 0.0973f },
    // Reserves FANUCM710_APPROX_MAX_TOOL_SPHERES (10) slots for whatever tool the current test
    // case mounts (RSW-2740) - populated at runtime by uploadToolSpheres(), same reasoning and
    // inert-placeholder convention as fanucm710_spheres_array's tool reservation above.
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f }
};

__device__ __constant__ float fanucm710_approx_fixed_transforms[] = {
    // joint 0
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 1
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 2
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.565,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 3
    1.0, -0.0, 0.0, 0.15,
    0.0, -4e-06, 1.0, 0.0,
    0.0, -1.0, -4e-06, 0.0,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 4
    -4e-06, -1.0, -7e-06, 0.0,
    -1.0, 4e-06, 0.0, -1.15,
    0.0, 7e-06, -1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 5
    -1.0, 0.0, 7e-06, 0.17,
    7e-06, 4e-06, 1.0, 0.0,
    -0.0, 1.0, -4e-06, 0.0,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 6
    -1.0, 0.0, 7e-06, 0.0,
    7e-06, 4e-06, 1.0, 0.0,
    -0.0, 1.0, -4e-06, -1.295,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 7
    1.0, -0.0, 0.0, 0.0,
    0.0, -4e-06, 1.0, -0.175,
    0.0, -1.0, -4e-06, 0.0,
    0.0, 0.0, 0.0, 1.0,
    
    
};

__device__ __constant__ int fanucm710_approx_sphere_to_joint[39] = {
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    2,
    2,
    3,
    3,
    3,
    3,
    4,
    5,
    5,
    5,
    5,
    6,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7
};

__device__ __constant__ int fanucm710_approx_flattened_joint_to_spheres[47] = {
    -1,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    -1,
    16,
    17,
    -1,
    18,
    19,
    20,
    21,
    -1,
    22,
    -1,
    23,
    24,
    25,
    26,
    -1,
    27,
    -1,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    -1
};

__device__ __constant__ int fanucm710_approx_joint_types[] = {
    3,
    1,
    5,
    5,
    5,
    5,
    5,
    5
};

__device__ __constant__ int fanucm710_approx_self_cc_ranges[28][3] = {
    { 0, 18, 38 },
    { 1, 18, 38 },
    { 2, 18, 38 },
    { 3, 18, 38 },
    { 4, 18, 38 },
    { 5, 18, 38 },
    { 6, 18, 38 },
    { 7, 18, 38 },
    { 8, 18, 38 },
    { 9, 18, 38 },
    { 10, 18, 38 },
    { 11, 18, 38 },
    { 12, 18, 38 },
    { 13, 18, 38 },
    { 14, 18, 38 },
    { 15, 18, 38 },
    { 16, 22, 38 },
    { 17, 22, 38 },
    { 18, 23, 38 },
    { 19, 23, 38 },
    { 20, 23, 38 },
    { 21, 23, 38 },
    { 22, 29, 38 },
    { 23, 28, 38 },
    { 24, 28, 38 },
    { 25, 28, 38 },
    { 26, 28, 38 },
    { 27, 29, 38 }
};

__device__ __constant__ int fanucm710_approx_joint_parents[8] = {
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6
};

__device__ __constant__ int fanucm710_approx_T_memory_idx[8] = {
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0
};

__device__ __constant__ int fanucm710_approx_dfs_order[8] = {
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7
};

__device__ __constant__ int fanucm710_approx_joint_id_to_dof[8] = {
    18446744073709551615,
    0,
    1,
    2,
    3,
    4,
    5,
    6
};

template <>
__device__ void fk_approx<ppln::robots::Fanucm710>(
    const float* q,
    volatile float* sphere_pos_approx, // 20 spheres x 16 robots x 3 coordinates (each column is a robot)
    float *T, // 16 robots x 1 x 4x4 transform matrix , column major
    const int tid
)
{
    // every 4 threads are responsible for one column of the transform matrix T
    // make_transform will calculate the necessary column of T_step needed for the thread
    const int col_ind = tid % 4;
    const int batch_ind = tid / 4;

    int T_offset = batch_ind * 1 * 16;
    float T_step_col[4]; // 4x1 column of the joint transform matrix for this thread
    float *T_base = T + T_offset; // 4x4 transform matrix for the batch
    
    #pragma unroll
    for (int i = 0; i < 1; ++i) {
        float *T_col_i = T_base + i * 16 + col_ind * 4;
        for (int r=0; r<4; r++) {
            T_col_i[r] = 0.0f;
        }
        T_col_i[col_ind] = 1.0f;
    }
    __syncthreads();

    int joint_to_sphere_ind = 0;

    for (int j = 0; j < FANUCM710_APPROX_JOINT_COUNT; ++j) {
        int i = fanucm710_approx_dfs_order[j];
        float T_col_tmp[4];
        int parent_idx = fanucm710_approx_joint_parents[i];
        int T_memory_idx_parent = fanucm710_approx_T_memory_idx[parent_idx];
        int T_memory_idx = fanucm710_approx_T_memory_idx[i];
        int q_idx = fanucm710_approx_joint_id_to_dof[i];
        if (j > 0) {
            int ft_addr_start = i * 16;
            int joint_type = fanucm710_approx_joint_types[i];

            if (joint_type <= Z_PRISM) {
                prism_fn(&fanucm710_approx_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col, joint_type);
            }
            else if (joint_type == X_ROT) {
                xrot_fn(&fanucm710_approx_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
            }
            else if (joint_type == Y_ROT) {
                yrot_fn(&fanucm710_approx_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
            }
            else if (joint_type == Z_ROT) {
                zrot_fn(&fanucm710_approx_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
            }
            
            for (int r=0; r<4; r++){
                T_col_tmp[r] = dot4_col(&T_base[T_memory_idx_parent*16 + r], T_step_col);
            }
            // RSW-2740: T_memory_idx_parent == T_memory_idx for this robot's chain (a single
            // shared accumulator slot, not one per joint) - the read above and the write below
            // therefore touch the SAME memory, and every col_ind thread does both. Without a
            // sync between them, a faster thread's write here can land before a slower thread's
            // read above finishes, handing that thread a mix of old and new matrix data - a real,
            // GPU-scheduling-dependent race (confirmed via compute-sanitizer --tool racecheck,
            // "Potential WAR hazard (Warp Level Programming)" at this exact line, and empirically
            // via pierce_primer's benchmark returning collision-passing-through-workpiece paths
            // nondeterministically despite provably identical input data across runs). All 4
            // col_ind threads for a batch share a warp (col_ind = tid % 4, so 4 consecutive
            // threads), so a plain __syncwarp() is sufficient here (and correctly required,
            // unlike the pre-existing __syncwarp() right after this block, which only guards the
            // start of the *next* section against this one, not the read/write pair within it).
            __syncwarp();
            for (int r=0; r<4; r++){
                T_base[T_memory_idx*16 + col_ind*4 + r] = T_col_tmp[r];
            }
        }
        __syncwarp();
        while (fanucm710_approx_flattened_joint_to_spheres[joint_to_sphere_ind] != -1) {
            int sphere_ind = fanucm710_approx_flattened_joint_to_spheres[joint_to_sphere_ind];
            if (col_ind < 3) {
                // sphere sphere_ind, robot batch_ind (BATCH_SIZE robots), coord col_ind
                sphere_pos_approx[sphere_ind * BATCH_SIZE * 3 + batch_ind * 3 + col_ind] = 
                    T_base[T_memory_idx*16 + col_ind] * fanucm710_approx_spheres_array[sphere_ind].x +
                    T_base[T_memory_idx*16 + col_ind + M] * fanucm710_approx_spheres_array[sphere_ind].y +
                    T_base[T_memory_idx*16 + col_ind + M*2] * fanucm710_approx_spheres_array[sphere_ind].z +
                    T_base[T_memory_idx*16 + col_ind + M*3];
            }
            joint_to_sphere_ind++;
        }
        joint_to_sphere_ind++;
        __syncthreads();
    }
}

// 4 threads per discretized motion for self-collision check
template <>
__device__ bool self_collision_check_approx<ppln::robots::Fanucm710>(volatile float* sphere_pos_approx, volatile int* joint_in_collision, const int tid){
    const int thread_ind = tid % 4;
    const int batch_ind = tid / 4;
    bool out = true;
    for (int i = thread_ind; i < FANUCM710_APPROX_SELF_CC_RANGE_COUNT; i+=4) {
        int sphere_1_ind = fanucm710_approx_self_cc_ranges[i][0];
        float sphere_1[3] = {
            sphere_pos_approx[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 0],
            sphere_pos_approx[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 1],
            sphere_pos_approx[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 2]
        };
        for (int j = fanucm710_approx_self_cc_ranges[i][1]; j <= fanucm710_approx_self_cc_ranges[i][2]; j++) {
            float sphere_2[3] = {
                sphere_pos_approx[j * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                sphere_pos_approx[j * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                sphere_pos_approx[j * BATCH_SIZE * 3 + batch_ind * 3 + 2]
            };
            // Inlined rather than calling the shared sphere_sphere_self_collision (utils.cuh,
            // used by every other robot) so the required clearance margin can be added without
            // changing that shared function's signature. Equivalent to CollisionManager's own
            // self-collision check (true GJK surface separation < m_self_collision_offset_):
            // inflating the combined radius by the offset flags the same condition, since both
            // reduce to center distance < r1 + r2 + offset. Skips joint-6/joint-7 pairs entirely
            // (see fanucm710_wrist_pair_excluded above) rather than applying the margin
            // unconditionally to every retained range pair.
            int joint_1 = fanucm710_approx_sphere_to_joint[sphere_1_ind];
            int joint_2 = fanucm710_approx_sphere_to_joint[j];
            if (fanucm710_wrist_pair_excluded(joint_1, joint_2)) {
                continue;
            }
            float dx = sphere_1[0] - sphere_2[0];
            float dy = sphere_1[1] - sphere_2[1];
            float dz = sphere_1[2] - sphere_2[2];
            float dist_sq = dx * dx + dy * dy + dz * dz;
            float rs = fanucm710_approx_spheres_array[sphere_1_ind].w + fanucm710_approx_spheres_array[j].w
                + fanucm710_self_collision_offset;
            if (dist_sq < rs * rs) {
                atomicAdd((int*)&joint_in_collision[20*batch_ind + joint_1], 1);
                out = false;
            }
        } 
    }
    return out;
}

// --- Benchmark-only SDF-based env collision checking, approximate tier (RSW-2740) ---
//
// Replaces the O(num_cuboids) sphere_environment_in_collision loop (still used by upstream
// pRRTC via env_collision_check_approx in the ORIGINAL fanuc_m710.cuh) with O(num_sdf_grids)
// direct SDF lookups via sdf_lookup() (src/collision/sdf_environment.hh).
//
// Deliberately NOT written as a `template <> ... env_collision_check_approx<Fanucm710>`
// specialization: the primary template lives in the SHARED src/planning/utils.cuh (used by
// every robot and by evaluate_mbm/single_mbm too), and changing a specialization's signature
// without changing that shared declaration would be a mismatch. A plain, distinctly-named
// function sidesteps the template dispatch system entirely - callers (pRRTC_benchmark.cu's
// rrtc()) call this by name.
//
// fanucm710_link_env_mask mirrors scene_config_m710.json's collisionMask (currently: base_link
// exempted from Unified_collision_model and rail_box) - populated once, host-side, and uploaded
// via cudaMemcpyToSymbol before the kernel launches, same mechanism as d_settings. It never
// varies across threads or across a kernel's lifetime, so reading it causes no more divergence
// than the joint-type dispatch in fk()/fk_approx() already does.
#define FANUCM710_MAX_SDF_GRIDS 8
__device__ __constant__ bool fanucm710_link_env_mask[FANUCM710_APPROX_JOINT_COUNT][FANUCM710_MAX_SDF_GRIDS];
// Global (not a kernel parameter): rrtc()'s own signature is internal to this file and free to
// change, but check_edge_segment/validate_candidates_kernel/shortcutPath (used for the shortcut
// refinement step) are called from a chain that ultimately still needs to satisfy solve()'s
// PUBLIC signature (declared in the shared Planners.hh, not ours to change). Global __constant__
// state, uploaded once via uploadSDFEnvironment() before solve()/shortcutPath() run, sidesteps
// needing to thread a new parameter through any of that - same mechanism as d_settings.
__device__ __constant__ ppln::collision::SDFGrid fanucm710_sdf_grids[FANUCM710_MAX_SDF_GRIDS];
__device__ __constant__ int fanucm710_num_sdf_grids;

__device__ bool fanucm710_env_collision_check_approx_sdf(
    volatile float* sphere_pos_approx,
    volatile int* joint_in_collision,
    const int tid
) {
    const int thread_ind = tid % 4;
    const int batch_ind = tid / 4;
    bool out = true;

    for (int i = thread_ind; i < FANUCM710_APPROX_SPHERE_COUNT; i += 4) {
        int joint = fanucm710_approx_sphere_to_joint[i];
        float3 sphere_pos = make_float3(
            sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
            sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
            sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 2]
        );
        // SDF distances are x100-scaled (see sdf_environment.hh) - scale the sphere radius to
        // match before comparing, same convention as CollisionManager's own
        // `sdf_distance - (sphere_radius * 100)`.
        float radius_scaled = fanucm710_approx_spheres_array[i].w * 100.0f;

        for (int g = 0; g < fanucm710_num_sdf_grids; g++) {
            if (fanucm710_link_env_mask[joint][g]) {
                continue;
            }
            float dist = ppln::collision::sdf_lookup(fanucm710_sdf_grids[g], sphere_pos, FLT_MAX);
            // Flag not just actual penetration but any clearance below this grid's required
            // margin - same comparison CollisionManager's envPhysXCollisionQuery/
            // workpiecePhysXCollisionQuery make against collision_offset.
            if (dist - radius_scaled < fanucm710_sdf_grids[g].offset_scaled) {
                atomicAdd((int*) &joint_in_collision[20 * batch_ind + joint], 1);
                out = false;
            }
        }
    }
    return out;
}




#define FANUCM710_SPHERE_COUNT 148
#define FANUCM710_MAX_TOOL_SPHERES 80
#define FANUCM710_JOINT_COUNT 8
#define FANUCM710_SELF_CC_RANGE_COUNT 63
#define FIXED -1
#define X_PRISM 0
#define Y_PRISM 1
#define Z_PRISM 2
#define X_ROT 3
#define Y_ROT 4
#define Z_ROT 5
#define BATCH_SIZE 16

__device__ __constant__ float4 fanucm710_spheres_array[148] = {
    { 0.279f, 0.653f, 0.188f, 0.23272f },
    { -0.132f, 0.972f, 0.158f, 0.20229f },
    { -0.104f, 0.213f, 0.157f, 0.20229f },
    { -0.478f, -0.084f, 0.157f, 0.22f },
    { 0.161f, -0.151f, 0.158f, 0.20229f },
    { -0.136f, 0.624f, 0.17f, 0.20229f },
    { 0.142f, 1.625f, 0.166f, 0.2f },
    { 0.408f, 0.946f, 0.157f, 0.20229f },
    { 0.152f, 1.283f, 0.177f, 0.2f },
    { 0.149f, 0.17f, 0.157f, 0.20229f },
    { 0.322f, 0.791f, 0.468f, 0.15f },
    { -0.155f, 1.376f, 0.169f, 0.17931f },
    { -0.132f, -0.087f, 0.158f, 0.20229f },
    { -0.463f, 0.146f, 0.143f, 0.22f },
    { 0.4f, 0.741f, 0.14f, 0.2f },
    { 0.321f, 0.614f, 0.468f, 0.15f },
    { 0.134f, 0.781f, 0.468f, 0.15f },
    { 0.141f, 0.626f, 0.468f, 0.15f },
    { -0.159f, 1.282f, 0.177f, 0.2f },
    { -0.164f, 1.64f, 0.177f, 0.2f },
    { 0.152f, 0.981f, 0.177f, 0.2f },
    { 0.376f, 0.522f, 0.156f, 0.2199f },
    { 0.084f, 0.45f, 0.177f, 0.2f },
    { -0.174f, 0.357f, 0.177f, 0.2f },
    { 0.026f, 0.128f, -0.049f, 0.265f },
    { -0.053f, -0.136f, -0.049f, 0.265f },
    { 0.142f, -0.065f, -0.042f, 0.258f },
    { -0.141f, 0.056f, -0.046f, 0.262f },
    { 0.0f, -0.571f, 0.362f, 0.105f },
    { 0.0f, -0.721f, 0.362f, 0.07f },
    { 0.0f, -0.471f, 0.362f, 0.07f },
    { 0.009f, -0.2f, 0.126f, 0.13f },
    { -0.02f, -0.22f, 0.132f, 0.13f },
    { 0.002f, -0.532f, 0.179f, 0.104f },
    { -0.031f, -1.15f, 0.179f, 0.104f },
    { -0.016f, -0.829f, 0.179f, 0.104f },
    { 0.007f, -1.008f, 0.179f, 0.104f },
    { -0.003f, -0.413f, 0.179f, 0.104f },
    { 0.013f, -0.664f, 0.179f, 0.104f },
    { -0.034f, -0.009f, 0.122f, 0.133f },
    { 0.053f, -0.025f, 0.14f, 0.13f },
    { 0.002f, -1.199f, 0.184f, 0.099f },
    { 0.143f, 0.019f, -0.019f, 0.182f },
    { 0.01f, 0.125f, 0.101f, 0.139f },
    { -0.002f, -0.05f, 0.153f, 0.128f },
    { 0.009f, -0.116f, -0.078f, 0.123f },
    { 0.01f, 0.095f, -0.061f, 0.14f },
    { 0.17f, -0.162f, 0.029f, 0.11f },
    { 0.159f, 0.075f, 0.183f, 0.109f },
    { 0.0f, 0.0f, -0.693f, 0.103f },
    { 0.0f, 0.001f, -1.109f, 0.103f },
    { 0.0f, 0.0f, -0.441f, 0.103f },
    { 0.0f, 0.004f, -0.341f, 0.1f },
    { 0.0f, 0.0f, -0.57f, 0.103f },
    { 0.007f, 0.054f, -1.281f, 0.079f },
    { 0.0f, 0.0f, -0.97f, 0.103f },
    { -0.001f, 0.001f, -0.834f, 0.102f },
    { 0.0f, 0.003f, -1.245f, 0.076f },
    { -0.011f, 0.069f, -1.337f, 0.064f },
    { -0.003f, -0.093f, 0.001f, 0.076f },
    { 0.0f, 0.03f, 0.007f, 0.07f },
    { -0.001f, 0.004f, -0.069f, 0.07f },
    { -0.001f, -0.06f, -0.062f, 0.07f },
    { -0.017f, -0.05f, 0.011f, 0.024f },
    { -0.045f, -0.003f, 0.011f, 0.024f },
    { 0.032f, -0.022f, 0.01f, 0.024f },
    { 0.029f, 0.029f, 0.011f, 0.024f },
    { -0.019f, 0.037f, 0.011f, 0.024f },
    // Reserves FANUCM710_MAX_TOOL_SPHERES (80) slots for whatever tool the current test case
    // mounts (RSW-2740) - populated at runtime by uploadToolSpheres(), not baked in here, so
    // switching tools/test cases never requires touching this file or re-running cricket. All
    // slots default to this inert placeholder (a sphere so far outside the workspace, with zero
    // radius, that it can never register a self- or env-collision under the existing formulas
    // unchanged - see sdf_lookup()'s out-of-bounds clamp-distance penalty and
    // sphere_sphere_sql2's r1+r2 sum) - uploadToolSpheres() overwrites the first N with the
    // active tool's real spheres and leaves the rest exactly as-is for tools with fewer than 80.
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f },
    { 1e6f, 1e6f, 1e6f, 0.0f }
};

__device__ __constant__ float fanucm710_fixed_transforms[] = {
    // joint 0
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 1
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 2
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.565,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 3
    1.0, -0.0, 0.0, 0.15,
    0.0, -4e-06, 1.0, 0.0,
    0.0, -1.0, -4e-06, 0.0,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 4
    -4e-06, -1.0, -7e-06, 0.0,
    -1.0, 4e-06, 0.0, -1.15,
    0.0, 7e-06, -1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 5
    -1.0, 0.0, 7e-06, 0.17,
    7e-06, 4e-06, 1.0, 0.0,
    -0.0, 1.0, -4e-06, 0.0,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 6
    -1.0, 0.0, 7e-06, 0.0,
    7e-06, 4e-06, 1.0, 0.0,
    -0.0, 1.0, -4e-06, -1.295,
    0.0, 0.0, 0.0, 1.0,
    
    // joint 7
    1.0, -0.0, 0.0, 0.0,
    0.0, -4e-06, 1.0, -0.175,
    0.0, -1.0, -4e-06, 0.0,
    0.0, 0.0, 0.0, 1.0,
    
    
};

__device__ __constant__ int fanucm710_sphere_to_joint[148] = {
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    2,
    2,
    2,
    2,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    6,
    6,
    6,
    6,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7
};

__device__ __constant__ int fanucm710_flattened_joint_to_spheres[156] = {
    -1,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    -1,
    24,
    25,
    26,
    27,
    -1,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    -1,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    -1,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    -1,
    59,
    60,
    61,
    62,
    -1,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    123,
    124,
    125,
    126,
    127,
    128,
    129,
    130,
    131,
    132,
    133,
    134,
    135,
    136,
    137,
    138,
    139,
    140,
    141,
    142,
    143,
    144,
    145,
    146,
    147,
    -1
};

__device__ __constant__ int fanucm710_joint_types[] = {
    3,
    1,
    5,
    5,
    5,
    5,
    5,
    5
};

__device__ __constant__ int fanucm710_self_cc_ranges[63][3] = {
    { 0, 28, 147 },
    { 1, 28, 147 },
    { 2, 28, 147 },
    { 3, 28, 147 },
    { 4, 28, 147 },
    { 5, 28, 147 },
    { 6, 28, 147 },
    { 7, 28, 147 },
    { 8, 28, 147 },
    { 9, 28, 147 },
    { 10, 28, 147 },
    { 11, 28, 147 },
    { 12, 28, 147 },
    { 13, 28, 147 },
    { 14, 28, 147 },
    { 15, 28, 147 },
    { 16, 28, 147 },
    { 17, 28, 147 },
    { 18, 28, 147 },
    { 19, 28, 147 },
    { 20, 28, 147 },
    { 21, 28, 147 },
    { 22, 28, 147 },
    { 23, 28, 147 },
    { 24, 42, 147 },
    { 25, 42, 147 },
    { 26, 42, 147 },
    { 27, 42, 147 },
    { 28, 49, 147 },
    { 29, 49, 147 },
    { 30, 49, 147 },
    { 31, 49, 147 },
    { 32, 49, 147 },
    { 33, 49, 147 },
    { 34, 49, 147 },
    { 35, 49, 147 },
    { 36, 49, 147 },
    { 37, 49, 147 },
    { 38, 49, 147 },
    { 39, 49, 147 },
    { 40, 49, 147 },
    { 41, 49, 147 },
    { 42, 68, 147 },
    { 43, 68, 147 },
    { 44, 68, 147 },
    { 45, 68, 147 },
    { 46, 68, 147 },
    { 47, 68, 147 },
    { 48, 68, 147 },
    { 49, 63, 147 },
    { 50, 63, 147 },
    { 51, 63, 147 },
    { 52, 63, 147 },
    { 53, 63, 147 },
    { 54, 63, 147 },
    { 55, 63, 147 },
    { 56, 63, 147 },
    { 57, 63, 147 },
    { 58, 63, 147 },
    { 59, 68, 147 },
    { 60, 68, 147 },
    { 61, 68, 147 },
    { 62, 68, 147 }
};

__device__ __constant__ int fanucm710_joint_parents[8] = {
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6
};

__device__ __constant__ int fanucm710_T_memory_idx[8] = {
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0
};

__device__ __constant__ int fanucm710_dfs_order[8] = {
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7
};

__device__ __constant__ int fanucm710_joint_id_to_dof[8] = {
    18446744073709551615,
    0,
    1,
    2,
    3,
    4,
    5,
    6
};

template <>
__device__ void fk<ppln::robots::Fanucm710>(
    const float* q,
    volatile float* sphere_pos, // 84 spheres x 16 robots x 3 coordinates (each column is a robot)
    float *T, // 16 robots x 1 x 4x4 transform matrix , column major
    const int tid
)
{
    // every 4 threads are responsible for one column of the transform matrix T
    // make_transform will calculate the necessary column of T_step needed for the thread
    const int col_ind = tid % 4;
    const int batch_ind = tid / 4;

    int T_offset = batch_ind * 1 * 16;
    float T_step_col[4]; // 4x1 column of the joint transform matrix for this thread
    float *T_base = T + T_offset; // 4x4 transform matrix for the batch
    
    #pragma unroll
    for (int i = 0; i < 1; ++i) {
        float *T_col_i = T_base + i * 16 + col_ind * 4;
        for (int r=0; r<4; r++) {
            T_col_i[r] = 0.0f;
        }
        T_col_i[col_ind] = 1.0f;
    }
    __syncthreads();

    int joint_to_sphere_ind = 0;

    for (int j = 0; j < FANUCM710_JOINT_COUNT; ++j) {
        int i = fanucm710_dfs_order[j];
        float T_col_tmp[4];
        int parent_idx = fanucm710_joint_parents[i];
        int T_memory_idx_parent = fanucm710_T_memory_idx[parent_idx];
        int T_memory_idx = fanucm710_T_memory_idx[i];
        int q_idx = fanucm710_joint_id_to_dof[i];
        if (j > 0) {
            int ft_addr_start = i * 16;
            int joint_type = fanucm710_joint_types[i];

            if (joint_type <= Z_PRISM) {
                prism_fn(&fanucm710_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col, joint_type);
            }
            else if (joint_type == X_ROT) {
                xrot_fn(&fanucm710_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
            }
            else if (joint_type == Y_ROT) {
                yrot_fn(&fanucm710_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
            }
            else if (joint_type == Z_ROT) {
                zrot_fn(&fanucm710_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
            }
            
            for (int r=0; r<4; r++){
                T_col_tmp[r] = dot4_col(&T_base[T_memory_idx_parent*16 + r], T_step_col);
            }
            // RSW-2740: T_memory_idx_parent == T_memory_idx for this robot's chain (a single
            // shared accumulator slot, not one per joint) - the read above and the write below
            // therefore touch the SAME memory, and every col_ind thread does both. Without a
            // sync between them, a faster thread's write here can land before a slower thread's
            // read above finishes, handing that thread a mix of old and new matrix data - a real,
            // GPU-scheduling-dependent race (confirmed via compute-sanitizer --tool racecheck,
            // "Potential WAR hazard (Warp Level Programming)" at this exact line, and empirically
            // via pierce_primer's benchmark returning collision-passing-through-workpiece paths
            // nondeterministically despite provably identical input data across runs). All 4
            // col_ind threads for a batch share a warp (col_ind = tid % 4, so 4 consecutive
            // threads), so a plain __syncwarp() is sufficient here (and correctly required,
            // unlike the pre-existing __syncwarp() right after this block, which only guards the
            // start of the *next* section against this one, not the read/write pair within it).
            __syncwarp();
            for (int r=0; r<4; r++){
                T_base[T_memory_idx*16 + col_ind*4 + r] = T_col_tmp[r];
            }
        }
        __syncwarp();
        while (fanucm710_flattened_joint_to_spheres[joint_to_sphere_ind] != -1) {
            int sphere_ind = fanucm710_flattened_joint_to_spheres[joint_to_sphere_ind];
            if (col_ind < 3) {
                // sphere sphere_ind, robot batch_ind (BATCH_SIZE robots), coord col_ind
                sphere_pos[sphere_ind * BATCH_SIZE * 3 + batch_ind * 3 + col_ind] = 
                    T_base[T_memory_idx*16 + col_ind] * fanucm710_spheres_array[sphere_ind].x +
                    T_base[T_memory_idx*16 + col_ind + M] * fanucm710_spheres_array[sphere_ind].y +
                    T_base[T_memory_idx*16 + col_ind + M*2] * fanucm710_spheres_array[sphere_ind].z +
                    T_base[T_memory_idx*16 + col_ind + M*3];
            }
            joint_to_sphere_ind++;
        }
        joint_to_sphere_ind++;
        __syncthreads();
    }
}

// 4 threads per discretized motion for self-collision check
template <>
__device__ bool self_collision_check<ppln::robots::Fanucm710>(volatile float* sphere_pos, volatile int* joint_in_collision, const int tid){
    const int thread_ind = tid % 4;
    const int batch_ind = tid / 4;
    bool has_collision = false;

    for (int i = thread_ind; i < FANUCM710_SELF_CC_RANGE_COUNT; i += 4) {
        if (warp_any_active_mask(has_collision)) return false;
        int sphere_1_ind = fanucm710_self_cc_ranges[i][0];
        if (joint_in_collision[20*batch_ind + fanucm710_sphere_to_joint[sphere_1_ind]] == 0) continue;
        float sphere_1[3] = {
            sphere_pos[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 0],
            sphere_pos[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 1],
            sphere_pos[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 2]
        };
        for (int j = fanucm710_self_cc_ranges[i][1]; j <= fanucm710_self_cc_ranges[i][2]; j++) {
            float sphere_2[3] = {
                sphere_pos[j * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                sphere_pos[j * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                sphere_pos[j * BATCH_SIZE * 3 + batch_ind * 3 + 2]
            };
            // See self_collision_check_approx above for why this is inlined rather than calling
            // the shared sphere_sphere_self_collision, and for fanucm710_wrist_pair_excluded.
            if (fanucm710_wrist_pair_excluded(fanucm710_sphere_to_joint[sphere_1_ind], fanucm710_sphere_to_joint[j])) {
                continue;
            }
            float dx = sphere_1[0] - sphere_2[0];
            float dy = sphere_1[1] - sphere_2[1];
            float dz = sphere_1[2] - sphere_2[2];
            float dist_sq = dx * dx + dy * dy + dz * dz;
            float rs = fanucm710_spheres_array[sphere_1_ind].w + fanucm710_spheres_array[j].w
                + fanucm710_self_collision_offset;
            if (dist_sq < rs * rs) {
                //return false;
                has_collision=true;
            }
        }
    }
    return !has_collision;

}

// --- Benchmark-only SDF-based env collision checking, detailed tier (RSW-2740) ---
// See env_sdf_approx.cuh above for the full rationale (plain function, not a template
// specialization; fanucm710_link_env_mask's semantics).
__device__ bool fanucm710_env_collision_check_sdf(
    volatile float* sphere_pos,
    volatile int* joint_in_collision,
    const int tid
) {
    const int thread_ind = tid % 4;
    const int batch_ind = tid / 4;
    bool has_collision = false;

    // The striped loop (i = thread_ind, thread_ind+4, ...) covers every sphere exactly once
    // regardless of whether FANUCM710_SPHERE_COUNT is a multiple of 4 - that part needs no
    // special-casing. What does require care is warp_any_active_mask below: it reads the actual
    // live-lane mask via __activemask(), so it stays correct even when threads' loop trip counts
    // differ (unlike a hardcoded-full-mask __any_sync, which requires every lane in the warp to
    // reach the same call on the same iteration - see the original cuboid-based env_collision_check's
    // floor-to-multiple-of-4-plus-explicit-tail pattern, needed there specifically because it uses
    // warp_any_full_mask instead).
    for (int i = thread_ind; i < FANUCM710_SPHERE_COUNT; i += 4) {
        int joint = fanucm710_sphere_to_joint[i];
        // Only re-check (in the finer detailed model) joints the approximate pass already
        // flagged as suspect - same per-joint gating the original detailed env_collision_check
        // uses, still valid here: the approximate spheres remain a conservative bound around
        // each joint's true geometry regardless of how the environment itself is represented.
        if (joint_in_collision[20 * batch_ind + joint] > 0) {
            float3 sphere_pos_i = make_float3(
                sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 2]
            );
            float radius_scaled = fanucm710_spheres_array[i].w * 100.0f;

            for (int g = 0; g < fanucm710_num_sdf_grids; g++) {
                if (fanucm710_link_env_mask[joint][g]) {
                    continue;
                }
                float dist = ppln::collision::sdf_lookup(fanucm710_sdf_grids[g], sphere_pos_i, FLT_MAX);
                if (dist - radius_scaled < fanucm710_sdf_grids[g].offset_scaled) {
                    has_collision = true;
                }
            }
        }
        if (warp_any_active_mask(has_collision)) {
            return false;
        }
    }
    return true;
}
}

namespace ppln::robots {

// Real definition of the extern arrays declared in Robots.hh's Fanucm710 struct (RSW-2740) - see
// that declaration's own comment for why this needs to live here (the one file this header's
// arrays' matching TU, pRRTC_benchmark.cu, includes) rather than in Robots.hh directly. Defaults
// are Collins' own case: dof 0 (rail) from its rail length (0 to 3.0, see
// robots/fanuc_m710/fine_spheres.urdf's world_joint limit); dofs 1-6 from robot.urdf's hardware
// limits (close to but not identical to Collins' own tighter operational jointLowerLimits/
// jointUpperLimits). Both get overwritten by uploadJointLimits() before any solve() call runs,
// for every case including Collins, so these compile-time values are in practice never read.
__device__ __constant__ float fanucm710_dof_s_m[7] = {
    3.0f, 6.2831854820251465f, 3.9269907474517822f, 6.457718372344971f, 12.566370964050293f, 4.363323211669922f, 12.566370964050293f
};
__device__ __constant__ float fanucm710_dof_s_a[7] = {
    0.0f, -3.1415927410125732f, -1.5707963705062866f, -1.5707963705062866f, -6.2831854820251465f, -2.181661605834961f, -6.2831854820251465f
};

}
