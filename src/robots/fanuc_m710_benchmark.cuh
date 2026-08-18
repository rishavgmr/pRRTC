#include "src/collision/sdf_environment.hh"

namespace ppln::collision {





#define FANUCM710_APPROX_SPHERE_COUNT 20
#define FANUCM710_APPROX_JOINT_COUNT 8
#define FANUCM710_APPROX_SELF_CC_RANGE_COUNT 16
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

__device__ __constant__ float4 fanucm710_approx_spheres_array[20] = {
    { -0.07f, 0.01f, 0.033f, 0.34908f },
    { 0.17f, 0.73f, 0.13f, 0.57f },
    { 0.0f, 1.4f, 0.1f, 0.6f },
    { -0.55f, 0.1f, 0.16f, 0.48f },
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
    { -0.13f, 0.009977f, -0.250001f, 0.3f },
    { 0.0f, 0.209984f, -0.170019f, 0.27f },
    { 0.041f, -0.12801f, -0.104988f, 0.115f }
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

__device__ __constant__ int fanucm710_approx_sphere_to_joint[20] = {
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
    7
};

__device__ __constant__ int fanucm710_approx_flattened_joint_to_spheres[28] = {
    -1,
    0,
    1,
    2,
    3,
    -1,
    4,
    5,
    -1,
    6,
    7,
    8,
    9,
    -1,
    10,
    -1,
    11,
    12,
    13,
    14,
    -1,
    15,
    -1,
    16,
    17,
    18,
    19,
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

__device__ __constant__ int fanucm710_approx_self_cc_ranges[16][3] = {
    { 0, 6, 19 },
    { 1, 6, 19 },
    { 2, 6, 19 },
    { 3, 6, 19 },
    { 4, 10, 19 },
    { 5, 10, 19 },
    { 6, 11, 19 },
    { 7, 11, 19 },
    { 8, 11, 19 },
    { 9, 11, 19 },
    { 10, 17, 19 },
    { 11, 16, 19 },
    { 12, 16, 19 },
    { 13, 16, 19 },
    { 14, 16, 19 },
    { 15, 17, 19 }
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




#define FANUCM710_SPHERE_COUNT 112
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

__device__ __constant__ float4 fanucm710_spheres_array[112] = {
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
    // Replaces the 16 generic-tool spheres previously here with the 44 spheres actually used by
    // benchmark_p2p.cpp's Collins comparison (TCP_5_1_P2P_spheres.yaml, loaded there via
    // addPhysXTool) - the old 16 were from whatever tool0 fine_spheres.urdf happened to be
    // authored against, not the tool this benchmark is supposed to be comparing against.
    // Transformed into joint7/link_6's local frame via (x, -y, -z) - the same 180-degree
    // rotation about X confirmed against the original 16 spheres (matched to ~2e-5 on 4
    // independent samples); radius is unaffected by rotation.
    { 0.001f, -0.009f, -0.111f, 0.121f },
    { -0.007f, 0.243f, -0.107f, 0.09f },
    { 0.041f, -0.128f, -0.105f, 0.115f },
    { -0.053f, 0.008f, -0.258f, 0.1f },
    { 0.061f, 0.011f, -0.227f, 0.1f },
    { -0.019f, -0.162f, -0.096f, 0.106f },
    { -0.001f, 0.242f, -0.021f, 0.1f },
    { -0.041f, 0.028f, -0.138f, 0.106f },
    { 0.044f, -0.008f, -0.119f, 0.118f },
    { -0.084f, -0.153f, -0.142f, 0.088f },
    { 0.058f, -0.002f, -0.264f, 0.094f },
    { 0.001f, -0.24f, -0.156f, 0.079f },
    { -0.062f, -0.087f, -0.115f, 0.088f },
    { 0.013f, -0.244f, -0.061f, 0.077f },
    { 0.088f, -0.151f, -0.161f, 0.073f },
    { 0.074f, 0.033f, -0.069f, 0.079f },
    { -0.052f, 0.009f, -0.202f, 0.099f },
    { 0.081f, -0.033f, -0.222f, 0.079f },
    { -0.133f, 0.001f, -0.263f, 0.06f },
    { -0.007f, 0.24f, 0.075f, 0.082f },
    { -0.082f, -0.154f, -0.066f, 0.069f },
    { -0.075f, 0.039f, -0.059f, 0.069f },
    { 0.007f, 0.128f, -0.005f, 0.07f },
    { 0.102f, -0.069f, -0.05f, 0.06f },
    { 0.105f, -0.158f, -0.054f, 0.057f },
    { 0.104f, 0.046f, -0.301f, 0.056f },
    { 0.026f, 0.046f, -0.301f, 0.056f },
    { 0.019f, -0.044f, -0.301f, 0.056f },
    { 0.103f, -0.04f, -0.302f, 0.056f },
    { -0.007f, 0.061f, -0.259f, 0.056f },
    { -0.348f, -0.0f, -0.258f, 0.025f },
    { -0.348f, -0.038f, -0.258f, 0.025f },
    { -0.348f, -0.029f, -0.283f, 0.025f },
    { -0.348f, -0.007f, -0.296f, 0.025f },
    { -0.348f, 0.019f, -0.291f, 0.025f },
    { -0.348f, 0.036f, -0.271f, 0.025f },
    { -0.348f, 0.036f, -0.245f, 0.025f },
    { -0.348f, 0.019f, -0.225f, 0.025f },
    { -0.348f, -0.007f, -0.22f, 0.025f },
    { -0.348f, -0.029f, -0.233f, 0.025f },
    { -0.286f, -0.0f, -0.258f, 0.062f },
    { -0.282f, -0.0f, -0.258f, 0.062f },
    { -0.22f, -0.0f, -0.258f, 0.062f },
    { -0.157f, -0.0f, -0.258f, 0.062f }
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

__device__ __constant__ int fanucm710_sphere_to_joint[112] = {
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
    // link_6's 5 spheres, unchanged.
    7,
    7,
    7,
    7,
    7,
    // tool0's 44 spheres (TCP_5_1), all sharing joint7's transform since tool0 is rigidly
    // fixed to link_6 with no relative DOF.
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

__device__ __constant__ int fanucm710_flattened_joint_to_spheres[120] = {
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
    { 0, 28, 111 },
    { 1, 28, 111 },
    { 2, 28, 111 },
    { 3, 28, 111 },
    { 4, 28, 111 },
    { 5, 28, 111 },
    { 6, 28, 111 },
    { 7, 28, 111 },
    { 8, 28, 111 },
    { 9, 28, 111 },
    { 10, 28, 111 },
    { 11, 28, 111 },
    { 12, 28, 111 },
    { 13, 28, 111 },
    { 14, 28, 111 },
    { 15, 28, 111 },
    { 16, 28, 111 },
    { 17, 28, 111 },
    { 18, 28, 111 },
    { 19, 28, 111 },
    { 20, 28, 111 },
    { 21, 28, 111 },
    { 22, 28, 111 },
    { 23, 28, 111 },
    { 24, 42, 111 },
    { 25, 42, 111 },
    { 26, 42, 111 },
    { 27, 42, 111 },
    { 28, 49, 111 },
    { 29, 49, 111 },
    { 30, 49, 111 },
    { 31, 49, 111 },
    { 32, 49, 111 },
    { 33, 49, 111 },
    { 34, 49, 111 },
    { 35, 49, 111 },
    { 36, 49, 111 },
    { 37, 49, 111 },
    { 38, 49, 111 },
    { 39, 49, 111 },
    { 40, 49, 111 },
    { 41, 49, 111 },
    { 42, 68, 111 },
    { 43, 68, 111 },
    { 44, 68, 111 },
    { 45, 68, 111 },
    { 46, 68, 111 },
    { 47, 68, 111 },
    { 48, 68, 111 },
    { 49, 63, 111 },
    { 50, 63, 111 },
    { 51, 63, 111 },
    { 52, 63, 111 },
    { 53, 63, 111 },
    { 54, 63, 111 },
    { 55, 63, 111 },
    { 56, 63, 111 },
    { 57, 63, 111 },
    { 58, 63, 111 },
    { 59, 68, 111 },
    { 60, 68, 111 },
    { 61, 68, 111 },
    { 62, 68, 111 }
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
