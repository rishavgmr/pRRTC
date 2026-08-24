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
            if (sphere_sphere_self_collision(
                sphere_1[0], sphere_1[1], sphere_1[2], fanucm710_approx_spheres_array[sphere_1_ind].w,
                sphere_2[0], sphere_2[1], sphere_2[2], fanucm710_approx_spheres_array[j].w
            )){
                atomicAdd((int*)&joint_in_collision[20*batch_ind + fanucm710_approx_sphere_to_joint[sphere_1_ind]], 1);
                out = false;
            }
        } 
    }
    return out;
}

// 4 threads per discretized motion for env collision check
template <>
__device__ bool env_collision_check_approx<ppln::robots::Fanucm710>(volatile float* sphere_pos_approx, volatile int* joint_in_collision, ppln::collision::Environment<float> *env, const int tid){
    const int thread_ind = tid % 4;
    const int batch_ind = tid / 4;
    bool out = true;

    for (int i = thread_ind; i < FANUCM710_APPROX_SPHERE_COUNT; i += 4){
        // sphere i, robot batch_ind (32 robots)
        if ( 
            sphere_environment_in_collision(
                env,
                sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 2],
                fanucm710_approx_spheres_array[i].w
            )
        ) {
            atomicAdd((int*)&joint_in_collision[20*batch_ind + fanucm710_approx_sphere_to_joint[i]],1);
            out = false;
        } 
    }
    return out;
}




#define FANUCM710_SPHERE_COUNT 84
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

__device__ __constant__ float4 fanucm710_spheres_array[84] = {
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
    { 0.035f, 0.201975f, -0.268019f, 0.08f },
    { 0.018f, 0.262991f, -0.094024f, 0.097f },
    { 0.064f, -0.206008f, -0.085981f, 0.12f },
    { 0.018f, -0.204013f, -0.139981f, 0.096f },
    { -0.26f, -0.002024f, -0.263f, 0.085f },
    { -0.057f, -0.20801f, -0.105981f, 0.11f },
    { 0.06f, 0.019971f, -0.317002f, 0.07f },
    { -0.123f, -0.002024f, -0.262f, 0.08f },
    { 0.043f, 0.289979f, -0.232027f, 0.069f },
    { 0.041f, -0.036012f, -0.127997f, 0.12f },
    { -0.038f, 0.026988f, -0.128002f, 0.063f },
    { 0.098f, 0.060989f, -0.124006f, 0.059f },
    { -0.003f, 0.035976f, -0.258003f, 0.067f },
    { -0.016f, 0.234996f, -0.039022f, 0.12f },
    { 0.063f, -0.033026f, -0.281997f, 0.067f },
    { -0.041f, 0.289979f, -0.232027f, 0.069f }
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

__device__ __constant__ int fanucm710_sphere_to_joint[84] = {
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
    7
};

__device__ __constant__ int fanucm710_flattened_joint_to_spheres[92] = {
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
    { 0, 28, 83 },
    { 1, 28, 83 },
    { 2, 28, 83 },
    { 3, 28, 83 },
    { 4, 28, 83 },
    { 5, 28, 83 },
    { 6, 28, 83 },
    { 7, 28, 83 },
    { 8, 28, 83 },
    { 9, 28, 83 },
    { 10, 28, 83 },
    { 11, 28, 83 },
    { 12, 28, 83 },
    { 13, 28, 83 },
    { 14, 28, 83 },
    { 15, 28, 83 },
    { 16, 28, 83 },
    { 17, 28, 83 },
    { 18, 28, 83 },
    { 19, 28, 83 },
    { 20, 28, 83 },
    { 21, 28, 83 },
    { 22, 28, 83 },
    { 23, 28, 83 },
    { 24, 42, 83 },
    { 25, 42, 83 },
    { 26, 42, 83 },
    { 27, 42, 83 },
    { 28, 49, 83 },
    { 29, 49, 83 },
    { 30, 49, 83 },
    { 31, 49, 83 },
    { 32, 49, 83 },
    { 33, 49, 83 },
    { 34, 49, 83 },
    { 35, 49, 83 },
    { 36, 49, 83 },
    { 37, 49, 83 },
    { 38, 49, 83 },
    { 39, 49, 83 },
    { 40, 49, 83 },
    { 41, 49, 83 },
    { 42, 68, 83 },
    { 43, 68, 83 },
    { 44, 68, 83 },
    { 45, 68, 83 },
    { 46, 68, 83 },
    { 47, 68, 83 },
    { 48, 68, 83 },
    { 49, 63, 83 },
    { 50, 63, 83 },
    { 51, 63, 83 },
    { 52, 63, 83 },
    { 53, 63, 83 },
    { 54, 63, 83 },
    { 55, 63, 83 },
    { 56, 63, 83 },
    { 57, 63, 83 },
    { 58, 63, 83 },
    { 59, 68, 83 },
    { 60, 68, 83 },
    { 61, 68, 83 },
    { 62, 68, 83 }
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
            if (sphere_sphere_self_collision(
                sphere_1[0], sphere_1[1], sphere_1[2], fanucm710_spheres_array[sphere_1_ind].w,
                sphere_2[0], sphere_2[1], sphere_2[2], fanucm710_spheres_array[j].w
            )){
                //return false;
                has_collision=true;
            }
        }
    }
    return !has_collision;

}

// 4 threads per discretized motion for env collision check
template <>
__device__ bool env_collision_check<ppln::robots::Fanucm710>(volatile float* sphere_pos, volatile int* joint_in_collision, ppln::collision::Environment<float> *env, const int tid){
    const int thread_ind = tid % 4;
    const int batch_ind = tid / 4;
    bool has_collision=false;

    for (int i = thread_ind; i < FANUCM710_SPHERE_COUNT-FANUCM710_SPHERE_COUNT%4; i += 4){
        // sphere i, robot batch_ind (16 robots)
        if (joint_in_collision[20*batch_ind + fanucm710_sphere_to_joint[i]] > 0 && 
            sphere_environment_in_collision(
                env,
                sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 2],
                fanucm710_spheres_array[i].w
            )
        ) {
            //return false;
            has_collision=true;
        } 
        if (warp_any_full_mask(has_collision)) return false;
    }
    int i=FANUCM710_SPHERE_COUNT-1-thread_ind;
    if (joint_in_collision[20*batch_ind + fanucm710_sphere_to_joint[i]] > 0 && 
        sphere_environment_in_collision(
            env,
            sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
            sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
            sphere_pos[i * BATCH_SIZE * 3 + batch_ind * 3 + 2],
            fanucm710_spheres_array[i].w
        )
    ) {
        return false;
    } 
    return true;
}
}

namespace ppln::robots {

// Real definition of the extern arrays declared in Robots.hh's Fanucm710 struct (RSW-2740) - see
// that declaration's own comment for why this needs to live here (the one file this header's
// arrays' matching TU includes) rather than in Robots.hh directly. Defaults are Collins' own
// case: dof 0 (rail) from its rail length; dofs 1-6 from robot.urdf's hardware limits. Both get
// overwritten by uploadJointLimits() before any solve() call runs, for every case including
// Collins, so these compile-time values are in practice never read. Mirrors
// fanuc_m710_benchmark.cuh's own copy of this same block exactly.
__device__ __constant__ float fanucm710_dof_s_m[7] = {
    3.0f, 6.2831854820251465f, 3.9269907474517822f, 6.457718372344971f, 12.566370964050293f, 4.363323211669922f, 12.566370964050293f
};
__device__ __constant__ float fanucm710_dof_s_a[7] = {
    0.0f, -3.1415927410125732f, -1.5707963705062866f, -1.5707963705062866f, -6.2831854820251465f, -2.181661605834961f, -6.2831854820251465f
};

}
