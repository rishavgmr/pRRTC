#pragma once

#include <array>
#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>


namespace ppln::robots {
    struct Panda
    {
        static constexpr auto name = "panda";
        static constexpr auto dimension = 7;
        using Configuration = std::array<float, dimension>;

        // necessary to generate the scale_cfg function at compile time
        __device__ static constexpr float get_s_m(int i) {
            constexpr float values[] = {
                5.9342f,
                3.6652f,
                5.9342f,
                3.2289f,
                5.9342f,
                3.9095999999999997f,
                5.9342f
            };
            return values[i];
        }
        
        __device__ static constexpr float get_s_a(int i) {
            constexpr float values[] = {
                -2.9671f,
                -1.8326f,
                -2.9671f,
                -3.1416f,
                -2.9671f,
                -0.0873f,
                -2.9671f
            };
            return values[i];
        }

        inline static void print_robot_config(Configuration &cfg) {
            for (int i = 0; i < dimension; i++) {
                std::cout << cfg[i] << ' ';
            }
            std::cout << '\n';
        };
        
        // template metaprogramming to generate the scale_cfg function
        template<size_t I = 0>
        __device__ __forceinline__ static void scale_cfg_impl(float *q)
        {
            if constexpr (I < dimension) {
                q[I] = q[I] * get_s_m(I) + get_s_a(I);
                scale_cfg_impl<I + 1>(q);
            }
        }

        __device__ __forceinline__ static void scale_cfg(float *q)
        {
            scale_cfg_impl(q);
        }
    };

    struct Fetch
    {
        static constexpr auto name = "fetch";
        static constexpr auto dimension = 8;
        using Configuration = std::array<float, dimension>;

        __device__ static constexpr float get_s_m(int i) {
            constexpr float values[] = {
                0.38615f,
                3.2112f,
                2.739f,
                6.28318f,
                4.502f,
                6.28318f,
                4.32f,
                6.28318f
            };
            return values[i];
        }
        
        __device__ static constexpr float get_s_a(int i) {
            constexpr float values[] = {
                0.0f,
                -1.6056f,
                -1.221f,
                -3.14159f,
                -2.251f,
                -3.14159f,
                -2.16f,
                -3.14159f
            };
            return values[i];
        }
        
        inline static void print_robot_config(Configuration &cfg) {
            for (int i = 0; i < dimension; i++) {
                std::cout << cfg[i] << ' ';
            }
            std::cout << '\n';
        };
        
        template<size_t I = 0>
        __device__ __forceinline__ static void scale_cfg_impl(float *q)
        {
            if constexpr (I < dimension) {
                q[I] = q[I] * get_s_m(I) + get_s_a(I);
                scale_cfg_impl<I + 1>(q);
            }
        }

        __device__ __forceinline__ static void scale_cfg(float *q)
        {
            scale_cfg_impl(q);
        }
    };

    struct Baxter
    {
        static constexpr auto name = "baxter";
        static constexpr auto dimension = 14;
        using Configuration = std::array<float, dimension>;

        __device__ static constexpr float get_s_m(int i) {
            constexpr float values[] = {
                3.40335987756,
                3.194,
                6.10835987756,
                2.6679999999999997,
                6.118,
                3.66479632679,
                6.118,
                3.40335987756,
                3.194,
                6.10835987756,
                2.6679999999999997,
                6.118,
                3.66479632679,
                6.118
            };
            return values[i];
        }
        
        __device__ static constexpr float get_s_a(int i) {
            constexpr float values[] = {
                -1.70167993878,
                -2.147,
                -3.05417993878,
                -0.05,
                -3.059,
                -1.57079632679,
                -3.059,
                -1.70167993878,
                -2.147,
                -3.05417993878,
                -0.05,
                -3.059,
                -1.57079632679,
                -3.059
            };
            return values[i];
        }

        inline static void print_robot_config(Configuration &cfg) {
            for (int i = 0; i < dimension; i++) {
                std::cout << cfg[i] << ' ';
            }
            std::cout << '\n';
        };
        
        template<size_t I = 0>
        __device__ __forceinline__ static void scale_cfg_impl(float *q)
        {
            if constexpr (I < dimension) {
                q[I] = q[I] * get_s_m(I) + get_s_a(I);
                scale_cfg_impl<I + 1>(q);
            }
        }

        __device__ __forceinline__ static void scale_cfg(float *q)
        {
            scale_cfg_impl(q);
        }
    };

    struct Sphere
    {
        static constexpr auto name = "sphere";
        static constexpr auto dimension = 3;
        static constexpr float radius = 1.0;
        using Configuration = std::array<float, dimension>;

        // necessary to generate the scale_cfg function at compile time
        __device__ static constexpr float get_s_m(int i) {
            constexpr float values[] = {
                10.0f,
                10.0f,
                10.0f
            };
            return values[i];
        }
        
        __device__ static constexpr float get_s_a(int i) {
            constexpr float values[] = {
                -5.0f,
                -5.0f,
                -5.0f
            };
            return values[i];
        }

        inline static void print_robot_config(Configuration &cfg) {
            for (int i = 0; i < dimension; i++) {
                std::cout << cfg[i] << ' ';
            }
            std::cout << '\n';
        };

        // template metaprogramming to generate the scale_cfg function
        template<size_t I = 0>
        __device__ __forceinline__ static void scale_cfg_impl(float *q)
        {
            if constexpr (I < dimension) {
                q[I] = q[I] * get_s_m(I) + get_s_a(I);
                scale_cfg_impl<I + 1>(q);
            }
        }

        __device__ __forceinline__ static void scale_cfg(float *q)
        {
            scale_cfg_impl(q);
        }
    };

    // Per-DOF sampling scale/offset for Fanucm710 (RSW-2740) - q[i]_sampled = q[i]_[0,1] * s_m[i] +
    // s_a[i], i.e. [s_a[i], s_a[i]+s_m[i]] is the range every random Halton sample gets mapped
    // into for that dof, so this is what bounds where EXTEND can ever place a node. Runtime-
    // overridable (not the constexpr arrays this used to be) because these bounds are case-
    // specific, not robot-level: dof 0 (the rail) is scene_config's own axisLowerLimits/
    // axisUpperLimits for whichever axis is active, and dofs 1-6 are scene_config's
    // jointLowerLimits/jointUpperLimits (the same source Manipulator::modifyChains() reads,
    // matching what P2P itself treats as valid - not robot.urdf's raw hardware limits) - see
    // uploadJointLimits(), pRRTC_benchmark.cu.
    //
    // Declared extern (no storage, no initializer) here on purpose: this header is included by
    // both pRRTC_benchmark.cu AND scripts/benchmark_fanuc_m710.cpp (a plain .cpp, via
    // Planners.hh), and a real namespace-scope __constant__ definition in a header gets full
    // external-linkage storage in every translation unit that includes it - unlike the
    // static-constexpr-inside-a-function form get_s_m/get_s_a used to have (internal linkage,
    // safe to repeat per-TU), giving a "multiple definition" link error the moment a second TU
    // includes this file. The one real definition (with Collins' own values as the compile-time
    // default) lives in fanuc_m710_benchmark.cuh, which only the .cu ever includes.
    extern __device__ __constant__ float fanucm710_dof_s_m[7];
    extern __device__ __constant__ float fanucm710_dof_s_a[7];

    struct Fanucm710{

    static constexpr auto name = "fanucm710";
    static constexpr std::size_t dimension = 7;
    using Configuration = std::array<float, dimension>;

    __device__ static float get_s_m(int i) {
        return fanucm710_dof_s_m[i];
    }

    __device__ static float get_s_a(int i) {
        return fanucm710_dof_s_a[i];
    }

    template<size_t I = 0>
    __device__ __forceinline__ static void scale_cfg_impl(float *q)
    {
        if constexpr (I < dimension) {
            q[I] = q[I] * get_s_m(I) + get_s_a(I);
            scale_cfg_impl<I + 1>(q);
        }
    }

    __device__ __forceinline__ static void scale_cfg(float *q)
    {
        scale_cfg_impl(q);
    }

    inline static void print_robot_config(Configuration &cfg) {
        for (int i = 0; i < dimension; i++) {
            std::cout << cfg[i] << ' ';
        }
        std::cout << '\n';
    };
};
}