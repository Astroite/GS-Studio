// Reduction operations adapted from Apache-2.0 gsplat1.5.3 Utils.cuh.
#pragma once
#include "Common.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace local_render {
namespace cg=cooperative_groups;
template <uint32_t DIM,class Warp> inline __device__ void warpSum(float* v,Warp& w) {
    #pragma unroll
    for(uint32_t i=0;i<DIM;++i)v[i]=cg::reduce(w,v[i],cg::plus<float>());
}
template <class Warp> inline __device__ void warpSum(float& v,Warp& w) {v=cg::reduce(w,v,cg::plus<float>());}
template <class Warp> inline __device__ void warpSum(vec2& v,Warp& w) {
    v.x=cg::reduce(w,v.x,cg::plus<float>());v.y=cg::reduce(w,v.y,cg::plus<float>());
}
template <class Warp> inline __device__ void warpSum(vec3& v,Warp& w) {
    v.x=cg::reduce(w,v.x,cg::plus<float>());v.y=cg::reduce(w,v.y,cg::plus<float>());v.z=cg::reduce(w,v.z,cg::plus<float>());
}
}
