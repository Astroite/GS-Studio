// Private minimal types; derived kernels retain their Apache-2.0 notices.
#pragma once
#include <ATen/ATen.h>
#include <cstdint>
namespace local_render {
struct vec2 { float x,y; };
struct vec3 { float x,y,z; };
#define ALPHA_THRESHOLD (1.f / 255.f)
}
