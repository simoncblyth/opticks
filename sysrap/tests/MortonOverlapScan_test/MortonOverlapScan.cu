
#include "scuda.h"
#include "squad.h"
#include "MortonOverlapScan.hh"

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/cuda/execution_policy.h>

// Helper to expand 21 bits into 64-bit slots for interleaving
__device__ uint64_t expandBits(uint32_t v)
{
    uint64_t x = v & 0x1fffff;
    x = (x | x << 32) & 0x1f00000000ffffL;
    x = (x | x << 16) & 0x1f0000ff0000ffL;
    x = (x | x << 8)  & 0x100f00f00f00f00fL;
    x = (x | x << 4)  & 0x10c30c30c30c30c3L;
    x = (x | x << 2)  & 0x1249249249249249L;
    return x;
}



/**

             |
     prev_id |  self_id  off_id[0] off_id[1] ...
             |

**/

struct OverlapPredicate
{
    const quad4* data;
    int n;
    int window;

    __device__ bool operator()(int i) const {
        if (i < window || i >= n - window) return false;

        int prev_id = data[i-1].simtrace_globalPrimIdx();
        int self_id = data[i].simtrace_globalPrimIdx();
        if (self_id == prev_id) return false;

        float3 n_self = make_float3(data[i].q0.f.x, data[i].q0.f.y, data[i].q0.f.z);

        for (int off = 1; off <= window; ++off)
        {
            int off_id = data[i + off].simtrace_globalPrimIdx();

            // If the identity we just came from (prev_id) reappears ahead,
            // the current point (self_id) is an 'intruder' sandwiched between points of prev_id.
            if (off_id == prev_id)
            {
                float3 n_off = make_float3(data[i + off].q0.f.x, data[i + off].q0.f.y, data[i + off].q0.f.z);

                // Normal Check: Compare 'self' (intruder) vs 'lookahead' (host volume)
                float dot = n_self.x * n_off.x +
                            n_self.y * n_off.y +
                            n_self.z * n_off.z;

                // Touching: dot ~ -1.0 (Normals point away from each other)
                // Overlap: dot > -0.9 (Surface orientations are not perfectly back-to-back)
                if (dot > -0.98f) {
                    return true;
                }
            }
        }
        return false;
    }
};


void MortonOverlapScan::Scan(
    const quad4* d_intersect,
    size_t num_intersect,
    quad4** d_overlap,
    size_t* num_overlap,
    float x0, float y0, float z0,
    float x1, float y1, float z1,
    int window,
    cudaStream_t stream
) {
    auto policy = thrust::cuda::par.on(stream);
    thrust::device_ptr<const quad4> i_ptr(d_intersect);

    // 1. Subset points within the input BBox
    thrust::device_vector<quad4> subset(num_intersect);
    auto subset_end = thrust::copy_if(policy, i_ptr, i_ptr + num_intersect, subset.begin(),
        [=] __device__ (const quad4& q) {
            return q.q1.f.x >= x0 && q.q1.f.x <= x1 &&
                   q.q1.f.y >= y0 && q.q1.f.y <= y1 &&
                   q.q1.f.z >= z0 && q.q1.f.z <= z1;
        });
    size_t n_subset = thrust::distance(subset.begin(), subset_end);
    subset.resize(n_subset);

    if (n_subset == 0) {
        *num_overlap = 0;
        *d_overlap = nullptr;
        return;
    }

    // 2. Compute Morton Codes normalized to the BBox
    thrust::device_vector<uint64_t> morton(n_subset);

    // 64 = 21*3 + 1, so 21 bits for x,y,z with one bit spare
    float mx21 =  2097151.0f ; // 2^21-1 : largest integer that can be represented by 21 bits
    float3 scale = { mx21 / (x1 - x0), mx21 / (y1 - y0), mx21 / (z1 - z0) };

    thrust::transform(policy, subset.begin(), subset.end(), morton.begin(),
        [=] __device__ (const quad4& q) {
            uint32_t ux = (uint32_t)((q.q1.f.x - x0) * scale.x);
            uint32_t uy = (uint32_t)((q.q1.f.y - y0) * scale.y);
            uint32_t uz = (uint32_t)((q.q1.f.z - z0) * scale.z);
            return (expandBits(ux) << 2) | (expandBits(uy) << 1) | expandBits(uz);
        });

    // 3. Sort subset by Morton Code (brings spatially close points together in 1D)
    thrust::sort_by_key(policy, morton.begin(), morton.end(), subset.begin());

    // 4. Stencil copy_if to find identity "flicker"
    thrust::device_vector<quad4> results(n_subset);
    thrust::counting_iterator<int> idx_first(0);

    OverlapPredicate  pred{ thrust::raw_pointer_cast(subset.data()), (int)n_subset, window };

    auto results_end = thrust::copy_if(policy, subset.begin(), subset.end(), idx_first, results.begin(), pred);
    *num_overlap = thrust::distance(results.begin(), results_end);

    // 5. Output Allocation
    if (*num_overlap > 0) {
        cudaMalloc(d_overlap, (*num_overlap) * sizeof(quad4));
        thrust::copy(policy, results.begin(), results_end, thrust::device_ptr<quad4>(*d_overlap));
    } else {
        *d_overlap = nullptr;
    }
}
