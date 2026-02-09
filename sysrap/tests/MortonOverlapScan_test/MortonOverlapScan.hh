#pragma once
#include <vector_types.h>
#include <cuda_runtime.h>
struct quad4 ;

struct MortonOverlapScan
{
    /**
     * @param d_intersect   [in]  Input buffer of intersects on GPU
     * @param num_intersect [in]  Count of input intersects
     * @param d_overlap     [out] Newly allocated GPU buffer containing overlaps
     * @param num_overlap   [out] Count of overlaps found
     * @param x0, y0, z0    [in]  Minimum corner of ROI BBox
     * @param x1, y1, z1    [in]  Maximum corner of ROI BBox
     * @param window        [in]  Neighborhood size for identity flicker check
     * @param stream        [in]  CUDA stream for async execution
     */
    static void Scan(
        const quad4* d_intersect,
        size_t        num_intersect,
        quad4** d_overlap,
        size_t* num_overlap,
        float x0, float y0, float z0,
        float x1, float y1, float z1,
        int           window = 4,
        int           focus = -1,
        cudaStream_t  stream = 0
    );
};
