/**
SPM_dev.cu
===========

SavePartialData
   struct used to communicate into callback

save_partial_callback_async


SPM::save_partial

SPM::load_partial

{
    Partial

    load_partial_for_merge
}

SPM::merge_incremental


TODO: generalize from sphotonlite via template

**/

#include "SPM_dev.hh"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/merge.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <fstream>
#include <vector>
#include <cassert>


/**
SavePartialData
----------------

*d_ptr* is included to allow cleanup

**/

struct SavePartialData
{
    sphotonlite* h_pinned;
    sphotonlite* d_ptr;
    std::string path;
    size_t count;
};


/**
save_partial_callback_async
----------------------------

1. open path and write h_pinned to it
2. free d_ptr and h_pinned

**/


extern "C" void CUDART_CB save_partial_callback_async(cudaStream_t stream, cudaError_t status, void* userData)
{
    (void)stream;
    SavePartialData* data = static_cast<SavePartialData*>(userData);
    if (status != cudaSuccess) {
        fprintf(stderr, "save_partial_callback: CUDA error %d\n", status);
    }

    // 1. open path and write h_pinned to it   (HMM: could instead save to .npy)

    std::ofstream f(data->path, std::ios::binary);
    if (f) {
        f.write(reinterpret_cast<const char*>(data->h_pinned), data->count * sizeof(sphotonlite));
    } else {
        fprintf(stderr, "save_partial: failed to open %s\n", data->path.c_str());
    }

    // 2. free d_ptr and h_pinned

    cudaFreeAsync(data->h_pinned, stream);
    cudaFree(data->d_ptr);
    delete data;
}


extern "C" void CUDART_CB save_partial_callback_templated(cudaStream_t stream, cudaError_t status, void* userData)
{
    auto* payload = static_cast<std::function<void(cudaStream_t,cudaError_t)>*>(userData);

    (*payload)(stream, status);  // invoke the real lambda/function

    delete payload;  // clean up
}

template <typename T>
void launch_save_async(cudaStream_t stream, const T* d_ptr, size_t count, const std::string& path)
{
    T* h_pinned = nullptr;
    cudaMallocHost(&h_pinned, count * sizeof(T));

    cudaMemcpyAsync(h_pinned, d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost, stream);

    // Capture everything we need in a lambda
    auto callback_body = [h_pinned, d_ptr, count, path]
                         (cudaStream_t s, cudaError_t status) mutable {
        if (status == cudaSuccess) {
            std::ofstream f(path, std::ios::binary);
            if (f) f.write(reinterpret_cast<char*>(h_pinned), count * sizeof(T));
        } else {
            fprintf(stderr, "Save failed: %s\n", cudaGetErrorString(status));
        }
        cudaFreeHost(h_pinned);
        // cudaFree(d_ptr);  // if you own it
    };

    // Heap-allocate the lambda (type-erased)
    auto* boxed = new std::function<void(cudaStream_t,cudaError_t)>(std::move(callback_body));

    cudaStreamAddCallback(stream, save_partial_callback_templated, boxed, 0);
}




/**
SPM_dev::save_partial_sphotonlite
------------------------------------

1. host allocate *h_pinned* space for *count* hits
2. async copy count hits to h_pinned from d_partial
3. setup callback that saves to file after copy completes then cleans up

Replaced ancient cudaMallocHost with cudaMallocAsync as the later is
memlock limited by "ulimit -l"  (default 8MB).

**/

void SPM_dev::save_partial_sphotonlite(
        const sphotonlite* d_partial,
        size_t count,
        const std::string& path,
        cudaStream_t stream )
{
    if (count == 0 || !d_partial) {
        cudaFree(const_cast<sphotonlite*>(d_partial));
        return;
    }

    sphotonlite* h_pinned;
    cudaMallocAsync(&h_pinned, count * sizeof(sphotonlite), stream);

    cudaMemcpyAsync(h_pinned, d_partial, count * sizeof(sphotonlite),
                    cudaMemcpyDeviceToHost, stream);

    SavePartialData* cb_data = new SavePartialData{h_pinned, const_cast<sphotonlite*>(d_partial), path, count};
    cudaStreamAddCallback(stream, save_partial_callback_async, cb_data, 0);
}


template<typename T>
void SPM_dev::save_partial(
        const T* d_partial,
        size_t count,
        const std::string& path,
        cudaStream_t stream )
{
    if (count == 0 || !d_partial) {
        cudaFree(const_cast<T*>(d_partial));
        return;
    }
    launch_save_async<T>(stream, d_partial, count, path);
}





/**
SPM_dev::load_partial
----------------------

1. open *path* and use tellg to determine *n*
2. read from file into *h* vector of sphotonlite
3. allocate *d* and async copy from *h* to *d*
4. set output args `*d_out` and `count`

**/

void SPM_dev::load_partial_sphotonlite(
        const std::string& path,
        sphotonlite**      d_out,
        size_t*            count,
        cudaStream_t       stream )
{

    // 1. open *path* and use tellg to determine *n*

    std::ifstream f(path, std::ios::binary | std::ios::ate);
    assert(f && "failed to open partial");
    size_t bytes = f.tellg();
    size_t n = bytes / sizeof(sphotonlite);
    f.seekg(0);

    // 2. read from file into *h* vector of sphotonlite

    std::vector<sphotonlite> h(n);
    f.read((char*)h.data(), bytes);

    // 3. allocate *d* and async copy from *h* to *d*

    sphotonlite* d = nullptr;
    cudaMallocAsync(&d, n * sizeof(sphotonlite), stream);
    cudaMemcpyAsync(d, h.data(), n * sizeof(sphotonlite),
                    cudaMemcpyHostToDevice, stream);

    // 4. set output args `*d_out` and `count`

    *d_out = d;
    if (count) *count = (int)n;
}

// ------------------------------------------------------------------
// merge_incremental – FULLY IMPLEMENTED
// ------------------------------------------------------------------


namespace
{
    struct Partial
    {
        uint64_t*     keys;
        sphotonlite*  hits;
        size_t        count;
    };

    /**
    load_partial_for_merge
    ----------------------

    1. open path, use tellg to determine number of hits
    2. read into h sphotonlite vector
    3. device alloc d_hits and d_keys
    4. copy to d_hits from h
    5. populate d_keys with bitwise combined (id,timebucket), OR for time_window 0.f just (id,)
    6. sort_by_key
    7. return Partial struct

    HUH, is that sort_by_key needed ? SPM::merge_partial_select does sort_by_key then reduce_by_key
    so should already be sorted ?

    **/


    Partial load_partial_for_merge(const char* path, float time_window, cudaStream_t stream)
    {

        // 1. open path with pointer at-end "ate", use tellg to determine number of hits

        std::ifstream f(path, std::ios::binary | std::ios::ate);
        assert(f && "failed to open partial");
        size_t bytes = f.tellg();
        size_t n = bytes / sizeof(sphotonlite);
        f.seekg(0);

        // 2. read into h sphotonlite vector

        std::vector<sphotonlite> h(n);
        f.read((char*)h.data(), bytes);

        // 3. device alloc d_hits and d_keys

        sphotonlite* d_hits = nullptr;
        uint64_t*     d_keys = nullptr;
        cudaMallocAsync(&d_hits, n * sizeof(sphotonlite), stream);
        cudaMallocAsync(&d_keys, n * sizeof(uint64_t),    stream);

        // 4. copy to d_hits from h

        cudaMemcpyAsync(d_hits, h.data(), n * sizeof(sphotonlite),
                        cudaMemcpyHostToDevice, stream);

        auto policy = thrust::cuda::par_nosync.on(stream);

        // 5. populate d_keys with bitwise combined (id,timebucket), OR for time_window 0.f just (id,)

        if (time_window == 0.f)
        {
            // Use identity as unique key → no merging
            thrust::transform(policy,
                thrust::device_ptr<const sphotonlite>(d_hits),
                thrust::device_ptr<const sphotonlite>(d_hits + n),
                thrust::device_ptr<uint64_t>(d_keys),
                [] __device__ (const sphotonlite& p) -> uint64_t { return p.identity(); }
                );
        }
        else
        {
            auto key_gen = [=] __device__ (const sphotonlite& p) -> uint64_t {
                unsigned id = p.identity() & 0xFFFFu;
                unsigned bk = unsigned(p.time / time_window);
                return (uint64_t(id) << 48) | bk;
            };
            thrust::transform(policy,
                thrust::device_ptr<const sphotonlite>(d_hits),
                thrust::device_ptr<const sphotonlite>(d_hits + n),
                thrust::device_ptr<uint64_t>(d_keys),
                key_gen);
        }


        // 6. sort_by_key

        thrust::sort_by_key(policy,
            thrust::device_ptr<uint64_t>(d_keys),
            thrust::device_ptr<uint64_t>(d_keys + n),
            thrust::device_ptr<sphotonlite>(d_hits));

        // 7. return Partial struct

        return {d_keys, d_hits, n};
    }
}

/**
SPM_dev::merge_incremental
----------------------------

1. load add partial from first path
2. load next partial from subsequent path
3. device allocate for acc.count + next.count
4. merge_by_key combining (acc.keys,acc.hits) and (next.keys,next.hits) into (d_merge_key,d_merge_val)
5. obtain merged_size (HUH: should always be same as the total?)

6. [tw==0.f] alloc merged_size items (d_reduced_key,d_reduced_val)
7. [tw==0.f] copy to d_reduced_key/val from d_merge_key/val

6. [tw!=0.f] alloc total items (d_reduced_key,d_reduced_val)
7. [tw!=0.f] reduce_by_key combining contiguous equal (id,timebucket) hits into (d_reduced_key,d_reduced_val)

8. free (acc.keys,acc.hits) (d_merge_keys,d_merge_val) (next.keys,next.hits)
9. change (acc.keys,acc.hits,acc_count) to (d_reduced_key,d_reduced_val,reduced_count)

10. repeat from 2. until all paths loaded and hits merged
11. set d_final and final_count, free acc.keys

**/

void SPM_dev::merge_incremental_sphotonlite(
        const char** partial_paths,
        sphotonlite** d_final,
        size_t*       final_count,
        float         time_window,
        cudaStream_t  stream )
{
    if (!partial_paths[0]) {
        *d_final = nullptr;
        if (final_count) *final_count = 0;
        return;
    }

    auto policy = thrust::cuda::par_nosync.on(stream);

    // 1. load add partial from first path

    Partial acc = load_partial_for_merge(partial_paths[0], time_window, stream);
    int i = 1;

    using reduce_op = typename sphotonlite::reduce_op;


    while (partial_paths[i] != nullptr)
    {
        // 2. load next partial from subsequent path
        Partial next = load_partial_for_merge(partial_paths[i], time_window, stream);
        ++i;

        // 3. device allocate for acc.count + next.count
        size_t total = size_t(acc.count) + next.count;

        uint64_t*     d_merge_key = nullptr;
        sphotonlite*  d_merge_val = nullptr;
        cudaMallocAsync(&d_merge_key, total * sizeof(uint64_t),   stream);
        cudaMallocAsync(&d_merge_val, total * sizeof(sphotonlite), stream);

        // 4. merge_by_key combining (acc.keys,acc.hits) and (next.keys,next.hits) into (d_merge_key,d_merge_val)

        auto merge_end = thrust::merge_by_key(
                policy,
                thrust::device_ptr<uint64_t>(acc.keys),   thrust::device_ptr<uint64_t>(acc.keys + acc.count),
                thrust::device_ptr<uint64_t>(next.keys),  thrust::device_ptr<uint64_t>(next.keys + next.count),
                thrust::device_ptr<sphotonlite>(acc.hits),
                thrust::device_ptr<sphotonlite>(next.hits),
                thrust::device_ptr<uint64_t>(d_merge_key),
                thrust::device_ptr<sphotonlite>(d_merge_val));

        // 5. obtain merged_size (HUH: should always be same as the total?)

        size_t merged_size = merge_end.first - thrust::device_ptr<uint64_t>(d_merge_key);

        uint64_t*     d_reduced_key = nullptr;
        sphotonlite*  d_reduced_val = nullptr;
        size_t        reduced_count;

        if (time_window == 0.f)
        {
            // No reduction → just copy
            // 6. [tw==0.f] alloc merged_size items (d_reduced_key,d_reduced_val)

            cudaMallocAsync(&d_reduced_key, merged_size * sizeof(uint64_t),   stream);
            cudaMallocAsync(&d_reduced_val, merged_size * sizeof(sphotonlite), stream);

            // 7. [tw==0.f] copy to d_reduced_key/val from d_merge_key/val

            cudaMemcpyAsync(d_reduced_key, d_merge_key, merged_size * sizeof(uint64_t),
                            cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(d_reduced_val, d_merge_val, merged_size * sizeof(sphotonlite),
                            cudaMemcpyDeviceToDevice, stream);
            reduced_count = merged_size;
        }
        else
        {

            // 6. [tw!=0.f] alloc total items (d_reduced_key,d_reduced_val)

            cudaMallocAsync(&d_reduced_key, total * sizeof(uint64_t),   stream);
            cudaMallocAsync(&d_reduced_val, total * sizeof(sphotonlite), stream);

            // 7. [tw!=0.f] reduce_by_key combining contiguous equal (id,timebucket) hits into (d_reduced_key,d_reduced_val)
            auto reduce_end = thrust::reduce_by_key(
                    policy,
                    thrust::device_ptr<uint64_t>(d_merge_key),
                    thrust::device_ptr<uint64_t>(d_merge_key + merged_size),
                    thrust::device_ptr<sphotonlite>(d_merge_val),
                    thrust::device_ptr<uint64_t>(d_reduced_key),
                    thrust::device_ptr<sphotonlite>(d_reduced_val),
                    thrust::equal_to<uint64_t>(),
                    reduce_op{} );

            reduced_count = reduce_end.first - thrust::device_ptr<uint64_t>(d_reduced_key);
        }

        // 8. free (acc.keys,acc.hits) (d_merge_keys,d_merge_val) (next.keys,next.hits)

        cudaFreeAsync(acc.keys, stream);
        cudaFreeAsync(acc.hits, stream);
        cudaFreeAsync(d_merge_key, stream);
        cudaFreeAsync(d_merge_val, stream);
        cudaFreeAsync(next.keys, stream);
        cudaFreeAsync(next.hits, stream);

        // 9. change (acc.keys,acc.hits,acc_count) to (d_reduced_key,d_reduced_val,reduced_count)
        acc.keys  = d_reduced_key;
        acc.hits  = d_reduced_val;
        acc.count = reduced_count;
    }
    // 10. repeat from 2. until all paths loaded and hits merged
    // 11. set d_final and final_count, free acc.keys

    *final_count = acc.count;
    *d_final     = acc.hits;
    cudaFreeAsync(acc.keys, stream);  // keys no longer needed
}



