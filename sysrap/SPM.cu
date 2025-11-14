/**
SPM.cu
=======

Grok generated code under test

sphotonlite_key_functor
    returns sentinel ~0ull or bitwise combination of id(16bits) and timebucket(48bits)

sphotonlite_reduce_op
    merge two hits

sphotonlite_select_pred
    predicate just selecting hits based on flagmask

SPM::merge_partial_select
    Flagmask select hits from input photons and merge the hits by (id,timebucket)
    with incremented counts. Obviously this requires the photons and hits to
    simultaneously fit into VRAM.

SavePartialData
   struct used to communicate into callback

save_partial_callback



**/

#include "SPM.hh"
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


using namespace thrust::placeholders;

/**
sphotonlite_key_functor
-------------------------

Returns sentinel ~0ull if not selected, otherwise bitwise combination
of id(16bits) and bucket(48bits)

HMM: I expect 16 bits for the  bucket would be enough
**/


struct sphotonlite_key_functor
{
    float    tw;
    unsigned select_mask;

    __device__ uint64_t operator()(const sphotonlite& p) const
    {
        if ((p.flagmask & select_mask) == 0) return ~0ull;
        unsigned id = p.identity() & 0xFFFFu;
        unsigned bucket = static_cast<unsigned>(p.time / tw);
        return (uint64_t(id) << 48) | uint64_t(bucket);
    }
};

struct sphotonlite_reduce_op
{
    __device__ sphotonlite operator()(const sphotonlite& a, const sphotonlite& b) const {
        sphotonlite r = a;
        r.time = fminf(a.time, b.time);
        r.flagmask |= b.flagmask;
        unsigned hc = a.hitcount() + b.hitcount();
        r.set_hitcount_identity(hc, a.identity());
        return r;
    }
};

struct sphotonlite_select_pred
{
    unsigned mask;
    __device__ bool operator()(const sphotonlite& p) const { return (p.flagmask & mask) != 0; }
};

/**
SPM::merge_partial_select
-------------------------

Flagmask select hits from input photons and merge the hits by (id,timebucket)
with incremented counts. Obviously this requires the photons and hits to
simultaneously fit into VRAM.

1. special case time_window:0.f to just select without merging, using count_if, allocate, copy_if
2. allocate `(uint64_t*)d_keys` and `(sphotonlite*)d_vals`
3. populate d_keys using sphotonlite_key_functor and d_vals using copy_n
4. sort_by_key arranging hits with same (id, timebucket) together
5. allocate d_out_key d_out_val with space for num_in [HMM, COULD DOUBLE PASS TO REDUCE MEMORY PERHAPS?]
6. reduce_by_key merging contiguous equal (id,timebucket) hits
7. get number of merged hits, potentially including a non-hitmask selected sentinel key ~0ull
8. remove trailing sentinel (~0ull) if present
9. allocate d_final to fit merged hits, d2d copy d_final from d_out_val
10. free temporary buffers
11. set d_out and num_out

**/


void SPM::merge_partial_select(
    const sphotonlite*  d_in,
    int               num_in,
    sphotonlite**      d_out,
    int*             num_out,
    unsigned select_flagmask,
    float        time_window,
    cudaStream_t      stream )
{
    if (num_in == 0) { *d_out = nullptr; if (num_out) *num_out = 0; return; }

    auto policy = thrust::cuda::par_nosync.on(stream);
    auto in = thrust::device_ptr<const sphotonlite>(d_in);

    // 1. special case time_window:0.f to just select without merging, using count_if, allocate, copy_if

    if (time_window == 0.f)
    {
        sphotonlite_select_pred pred{select_flagmask};
        int kept = thrust::count_if(policy, in, in + num_in, pred);
        sphotonlite* d_filtered = nullptr;
        if (kept > 0) {
            cudaMallocAsync(&d_filtered, kept * sizeof(sphotonlite), stream);
            thrust::copy_if(policy, in, in + num_in,
                            thrust::device_ptr<sphotonlite>(d_filtered), pred);
        }
        *d_out = d_filtered;
        if (num_out) *num_out = kept;
        return;
    }


    // 2. allocate `(uint64_t*)d_keys` and `(sphotonlite*)d_vals`

    uint64_t* d_keys = nullptr;
    sphotonlite* d_vals = nullptr;
    cudaMallocAsync(&d_keys, num_in * sizeof(uint64_t),   stream);
    cudaMallocAsync(&d_vals, num_in * sizeof(sphotonlite), stream);


    // 3. populate d_keys using key_functor and d_vals using copy_n

    auto keys_it = thrust::make_transform_iterator(in, sphotonlite_key_functor{time_window, select_flagmask});
    thrust::copy(  policy, keys_it, keys_it + num_in, thrust::device_ptr<uint64_t>(d_keys));
    thrust::copy_n(policy, in     , num_in          , thrust::device_ptr<sphotonlite>(d_vals));


    // 4. sort_by_key arranging hits with same (id, timebucket) to be contiguous

    thrust::sort_by_key(policy,
                        thrust::device_ptr<uint64_t>(d_keys),
                        thrust::device_ptr<uint64_t>(d_keys + num_in),
                        thrust::device_ptr<sphotonlite>(d_vals));

    // 5. allocate d_out_key d_out_val with space for num_in

    uint64_t* d_out_key = nullptr; sphotonlite* d_out_val = nullptr;
    cudaMallocAsync(&d_out_key, num_in * sizeof(uint64_t),   stream);
    cudaMallocAsync(&d_out_val, num_in * sizeof(sphotonlite), stream);

    // 6. reduce_by_key merging contiguous equal (id,timebucket) hits

    auto ends = thrust::reduce_by_key(policy,
                thrust::device_ptr<uint64_t>(d_keys),
                thrust::device_ptr<uint64_t>(d_keys + num_in),
                thrust::device_ptr<sphotonlite>(d_vals),
                thrust::device_ptr<uint64_t>(d_out_key),
                thrust::device_ptr<sphotonlite>(d_out_val),
                thrust::equal_to<uint64_t>(),
                sphotonlite_reduce_op());

    // 7. get number of merged hits, potentially including a non-hitmask selected sentinel key ~0ull

    int merged = ends.first - thrust::device_ptr<uint64_t>(d_out_key);


    // 8. remove trailing sentinel (~0ull) if present

    if (merged > 0) {
        uint64_t last_key;
        cudaMemcpyAsync(&last_key, d_out_key + merged - 1, sizeof(uint64_t),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        if (last_key == ~0ull) --merged;
    }



    // 9. allocate d_final to fit merged hits, d2d copy d_final from d_out_val
    sphotonlite* d_final = nullptr;
    if (merged > 0) {
        cudaMallocAsync(&d_final, merged * sizeof(sphotonlite), stream);
        cudaMemcpyAsync(d_final, d_out_val, merged * sizeof(sphotonlite),
                        cudaMemcpyDeviceToDevice, stream);
    }


    // 10. free temporary buffers

    cudaFreeAsync(d_keys, stream);
    cudaFreeAsync(d_vals, stream);
    cudaFreeAsync(d_out_key, stream);
    cudaFreeAsync(d_out_val, stream);


    // 11. set d_out and num_out

    *d_out = d_final;
    if (num_out) *num_out = merged;
}




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
    int count;
};


/**
save_partial_callback
----------------------

1. open path and write h_pinned to it
2. free d_ptr and h_pinned

**/


extern "C" void CUDART_CB save_partial_callback(cudaStream_t stream, cudaError_t status, void* userData)
{
    (void)stream;
    SavePartialData* data = static_cast<SavePartialData*>(userData);
    if (status != cudaSuccess) {
        fprintf(stderr, "save_partial_callback: CUDA error %d\n", status);
    }

    // 1. open path and write h_pinned to it

    std::ofstream f(data->path, std::ios::binary);
    if (f) {
        f.write(reinterpret_cast<const char*>(data->h_pinned), data->count * sizeof(sphotonlite));
    } else {
        fprintf(stderr, "save_partial: failed to open %s\n", data->path.c_str());
    }

    // 2. free d_ptr and h_pinned

    cudaFree(data->d_ptr);
    cudaFreeHost(data->h_pinned);
    delete data;
}


/**
SPM::save_partial
------------------

1. host allocate *h_pinned* space for *count* hits
2. async copy count hits to h_pinned from d_partial
3. setup callback that saves to file after copy completes then cleans up

**/


void SPM::save_partial(
        const sphotonlite* d_partial,
        int                count,
        const std::string& path,
        cudaStream_t       stream )
{
    if (count == 0 || !d_partial) {
        cudaFree(const_cast<sphotonlite*>(d_partial));
        return;
    }

    // 1. host allocate *h_pinned* space for *count* hits

    sphotonlite* h_pinned = nullptr;
    cudaMallocHost(&h_pinned, count * sizeof(sphotonlite));

    // 2. async copy count hits to h_pinned from d_partial

    cudaMemcpyAsync(h_pinned, d_partial, count * sizeof(sphotonlite),
                    cudaMemcpyDeviceToHost, stream);

    // 3. setup callback that saves to file after copy completes then cleans up

    SavePartialData* cb_data = new SavePartialData{h_pinned, const_cast<sphotonlite*>(d_partial), path, count};
    cudaStreamAddCallback(stream, save_partial_callback, cb_data, 0);
}



/**
SPM::load_partial
------------------

1. open *path* and use tellg to determine *n*
2. read from file into *h* vector of sphotonlite
3. allocate *d* and async copy from *h* to *d*
4. set output args `*d_out` and `count`

**/


void SPM::load_partial(
        const std::string& path,
        sphotonlite**      d_out,
        int*               count,
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
        int           count;
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

        return {d_keys, d_hits, (int)n};
    }
}

/**
SPM::merge_incremental
-----------------------

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



void SPM::merge_incremental(
        const char** partial_paths,
        sphotonlite** d_final,
        int*          final_count,
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
        int           reduced_count;

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
                    sphotonlite_reduce_op());

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
