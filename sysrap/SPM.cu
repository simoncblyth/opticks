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

save_partial_callback_async


SPM::save_partial

SPM::load_partial

{
    Partial

    load_partial_for_merge
}

SPM::merge_incremental

SPM::copy_device_to_host_async

SPM::free

SPM::free_async


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
sphotonlite_key_functor_with_select_mask
-----------------------------------------

Returns sentinel ~0ull if not selected, otherwise bitwise combination
of id(16bits) and bucket(48bits)

HMM: I expect 16 bits for the  bucket would be enough
**/


struct sphotonlite_key_functor_with_select_mask
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

struct sphotonlite_key_functor
{
    float    tw;
    __device__ uint64_t operator()(const sphotonlite& p) const
    {
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



SPM_future SPM::merge_partial_select_async(
    const sphotonlite*  d_in,
    size_t            num_in,
    unsigned select_flagmask,
    float        time_window,
    cudaStream_t      stream )
{
    SPM_future result  ;

    if (num_in == 0 )
    {
        result.count = 0;
        result.ptr   = nullptr;
    }
    else
    {
        merge_partial_select(
                d_in,
                num_in,
                &result.ptr,
                &result.count,
                select_flagmask,
                time_window,
                stream );
    }

    cudaEventCreateWithFlags(&result.ready, cudaEventDisableTiming);
    cudaEventRecord(result.ready, stream);
    // Record event immediately after last operation

    return result ;
}


/**
SPM::merge_partial_select
-------------------------

Flagmask select hits from input photons and merge the hits by (id,timebucket)
with incremented counts. Obviously this requires the photons and hits to
simultaneously fit into VRAM.

0. apply selection using count_if, allocate, copy_if
1. special case time_window:0.f to just return selected without merging
2. allocate `(uint64_t*)d_keys` and `(sphotonlite*)d_vals`
3. populate d_keys using sphotonlite_key_functor and d_vals using copy_n
4. sort_by_key arranging hits with same (id, timebucket) together
5. allocate d_out_key d_out_val with space for num_in [HMM, COULD DOUBLE PASS TO REDUCE MEMORY PERHAPS?]
6. reduce_by_key merging contiguous equal (id,timebucket) hits
7. get number of merged hits
8. allocate d_final to fit merged hits, d2d copy d_final from d_out_val
9. free temporary buffers
10. set d_out and num_out


"Safe, Seamless, And Scalable Integration Of Asynchronous GPU Streams In PETSc"
    https://arxiv.org/pdf/2306.17801

"CUDA C/C++ Streams and Concurrency, Steve Rennich, NVIDIA"
    https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf


par_nosync is quite recent (Feb 9, 2022)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

https://github.com/NVIDIA/thrust/blob/main/examples/cuda/explicit_cuda_stream.cu
    par_nosync does not stop sync when that is needed for correctness,
    ie to return num_selected to the host.

https://github.com/NVIDIA/thrust/issues/1515
https://github.com/petsc/petsc/commit/0fa675732414ab06118e41f905207ba3ccea9c4e


https://github.com/NVIDIA/thrust/discussions/1616
    Feb 9, 2022
    Thrust 1.16.0 provides a new “nosync” hint


**/

void SPM::merge_partial_select(
    const sphotonlite*  d_in,
    size_t            num_in,
    sphotonlite**      d_out,
    size_t*          num_out,
    unsigned select_flagmask,
    float        time_window,
    cudaStream_t      stream )
{

    //printf("[SPM::merge_partial_select num_in %d select_flagmask %d time_window %7.3f \n", num_in, select_flagmask, time_window );

    if (num_in == 0) { *d_out = nullptr; if (num_out) *num_out = 0; return; }

    auto policy = thrust::cuda::par_nosync.on(stream);
    auto in = thrust::device_ptr<const sphotonlite>(d_in);


    // 0. apply selection using count_if, allocate, copy_if

    sphotonlite_select_pred selector{select_flagmask};
    size_t num_selected = thrust::count_if(policy, in, in + num_in, selector);

    if (num_selected == 0)
    {
        *d_out = nullptr;
        if (num_out) *num_out = 0;
        return;
    }

    sphotonlite* d_selected = nullptr;
    cudaMallocAsync(&d_selected, num_selected * sizeof(sphotonlite), stream);
    thrust::copy_if(policy, in, in + num_in, thrust::device_ptr<sphotonlite>(d_selected), selector);

    // 1. special case time_window:0.f to just return selected without merging
    if (time_window == 0.f)
    {
        *d_out = d_selected;
        if (num_out) *num_out = num_selected ;
        return;
    }


    // 2. allocate `(uint64_t*)d_keys` and `(sphotonlite*)d_vals`

    uint64_t* d_keys = nullptr;
    cudaMallocAsync(&d_keys, num_selected * sizeof(uint64_t),    stream);



    // 3. populate d_keys using key_functor and d_vals using copy_n
    auto selected = thrust::device_ptr<const sphotonlite>(d_selected);
    auto keys_it = thrust::make_transform_iterator(selected, sphotonlite_key_functor{time_window});
    thrust::copy(  policy, keys_it , keys_it + num_selected, thrust::device_ptr<uint64_t>(d_keys));



    // 4. sort_by_key arranging hits with same (id, timebucket) to be contiguous

    thrust::sort_by_key(policy,
                        thrust::device_ptr<uint64_t>(d_keys),
                        thrust::device_ptr<uint64_t>(d_keys + num_selected),
                        thrust::device_ptr<sphotonlite>(d_selected));

    // 5. allocate d_out_key d_out_val with space for num_selected

    uint64_t*    d_out_key = nullptr;
    sphotonlite* d_out_val = nullptr;
    cudaMallocAsync(&d_out_key, num_selected * sizeof(uint64_t),   stream);
    cudaMallocAsync(&d_out_val, num_selected * sizeof(sphotonlite), stream);

    // 6. reduce_by_key merging contiguous equal (id,timebucket) hits

    auto d_out_key_begin = thrust::device_ptr<uint64_t>(d_out_key);

    auto ends = thrust::reduce_by_key(policy,
                thrust::device_ptr<uint64_t>(d_keys),
                thrust::device_ptr<uint64_t>(d_keys + num_selected),
                thrust::device_ptr<sphotonlite>(d_selected),
                d_out_key_begin,                      // output keys
                thrust::device_ptr<sphotonlite>(d_out_val),
                thrust::equal_to<uint64_t>{},
                sphotonlite_reduce_op());

    // Synchronize the stream here to ensure reduce_by_key results are ready for host access
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    if (sync_err != cudaSuccess) {
        fprintf(stderr, "cudaStreamSynchronize failed: %s\n", cudaGetErrorString(sync_err));
        return ;
    }

    // 7. get number of merged hits
    size_t merged = ends.first.get() - d_out_key ;
    // (Proceed to step 8 as-is; the sync ensures d_out_val is also ready for the subsequent cudaMemcpyAsync)


    // 8. allocate d_final to fit merged hits, d2d copy d_final from d_out_val
    sphotonlite* d_final = nullptr;
    if (merged > 0) {
        cudaMallocAsync(&d_final, merged * sizeof(sphotonlite), stream);
        cudaMemcpyAsync(d_final, d_out_val, merged * sizeof(sphotonlite),
                        cudaMemcpyDeviceToDevice, stream);
    }

    // 9. free temporary buffers

    cudaFreeAsync(d_selected, stream);   // omitting this caused leak steps of 0.9GB in the whopper 8.25 billion test
    cudaFreeAsync(d_keys, stream);
    cudaFreeAsync(d_out_key, stream);
    cudaFreeAsync(d_out_val, stream);


    // 10. set d_out and num_out

    *d_out = d_final;
    if(num_out) *num_out = merged;

    //printf("]SPM::merge_partial_select merged %d  \n", merged );

    float select_frac = float(num_selected)/float(num_in);
    float merge_frac = float(merged)/float(num_selected) ;
    printf("]SPM::merge_partial_select select_flagmask %d time_window %7.3f in %d selected %d merged %d selected/in %7.3f merged/selected %7.3f \n",
                select_flagmask, time_window, num_in, num_selected, merged, select_frac, merge_frac );

}
// sed -n '/^void SPM::merge_partial_select/,/^}/p' ~/o/sysrap/SPM.cu | pbcopy


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
save_partial_callback
----------------------

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


/**
SPM::save_partial
------------------

1. host allocate *h_pinned* space for *count* hits
2. async copy count hits to h_pinned from d_partial
3. setup callback that saves to file after copy completes then cleans up

Replaced ancient cudaMallocHost with cudaMallocAsync as the later is
memlock limited by "ulimit -l"  (default 8MB).

**/



void SPM::save_partial(
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





template<typename T>
int SPM::copy_device_to_host_async( T* h, T* d,  size_t num_items, cudaStream_t stream )
{
    if( d == nullptr ) std::cerr
        << "SPM::copy_device_to_host_async"
        << " ERROR : device pointer is null "
        << std::endl
        ;

    if( d == nullptr ) return 1 ;

    size_t size = num_items*sizeof(T) ;

    cudaMemcpyAsync(h, d, size, cudaMemcpyDeviceToHost, stream);

    return 0 ;
}


template int SPM::copy_device_to_host_async<sphotonlite>( sphotonlite* h, sphotonlite* d,  size_t num_items, cudaStream_t stream );

void SPM::free( void* d_ptr )   // static
{
    cudaFree(d_ptr);
}

void SPM::free_async(void* d_ptr, cudaStream_t stream)  // static
{
    if (d_ptr) cudaFreeAsync(d_ptr, stream);
}


