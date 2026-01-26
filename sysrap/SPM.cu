/**
SPM.cu
=======


SPM::merge_partial_select_async
    Uses merge_partial_select and non-blockingly returns SPM_future<T>

SPM::merge_partial_select
    Flagmask select hits from input photons and merge the hits by (id,timebucket)
    with incremented counts. Obviously this requires the photons and hits to
    simultaneously fit into VRAM.

SPM::copy_device_to_host_async

SPM::free

SPM::free_async



Background on async CUDA
--------------------------

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

template<typename T>
SPM_future<T> SPM::merge_partial_select_async(
    const T*          d_in,
    size_t            num_in,
    unsigned select_flagmask,
    float        time_window,
    cudaStream_t      stream )
{
    SPM_future<T> result  ;

    if (num_in == 0 )
    {
        result.count = 0;
        result.ptr   = nullptr;
    }
    else
    {
        merge_partial_select<T>(
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


template SPM_future<sphotonlite> SPM::merge_partial_select_async( const sphotonlite* d_in, size_t num_in, unsigned select_flagmask, float time_window, cudaStream_t stream );
template SPM_future<sphoton>     SPM::merge_partial_select_async( const sphoton*     d_in, size_t num_in, unsigned select_flagmask, float time_window, cudaStream_t stream );


/**
SPM::merge_partial_select
-------------------------

Flagmask select hits from input photons and merge the hits by (id,timebucket)
with incremented counts. Obviously this requires the photons and hits to
simultaneously fit into VRAM.

0. apply selection using count_if, allocate, copy_if
1. special case time_window:0.f to just return selected without merging
2. allocate `(uint64_t*)d_keys`
3. populate d_keys using T::key_functor and d_selected using copy_n
4. sort_by_key arranging d_selected hits with same (id, timebucket) together
5. allocate d_out_key d_out_val with space for num_in [HMM, COULD DOUBLE PASS TO REDUCE MEMORY PERHAPS?]
6. reduce_by_key merging contiguous equal (id,timebucket) hits
7. get number of merged hits
8. allocate d_final to fit merged hits, d2d copy d_final from d_out_val
9. free temporary buffers
10. set d_out and num_out

final merge special case
~~~~~~~~~~~~~~~~~~~~~~~~~~

With the final merge of the concatenated per-launch select+merge results
the selection has been done already and hence does not need to be repeated.
As d_selected is then the same as d_in the free is skipped as d_in belongs
to the caller.

**/

template<typename T> void SPM::merge_partial_select(
    const T*          d_in,
    size_t            num_in,
    T**               d_out,
    size_t*          num_out,
    unsigned select_flagmask,
    float        time_window,
    cudaStream_t      stream )
{
    using select_pred   = typename T::select_pred;
    using reduce_op     = typename T::reduce_op;
    using key_functor   = typename T::key_functor;


    printf("[SPM::merge_partial_select num_in %d select_flagmask %d time_window %7.3f \n", num_in, select_flagmask, time_window );

    if (num_in == 0) { *d_out = nullptr; if (num_out) *num_out = 0; return; }

    auto policy = thrust::cuda::par_nosync.on(stream);
    auto in = thrust::device_ptr<const T>(d_in);


    // 0. apply selection using count_if, allocate, copy_if

    T* d_selected = nullptr;
    size_t num_selected = 0 ;

    bool apply_selection = select_flagmask != ALREADY_HITMASK_SELECTED ;
    if( apply_selection )
    {
        select_pred selector{select_flagmask} ;
        num_selected = thrust::count_if(policy, in, in + num_in, selector);

        if (num_selected == 0)
        {
            *d_out = nullptr;
            if (num_out) *num_out = 0;
            return;
        }

        cudaMallocAsync(&d_selected, num_selected * sizeof(T), stream);
        thrust::copy_if(policy, in, in + num_in, thrust::device_ptr<T>(d_selected), selector);
    }
    else
    {
         d_selected = (T*)d_in ;
         num_selected = num_in ;
    }


    // 1. special case time_window:0.f to just return selected without merging
    if (time_window == NOMERGE_TIME_WINDOW)
    {
        *d_out = d_selected;
        if (num_out) *num_out = num_selected ;
        return;
    }


    // 2. allocate `(uint64_t*)d_keys`

    uint64_t* d_keys = nullptr;
    cudaMallocAsync(&d_keys, num_selected * sizeof(uint64_t),    stream);



    // 3. populate d_keys using key_functor and d_vals using copy_n
    auto selected = thrust::device_ptr<const T>(d_selected);
    auto keys_it = thrust::make_transform_iterator(selected, key_functor{time_window});
    thrust::copy(  policy, keys_it , keys_it + num_selected, thrust::device_ptr<uint64_t>(d_keys));



    // 4. sort_by_key arranging d_selected hits with same (id, timebucket) to be contiguous

    thrust::sort_by_key(policy,
                        thrust::device_ptr<uint64_t>(d_keys),
                        thrust::device_ptr<uint64_t>(d_keys + num_selected),
                        thrust::device_ptr<T>(d_selected));

    cudaFreeAsync(d_keys, stream);



    // 5. allocate d_out_key d_out_val with space for num_selected

    uint64_t*    d_out_key = nullptr;
    T*           d_out_val = nullptr;
    cudaMallocAsync(&d_out_key, num_selected * sizeof(uint64_t),   stream);
    cudaMallocAsync(&d_out_val, num_selected * sizeof(T),          stream);

    // 6. reduce_by_key merging contiguous equal (id,timebucket) hits

    auto d_out_key_begin = thrust::device_ptr<uint64_t>(d_out_key);

    auto ends = thrust::reduce_by_key(policy,
                thrust::device_ptr<uint64_t>(d_keys),
                thrust::device_ptr<uint64_t>(d_keys + num_selected),
                thrust::device_ptr<T>(d_selected),
                d_out_key_begin,                      // output keys
                thrust::device_ptr<T>(d_out_val),
                thrust::equal_to<uint64_t>{},
                reduce_op{});


    if(apply_selection )
    {
        cudaFreeAsync(d_selected, stream);
        // omitting this caused leak steps of 0.9GB in the whopper 8.25 billion test
    }
    else
    {
        assert( d_selected == d_in && num_selected == num_in );
        // for apply_selection:false d_selected is same as d_in which belongs to caller
    }


    // Synchronize the stream here to ensure reduce_by_key results are ready for host access
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    if (sync_err != cudaSuccess) {
        fprintf(stderr, "cudaStreamSynchronize failed: %s\n", cudaGetErrorString(sync_err));
        return ;
    }

    // 7. get number of merged hits
    size_t merged = ends.first.get() - d_out_key ;
    // (Proceed to step 8 as-is; the sync ensures d_out_val is also ready for the subsequent cudaMemcpyAsync)

    cudaFreeAsync(d_out_key, stream);

    // 8. allocate d_final to fit merged hits, d2d copy d_final from d_out_val
    T* d_final = nullptr;
    if (merged > 0) {
        cudaMallocAsync(&d_final, merged * sizeof(T), stream);
        cudaMemcpyAsync(d_final, d_out_val, merged * sizeof(T),
                        cudaMemcpyDeviceToDevice, stream);
    }

    // 9. free temporary buffers

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
// sed -n '/^template<typename T> void SPM::merge_partial_select/,/^}/p' ~/o/sysrap/SPM.cu | pbcopy

template void SPM::merge_partial_select<sphotonlite>( const sphotonlite* d_in, size_t num_in, sphotonlite** d_out, size_t* num_out, unsigned select_flagmask, float time_window, cudaStream_t stream );
template void SPM::merge_partial_select<sphoton>(     const sphoton*     d_in, size_t num_in, sphoton**     d_out, size_t* num_out, unsigned select_flagmask, float time_window, cudaStream_t stream );


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
template int SPM::copy_device_to_host_async<sphoton>(     sphoton* h,     sphoton* d,      size_t num_items, cudaStream_t stream );


void SPM::free( void* d_ptr )   // static
{
    cudaFree(d_ptr);
}

void SPM::free_async(void* d_ptr, cudaStream_t stream)  // static
{
    if (d_ptr) cudaFreeAsync(d_ptr, stream);
}


