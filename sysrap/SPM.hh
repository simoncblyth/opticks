#pragma once
/**
sysrap/SPM.hh
===============

This CUDA hit merging functionality is used from::

   QEvt::PerLaunchMerge
   QEvt::FinalMerge
   QEvt::FinalMerge_async

**/

#include <string>
#include "sphoton.h"
#include "sphotonlite.h"
#include "SPM_future.h"



struct SPM
{
    static constexpr unsigned ALREADY_HITMASK_SELECTED = 0xffffffffu ;
    static constexpr float DEFAULT_TIME_WINDOW = 1.0f ;   // ns
    static constexpr float NOMERGE_TIME_WINDOW = 0.0f ;   // only selects

    template<typename T>
    static SPM_future<T> merge_partial_select_async(
           const T*            d_photonlite,
           size_t            num_photonlite,
           unsigned         select_flagmask,
           float                time_window,
           cudaStream_t              stream );

    template<typename T>
    static void merge_partial_select(
            const T*           d_in,
            size_t             num_in,
            T**                d_out,
            size_t*            num_out,
            unsigned           select_flagmask = 0xffffffffu,
            float              time_window     = DEFAULT_TIME_WINDOW,
            cudaStream_t       stream = 0 );


    template<typename T>
    static int copy_device_to_host_async( T* h, T* d,  size_t num_items, cudaStream_t stream = 0 );

    static void free( void* d_ptr );
    static void free_async(void* d_ptr, cudaStream_t stream = 0) ;


};
