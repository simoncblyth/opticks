#pragma once
/**
sysrap/SPM_dev.hh
==================

Experiment with async CUDA download and save

**/

#include <string>

//#include "sphoton.h"
#include "sphotonlite.h"

struct SPM_dev
{
    static constexpr float DEFAULT_TIME_WINDOW = 1.0f ;   // ns

    static void merge_incremental_sphotonlite(
            const char**        partial_paths,
            sphotonlite**       d_final,
            size_t*             final_count,
            float               time_window = DEFAULT_TIME_WINDOW,
            cudaStream_t        stream = 0 );

    static void save_partial_sphotonlite(
            const sphotonlite* d_partial,
            size_t             count,
            const std::string& path,
            cudaStream_t       stream = 0 );


    template<typename T>
    void save_partial(
            const T* d_partial,
            size_t             count,
            const std::string& path,
            cudaStream_t       stream = 0 );



    static void load_partial_sphotonlite(
            const std::string& path,
            sphotonlite**      d_out,
            size_t*            count,
            cudaStream_t       stream = 0 );

};




