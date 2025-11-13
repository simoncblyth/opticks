#pragma once
/**
sysrap/SPM.hh
===============

Grok generated code under test

**/

#include "sphotonlite.h"
#include <string>

struct SPM
{
    static constexpr float DEFAULT_TIME_WINDOW = 1.0f ;   // ns

    // time_window == 0.f  →  only selection (two-pass, exact size)
    static void merge_partial_select(
            const sphotonlite* d_in,
            int                 num_in,
            sphotonlite**       d_out,
            int*                num_out,
            unsigned            select_flagmask = 0xffffffffu,
            float               time_window     = DEFAULT_TIME_WINDOW,
            cudaStream_t        stream = 0 );

    static void merge_incremental(
            const char**        partial_paths,
            sphotonlite**       d_final,
            int*                final_count,
            float               time_window = DEFAULT_TIME_WINDOW,
            cudaStream_t        stream = 0 );

    // ------------------------------------------------------------------
    // Async I/O utilities (non-blocking, stream-aware)
    // ------------------------------------------------------------------
    static void save_partial(
            const sphotonlite* d_partial,
            int                count,
            const std::string& path,
            cudaStream_t       stream = 0 );

    static void load_partial(
            const std::string& path,
            sphotonlite**      d_out,
            int*               count,
            cudaStream_t       stream = 0 );

    static void free( void* d_ptr ) { cudaFree(d_ptr); }
};
