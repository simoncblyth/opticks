#pragma once

/**
NP_future.h
============

General pattern of async functions::

    // Inside every async function:
    cudaStreamWaitEvent(my_stream, input.ready, 0);   // consume input
    // ... do work ...
    cudaEventRecord(output.ready, my_stream);         // produce output
    cudaEventDestroy(input.ready);                    // safe cleanup
    return output;                                    // ONE event


**/



struct NP_future
{
    NP*          arr   = nullptr;
    cudaEvent_t  ready = nullptr;

    void wait(cudaStream_t stream) const;
};

inline void NP_future::wait(cudaStream_t stream) const
{
    if (ready && stream) {
        cudaStreamWaitEvent(stream, ready, 0);
    }
}




