#pragma once

struct sphotonlite ;

struct SPM_future
{
    sphotonlite* ptr = nullptr;
    size_t       count = 0;
    cudaEvent_t  ready = nullptr;   // caller must destroy

    void wait( cudaStream_t stream ) const ;
};

inline void SPM_future::wait(cudaStream_t stream = 0) const
{
    if (ready && stream) {
        cudaStreamWaitEvent(stream, ready, 0);
    }
}

