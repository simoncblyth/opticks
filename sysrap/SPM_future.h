#pragma once


template<typename T>
struct SPM_future
{
    T* ptr = nullptr;
    size_t       count = 0;
    cudaEvent_t  ready = nullptr;   // caller must destroy

    void wait( cudaStream_t stream ) const ;
};

template<typename T>
inline void SPM_future<T>::wait(cudaStream_t stream ) const
{
    if (ready && stream) {
        cudaStreamWaitEvent(stream, ready, 0);
    }
}

