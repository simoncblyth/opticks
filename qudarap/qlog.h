#pragma once
/**
qlog.h : logarithm dispatchers - avoiding unintentional use of double
=====================================================================

**/


template <typename T>
__device__ inline T qlog(T val);

template <>
__device__ inline float qlog<float>(float val) {
    return ::logf(val); // Explicit FP32 single-precision intrinsic
}

template <>
__device__ inline double qlog<double>(double val) {
    return ::log(val);  // Explicit FP64 double-precision intrinsic
}



