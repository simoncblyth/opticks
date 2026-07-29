#pragma once
/**
smath_test.h
=============

Template specializations to make it look like are using
templates symbols across compilers when actually using a C interface
to cross the divide.

**/

#include <cstddef>

extern "C" {
    void launch_log_kernel_float(  float* values, float* domain,  size_t num_values, float x0,  float x1 );
    void launch_log_kernel_double(double* values, double* domain, size_t num_values, double x0, double x1);
}

template <typename T> void launch_log_kernel(T* values, T* domain, size_t num_values, T x0, T x1);

template <>
inline void launch_log_kernel<float>(float* values, float* domain, size_t num_values, float x0, float x1)
{
    launch_log_kernel_float(values, domain, num_values, x0, x1);
}

template <>
inline void launch_log_kernel<double>(double* values, double* domain, size_t num_values, double x0, double x1)
{
    launch_log_kernel_double(values, domain, num_values, x0, x1);
}

