#pragma once
/**
QPlanck_test.h
===============

This provides a templated interface across to symbol definitions from qudarap/QPlanck_test.cu
without actually using templated symbols between compilers - as that can be unreliable.

**/

#include <cstddef>

extern "C" {
    void qplanck_test_float( float*  values, size_t num_values, unsigned long seed, qplanck* d_planck);
    void qplanck_test_double(double* values, size_t num_values, unsigned long seed, qplanck* d_planck);
}

template <typename T>
void qplanck_test(T* values, size_t num_values, unsigned long seed, qplanck* d_planck);

template <>
inline void qplanck_test<float>(float* values, size_t num_values, unsigned long seed, qplanck* d_planck)
{
    qplanck_test_float(values, num_values, seed, d_planck);
}

template <>
inline void qplanck_test<double>(double* values, size_t num_values, unsigned long seed, qplanck* d_planck) 
{
    qplanck_test_double(values, num_values, seed, d_planck);
}



