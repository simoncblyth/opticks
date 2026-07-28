#pragma once

#include <cstddef>

extern "C" {
    void run_qcurand_test_float(  float* values, size_t num_values, unsigned long seed);
    void run_qcurand_test_double(double* values, size_t num_values, unsigned long seed);
}

// C++ host-side inline wrapper providing a unified templated interface
template <typename T>
void run_qcurand_test(T* values, size_t num_values, unsigned long seed);

template <>
inline void run_qcurand_test<float>(float* values, size_t num_values, unsigned long seed) {
    run_qcurand_test_float(values, num_values, seed);
}

template <>
inline void run_qcurand_test<double>(double* values, size_t num_values, unsigned long seed) {
    run_qcurand_test_double(values, num_values, seed);
}

