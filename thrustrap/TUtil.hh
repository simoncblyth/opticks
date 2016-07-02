#pragma once

struct CBufSpec ; 
#include <thrust/device_vector.h>

#include "THRAP_API_EXPORT.hh" 

template <typename T>
CBufSpec make_bufspec(const thrust::device_vector<T>& d_vec );

