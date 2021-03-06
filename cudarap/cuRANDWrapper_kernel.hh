/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#pragma once

/**
cuRANDWrapper_kernel
======================

Not exporting API, as look local.

**/

#include "cuda.h"
#include "curand_kernel.h"

class LaunchSequence ; 

// allocate curandState GPU buffer large enough for rngmax launchseq items  
CUdeviceptr allocate_rng_wrapper( const LaunchSequence* launchseq);

// free the curandState GPU buffer
void free_rng_wrapper( CUdeviceptr dev_rng_states );

// allocates host memory and copies from GPU buffer to it, returning pointer 
curandState* copytohost_rng_wrapper( const LaunchSequence* launchseq, CUdeviceptr dev_rng_states);

// allocates device memory and copues from host buffer to it 
CUdeviceptr copytodevice_rng_wrapper( const LaunchSequence* launchseq, void* host_rng_states);

// multiple launches to initialized the curandState GPU buffer
void init_rng_wrapper( const LaunchSequence* launchseq, CUdeviceptr dev_rng_states, unsigned long long seed, unsigned long long offset);

// multiple launches with each thread generating a single curand_uniform float 
void test_rng_wrapper( const LaunchSequence* launchseq, CUdeviceptr dev_rng_states, float* host_a, bool update_states );

// cudaDeviceSynchronize 
void devicesync_wrapper();


