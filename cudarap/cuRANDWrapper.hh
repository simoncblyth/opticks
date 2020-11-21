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

class LaunchSequence ; 

#include <vector>
#include "plog/Severity.h"
#include "cuda.h"
#include "curand_kernel.h"

/**
cuRANDWrapper
================

TODO
--------

* bring this ancient code upto scratch, needs a drastic rewrite to become comprehensible
* perhaps use a simular Thrust based approach similar to TRngBufTest 


Ancient Issues
-----------------

OptiX interop issues
~~~~~~~~~~~~~~~~~~~~

cuRANDWrapper works reliably with plain CUDA, but interop
with OptiX is fraught with difficulties.  However cannot
get rid of this as need this to setup the RNG caches that raytrace
consumes.

[SOLVED] Inconsistent save/load digests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Was initially getting inconsistent save and load digests, but the 
test digests were coming out the same. So something was
wrong but it was not influencing the random numbers generated.

After using cudaMemset to zero the device buffer at allocation
the load and save digests are matching. This suggests that the 
init_curand leaves some of the buffer undefined.


buffer digests do not match file digests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A commandline digest on the file does not match the digest 
from the buffer ? 

::

    delta:cudawrap blyth$ md5 /tmp/env/cuRANDWrapperTest/cachedir/cuRANDWrapper_786432_0_0.bin
    MD5 (/tmp/env/cuRANDWrapperTest/cachedir/cuRANDWrapper_786432_0_0.bin) = eb42d4f18f8636a9a7f70577092ff22d

**/


#include "CUDARAP_API_EXPORT.hh"
#include "CUDARAP_HEAD.hh"

class CUDARAP_API cuRANDWrapper {
  public: 
     static const plog::Severity LEVEL ; 
  public: 
     cuRANDWrapper( const LaunchSequence* launchseq, unsigned long long seed=0, unsigned long long offset=0, bool verbose=false);
 
     static cuRANDWrapper* instanciate(
         unsigned int elements, 
         const char* cachedir=NULL,
         unsigned long long seed=0,
         unsigned long long offset=0,
         unsigned int max_blocks=128,
         unsigned int threads_per_block=256,
         bool verbose=false
     );

  public: 
     const char* getCachePath() const ;
  public: 
     unsigned getSeed() const ;
     unsigned getOffset() const ;
     bool isVerbose() const ;
     const LaunchSequence* getLaunchSequence() const ;
  public: 
     unsigned getItems() const ;
     bool hasCacheEnabled() const ;
  public: 
     void setItems(unsigned items);  // via a const_cast 
     void setCacheEnabled(bool enabled);
  public: 
     void setDevRngStates(CUdeviceptr dev_rng_states, bool owner );
  public: 
     CUdeviceptr getDevRngStates() const ;
     bool        isOwner() const ;
  public: 
     curandState* getHostRngStates() const ;
  public: 
     char* digest();
     char* testdigest();
  public: 
     void setCacheDir(const char* dir);
     void setImod(unsigned int imod);
  public: 
     void resize(unsigned int elements);
     int LoadIntoHostBuffer(curandState* host_rng_states, unsigned int elements);
     int LoadIntoHostBufferMasked(curandState* host_rng_states, const std::vector<unsigned>& mask);

     void Summary(const char* msg) const ;
     void Dump(const char* msg="cuRANDWrapper::Dump", unsigned int imod=1000);

  public:
     int Allocate();
     int Free();
     int InitFromCacheIfPossible();

     int hasCache();
     int Init();
     int Save();
     int Load();
     int Test();

  private:
     int hasCache(const char* path);
     int SaveToFile(const char* path);
     int LoadFromFile(const char* path);

  private: 
     void devicesync();
     void allocate_rng();
     void test_rng(const char* tag="test_rng", bool update_states=true );

  private:
     unsigned long long m_seed ;
     unsigned long long m_offset ;
     bool               m_verbose ; 

  private:
     CUdeviceptr            m_dev_rng_states ;
     curandState*           m_host_rng_states ;
     float*                 m_test ;
     const LaunchSequence*  m_launchseq ; 
     unsigned               m_imod ;
     char*                  m_cache_dir ; 
     bool                   m_cache_enabled ;
     bool                   m_owner ; 
     bool                   m_first_resize ; 

     std::vector<const LaunchSequence*> m_launchrec ; 

};

#include "CUDARAP_TAIL.hh"



