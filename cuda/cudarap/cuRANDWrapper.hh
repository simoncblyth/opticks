#pragma once

class LaunchSequence ; 

#include <vector>
#include "cuda.h"
#include "curand_kernel.h"

/*

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



*/





class cuRANDWrapper {
  public: 
     cuRANDWrapper( LaunchSequence* launchseq, unsigned long long seed=0, unsigned long long offset=0 );
 
     static cuRANDWrapper* instanciate(
         unsigned int elements, 
         const char* cachedir=NULL,
         unsigned long long seed=0,
         unsigned long long offset=0,
         unsigned int max_blocks=128,
         unsigned int threads_per_block=256
     );

  public: 
     char* getCachePath();
     unsigned int getSeed();
     unsigned int getOffset();
     LaunchSequence* getLaunchSequence();
     unsigned int getItems();
     bool isOwner();
     char* digest();
     char* testdigest();
     CUdeviceptr getDevRngStates();
     curandState* getHostRngStates();

  public: 
     void setCacheDir(const char* dir);
     void setDevRngStates(CUdeviceptr dev_rng_states, bool owner );
     void setItems(unsigned int items);
     void setCacheEnabled(bool enabled);
     void setImod(unsigned int imod);
     bool hasCacheEnabled();

  public: 
     void resize(unsigned int elements);
     int fillHostBuffer(curandState* host_rng_states, unsigned int elements);
     int LoadIntoHostBuffer(curandState* host_rng_states, unsigned int elements);

     void Summary(const char* msg);
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
     int Save(const char* path);
     int Load(const char* path);

  private: 
     void devicesync();
     void allocate_rng();
     void test_rng(const char* tag="test_rng");

  private:
     unsigned long long m_seed ;
     unsigned long long m_offset ;

  private:
     CUdeviceptr      m_dev_rng_states ;
     curandState*     m_host_rng_states ;
     float*           m_test ;
     LaunchSequence*  m_launchseq ; 
     unsigned int     m_imod ;
     char*            m_cache_dir ; 
     bool             m_cache_enabled ;
     bool             m_owner ; 
     bool             m_first_resize ; 

     std::vector<LaunchSequence*> m_launchrec ; 

};



inline unsigned int cuRANDWrapper::getSeed()
{  
    return m_seed ; 
}
inline unsigned int cuRANDWrapper::getOffset()
{ 
    return m_offset ; 
}
inline LaunchSequence* cuRANDWrapper::getLaunchSequence()
{ 
    return m_launchseq ; 
}
inline void cuRANDWrapper::setCacheEnabled(bool enabled)
{ 
    m_cache_enabled = enabled ; 
}
inline bool cuRANDWrapper::hasCacheEnabled()
{
    return m_cache_enabled ; 
}

inline void cuRANDWrapper::setDevRngStates(CUdeviceptr dev_rng_states, bool owner )
{
    m_dev_rng_states = dev_rng_states ;
    m_owner = owner ; 
} 
inline CUdeviceptr cuRANDWrapper::getDevRngStates()
{ 
    return m_dev_rng_states ; 
}
inline bool cuRANDWrapper::isOwner()
{ 
    return m_owner ; 
} 

inline curandState* cuRANDWrapper::getHostRngStates()
{ 
    return m_host_rng_states ; 
}

inline void cuRANDWrapper::setImod(unsigned int imod)
{
    m_imod = imod ; 
}





