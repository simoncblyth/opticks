#ifndef CURANDWRAPPER_H
#define CURANDWRAPPER_H

class LaunchSequence ; 

#include <vector>
#include "cuda.h"
#include "curand_kernel.h"


//struct curandState ; 


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
     virtual ~cuRANDWrapper();
 
     static cuRANDWrapper* instanciate(
         unsigned int elements, 
         const char* cachedir=NULL,
         unsigned long long seed=0,
         unsigned long long offset=0,
         unsigned int max_blocks=128,
         unsigned int threads_per_block=256
     );

     unsigned int getSeed(){   return m_seed ; }
     unsigned int getOffset(){ return m_offset ; }
     LaunchSequence* getLaunchSequence(){ return m_launchseq ; }

     unsigned int getItems();
     void setItems(unsigned int items);

     void setCacheEnabled(bool enabled){ m_cache_enabled = enabled ; }
     bool hasCacheEnabled(){ return m_cache_enabled ; }

     int fillHostBuffer(curandState* host_rng_states, unsigned int elements);
     int LoadIntoHostBuffer(curandState* host_rng_states, unsigned int elements);

     void Summary(const char* msg);
     void Dump(const char* msg="cuRANDWrapper::Dump", unsigned int imod=1000);
     char* digest();
     char* testdigest();

     bool isOwner(){ return m_owner ; } 

     void setDevRngStates(CUdeviceptr dev_rng_states, bool owner )
     {
         m_dev_rng_states = dev_rng_states ;
         m_owner = owner ; 
     } 
     CUdeviceptr getDevRngStates(){ return m_dev_rng_states ; }
     curandState* getHostRngStates(){ return m_host_rng_states ; }
     
     void resize(unsigned int elements);

     void setImod(unsigned int imod)
     {
         m_imod = imod ; 
     }


     char* getCachePath();
     void setCacheDir(const char* dir);

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
     CUdeviceptr m_dev_rng_states ;
     curandState* m_host_rng_states ;
     float* m_test ;
     LaunchSequence* m_launchseq ; 
     unsigned int m_imod ;
     char* m_cache_dir ; 
     bool m_cache_enabled ;
     bool m_owner ; 
     bool m_first_resize ; 


     std::vector<LaunchSequence*> m_launchrec ; 


};


#endif
