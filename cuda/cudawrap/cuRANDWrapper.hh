#ifndef CURANDWRAPPER_H
#define CURANDWRAPPER_H

class LaunchSequence ; 

#include <vector>


class cuRANDWrapper {
  public: 
     cuRANDWrapper
           (
             LaunchSequence* launchseq,
             unsigned long long seed=0, 
             unsigned long long offset=0 
           ) 
           :
             m_launchseq(launchseq),
             m_seed(seed),
             m_offset(offset),
             m_dev_rng_states(0),
             m_host_rng_states(0),
             m_test(0),
             m_imod(100000),
             m_cache_dir(0)
           {
              setCacheDir("/tmp"); 
           }
     virtual ~cuRANDWrapper();
 

     unsigned int getSeed(){   return m_seed ; }
     unsigned int getOffset(){ return m_offset ; }
     LaunchSequence* getLaunchSequence(){ return m_launchseq ; }
     unsigned int getItems();

     void Summary(const char* msg);
     void Dump(const char* msg="cuRANDWrapper::Dump", unsigned int imod=1000);
     char* digest();
     char* testdigest();

     void setDevRngStates(void* dev_rng_states)
     {
         m_dev_rng_states = dev_rng_states ;
     } 
     void setImod(unsigned int imod)
     {
         m_imod = imod ; 
     }


     char* getCachePath();
     void setCacheDir(const char* dir);

  public:
     void Setup(bool create=false);

     int hasCache();
     int Init(bool create=false);
     int Save();
     int Load(bool roundtrip=false);
     int Test();

  private:
     int hasCache(const char* path);
     int Save(const char* path);
     int Load(const char* path);

  private: 
     void create_rng();
     void init_rng(const char* tag="init_rng");
     void copytohost_rng();
     void copytodevice_rng();
     void test_rng(const char* tag="test_rng");

  private:
     unsigned long long m_seed ;
     unsigned long long m_offset ;

  private:
     void* m_dev_rng_states ;
     void* m_host_rng_states ;
     float* m_test ;
     LaunchSequence* m_launchseq ; 
     unsigned int m_imod ;
     char* m_cache_dir ; 

     std::vector<LaunchSequence*> m_launchrec ; 


};


#endif
