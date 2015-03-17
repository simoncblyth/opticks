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
             m_imod(100000)
           {}
 
     virtual ~cuRANDWrapper(){}

     unsigned int getSeed(){   return m_seed ; }
     unsigned int getOffset(){ return m_offset ; }
     LaunchSequence* getLaunchSequence(){ return m_launchseq ; }

     void Summary(const char* msg);
     void Dump(const char* msg="cuRANDWrapper::Dump", unsigned int imod=1000);
     char* digest();

     void Save(const char* path);
     void Load(const char* path);

     void setLaunchSequence(LaunchSequence* launchseq)
     {
         m_launchseq = launchseq ; 
     } 
     void setRngStates(void* dev_rng_states)
     {
         m_dev_rng_states = dev_rng_states ;
     } 
     void setImod(unsigned int imod)
     {
         m_imod = imod ; 
     }

     void init_rng(const char* tag="init_rng");
     void test_rng(const char* tag="test_rng");


     void create_rng();
     void copytohost_rng();
     void copytodevice_rng();

  private:
     unsigned long long m_seed ;
     unsigned long long m_offset ;

  private:
     void* m_dev_rng_states ;
     void* m_host_rng_states ;
     float* m_test ;
     LaunchSequence* m_launchseq ; 
     unsigned int m_imod ;

     std::vector<LaunchSequence*> m_launchrec ; 


};


#endif
