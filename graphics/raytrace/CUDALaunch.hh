#ifndef CUDALAUNCH_H
#define CUDALAUNCH_H

#include <vector>

class CUDALaunch {
  public:
      static std::vector<CUDALaunch> Make(unsigned int nwork, unsigned int threads_per_block=64, unsigned int max_blocks=1024 ) 
      {

      }


      CUDALaunch(unsigned int first, unsigned int elements, unsigned int blocks ) : 
           m_first(first),
           m_elements(elements),
           m_blocks(blocks) {}

      virtual ~CUDALaunch();

  private:
      unsigned int m_first ;
      unsigned int m_elements ;
      unsigned int m_blocks ;


};



#endif
