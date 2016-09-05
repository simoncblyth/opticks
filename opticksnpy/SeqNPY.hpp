#pragma once

#include <vector>
template <typename T> class NPY ; 

#include "NPY_API_EXPORT.hh"

class NPY_API SeqNPY {
       static const unsigned N ; 
   public:  
       SeqNPY(NPY<unsigned long long>* sequence); // weak reference to NPY* only
       void dump(const char* msg="SeqNPY::dump");
       int getCount(unsigned code);
       std::vector<int> getCounts();
  private:
       void init();
       void countPhotons();
  private:
        NPY<unsigned long long>* m_sequence ; 
        int*                     m_counts ; 
 
};



