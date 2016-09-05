#pragma once

#include <vector>
template <typename T> class NPY ; 

#include "NPY_API_EXPORT.hh"

//
// Hmm : would be better to use the GPU derived
//       indices rather than going back to the raw sequence
//       which requires sequence data copied back to host 
//
// BUT: this is handy as a check anyhow
//

class NPY_API SeqNPY {
       static const unsigned N ; 
   public:  
       SeqNPY(NPY<unsigned long long>* sequence); 
       virtual ~SeqNPY();

       void dump(const char* msg="SeqNPY::dump");
       int getCount(unsigned code);
       std::vector<int> getCounts();
  private:
       void init();
       void countPhotons();
  private:
        NPY<unsigned long long>* m_sequence ;  // weak 
        int*                     m_counts ; 
 
};



