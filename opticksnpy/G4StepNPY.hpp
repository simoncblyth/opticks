#pragma once

#include <set>
#include <map>
#include <vector>
template <typename T> class NPY ; 
class NLookup ; 

//
// hmm CerenkovStep and ScintillationStep have same shapes but different meanings see
//     /usr/local/env/chroma_env/src/chroma/chroma/cuda/cerenkov.h
//     /usr/local/env/chroma_env/src/chroma/chroma/cuda/scintillation.h
//
//  but whats needed for visualization should be in the same locations ?
//
//
// resist temptation to use inheritance here, 
// it causes much grief for little benefit 
// instead if needed use "friend class" status to 
// give G4StepNPY access to innards of NPY
//
 

#include "NPY_API_EXPORT.hh"

class NPY_API G4StepNPY {
   public:  
        typedef std::set<unsigned int> Set_t ; 
   public:  
       G4StepNPY(NPY<float>* npy); // weak reference to NPY* only
       NPY<float>* getNPY();
   public:  
       void relabel(int cerenkov_label, int scintillation_label);
       void checklabel(int xlabel, int ylabel=-1);
       void checkCounts(std::vector<int>& counts, const char* msg="G4StepNPY::checkCounts");
   public:  
       void countPhotons();
       int getNumPhotons(int label);
       int getNumPhotons();
       void Summary(const char* msg="G4StepNPY::Summary");
   public:  
       void setLookup(NLookup* lookup);
       NLookup* getLookup();
       void applyLookup(unsigned int jj, unsigned int kk);
       void dump(const char* msg);
       void dumpLines(const char* msg);
   public:  
       int  getStepId(unsigned int i=0);
       //bool isCerenkovStep(unsigned int i=0);
       //bool isScintillationStep(unsigned int i=0);
  private:
       // the heart of the lookup:  int bcode = m_lookup->a2b(acode) ;
       bool applyLookup(unsigned int index);
  private:
        NPY<float>*  m_npy ; 
        NLookup*  m_lookup ; 
        Set_t    m_lines ;
        std::map<int, int> m_photons ; 
        int                m_total_photons ; 
 
};



