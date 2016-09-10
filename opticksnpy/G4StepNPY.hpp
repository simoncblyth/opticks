#pragma once

#include <set>
#include <map>
#include <vector>
#include <string>
template <typename T> class NPY ; 
class NLookup ; 

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
       unsigned getNumSteps();
       unsigned getNumPhotons(unsigned step);
       unsigned getGencode(unsigned step);
       unsigned getNumPhotonsTotal();
       unsigned* makePhotonSeedArray();  // genstep id for each photon
   public:  
       void countPhotons();
       int getNumPhotonsCounted(int label);
       int getNumPhotonsCounted();
       std::string description();
       void Summary(const char* msg="G4StepNPY::Summary");
   public:  
       void setLookup(NLookup* lookup);
       NLookup* getLookup();
       void applyLookup(unsigned int jj, unsigned int kk);
       void dump(const char* msg);
       void dumpLines(const char* msg);
   public:  
       int  getStepId(unsigned int i=0);
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



