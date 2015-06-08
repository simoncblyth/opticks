#pragma once

#include "glm/fwd.hpp"

#include <map>
#include <string>
#include <vector>

#include "NPY.hpp"

class PhotonsNPY {
   public:  
       typedef std::vector< std::pair<int, std::string> >  Choices_t ; 

       PhotonsNPY(NPY<float>* npy); // weak reference to NPY* only
       NPY<float>* getNPY();

       // boundary names corresponding to absolute integer codes 
       // TODO: offset codes by one to avoid confusion regarding sign of Vacuum/Vacuum 0 

       void setBoundaryNames(std::map<int, std::string> names);    

       // signed mode : signs the boundary code according to the sign of (2,0) vpol.x (currently cos_theta)
       void classify(bool sign=false);

   public:
       void readFlags(const char* path); // parse enum flags from photon.h
       void dumpFlags(const char* msg="PhotonsNPY::dumpFlags");
       glm::ivec4 getFlags();

   public:  
       // ivec4 containing 1st four boundary codes provided by the selection
       glm::ivec4                                  getSelection();

   public:  
       // interface to ImGui checkboxes that make the boundary selection
       bool*        getBoundariesSelection(); 
       Choices_t*   getBoundariesPointer(); 
       Choices_t&   getBoundaries(); 

   private:
       bool* initBooleanSelection(unsigned int n);
       Choices_t findBoundaries(bool sign);
       void dumpBoundaries(const char* msg);

   public:  
       void dump(const char* msg);

   private:
       NPY<float>*                  m_npy ; 

   protected:
       std::map<int, std::string>   m_names ; 

       Choices_t                    m_boundaries ; 
       bool*                        m_boundaries_selection ; 

       Choices_t                    m_flags ; 
       bool*                        m_flags_selection ; 
 
};



inline PhotonsNPY::PhotonsNPY(NPY<float>* npy) 
       :  
       m_npy(npy),
       m_boundaries_selection(NULL)
{
}

inline NPY<float>* PhotonsNPY::getNPY()
{
    return m_npy ; 
}

inline void PhotonsNPY::setBoundaryNames(std::map<int, std::string> names)
{
    m_names = names ; 
}


inline bool* PhotonsNPY::getBoundariesSelection()
{
    return m_boundaries_selection ; 
}


inline PhotonsNPY::Choices_t& PhotonsNPY::getBoundaries()
{
    return m_boundaries ; 
}

inline PhotonsNPY::Choices_t* PhotonsNPY::getBoundariesPointer()
{
    return &m_boundaries ; 
}



