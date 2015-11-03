#pragma once

#include "glm/fwd.hpp"

#include <map>
#include <string>
#include <vector>

#include "Types.hpp"
#include "NPY.hpp"


// host based indexing of photon data 

class BoundariesNPY {
   public:  
       BoundariesNPY(NPY<float>* photons); 
   public:
       void setTypes(Types* types);
       void setBoundaryNames(std::map<unsigned int, std::string> names);     // just used to give a name to a boundary code

   public:
       // boundary code integer is cos_theta signed by OptiX in cu/material1_propagate.cu
       void indexBoundaries();
       void dump(const char* msg="BoundariesNPY::dump");

   private:  
       static bool second_value_order(const std::pair<int,int>&a, const std::pair<int,int>&b);

   public:  
       glm::ivec4                                    getSelection();             // 1st four boundary codes provided by the selection
       //bool*                                         getBoundariesSelection(); 
       std::vector< std::pair<int, std::string> >&   getBoundaries(); 

   protected:
       NPY<float>*                                  m_photons ; 
       std::map<unsigned int, std::string>          m_names ; 
       Types*                                       m_types ; 
       std::vector< std::pair<int, std::string> >   m_boundaries ; 
       unsigned int                                 m_total ; 

       //bool*                                        m_boundaries_selection ; 

};


inline BoundariesNPY::BoundariesNPY(NPY<float>* photons) 
       :  
       m_photons(photons),
       m_total(0)
    //   m_types(NULL),
    //   m_boundaries_selection(NULL)
{
}
inline void BoundariesNPY::setTypes(Types* types)
{  
    m_types = types ; 
}

inline void BoundariesNPY::setBoundaryNames(std::map<unsigned int, std::string> names)
{
    m_names = names ; 
}
//inline bool* BoundariesNPY::getBoundariesSelection()
//{
//    return m_boundaries_selection ; 
//}
inline std::vector< std::pair<int, std::string> >& BoundariesNPY::getBoundaries()
{
    return m_boundaries ; 
}



 
