#pragma once

#include "glm/fwd.hpp"

#include <map>
#include <string>
#include <vector>

#include "Types.hpp"
#include "NPY.hpp"


class BoundariesNPY {
   public:  
       BoundariesNPY(NPY<float>* photons); 
   public:
       void setTypes(Types* types);
       void setBoundaryNames(std::map<int, std::string> names);    

   public:
       // signed mode : signs the boundary code according to the sign of (2,0) vpol.x (currently cos_theta)
       void indexBoundaries(bool sign=true);
       void dump(const char* msg="BoundariesNPY::dump");

   private:  
       static bool second_value_order(const std::pair<int,int>&a, const std::pair<int,int>&b);

   public:  
       glm::ivec4                                    getSelection();             // 1st four boundary codes provided by the selection
       bool*                                         getBoundariesSelection(); 
       std::vector< std::pair<int, std::string> >&   getBoundaries(); 

   protected:
       NPY<float>*                                  m_photons ; 
       std::map<int, std::string>                   m_names ; 
       Types*                                       m_types ; 
       std::vector< std::pair<int, std::string> >   m_boundaries ; 
       bool*                                        m_boundaries_selection ; 

};


inline BoundariesNPY::BoundariesNPY(NPY<float>* photons) 
       :  
       m_photons(photons),
       m_types(NULL),
       m_boundaries_selection(NULL)
{
}
inline void BoundariesNPY::setTypes(Types* types)
{  
    m_types = types ; 
}
inline void BoundariesNPY::setBoundaryNames(std::map<int, std::string> names)
{
    m_names = names ; 
}
inline bool* BoundariesNPY::getBoundariesSelection()
{
    return m_boundaries_selection ; 
}
inline std::vector< std::pair<int, std::string> >& BoundariesNPY::getBoundaries()
{
    return m_boundaries ; 
}



 
