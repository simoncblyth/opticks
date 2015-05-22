#pragma once

#include "glm/fwd.hpp"

#include <map>
#include <string>
#include <vector>


class NPY ;

class PhotonsNPY {
   public:  
       PhotonsNPY(NPY* npy); // weak reference to NPY* only
       NPY* getNPY();

       void setBoundaryNames(std::map<int, std::string> names);    
       void classify();

       glm::ivec4 getSelection();
       //std::vector<int>&          getBoundariesSelection(); 
       //std::vector<bool>&         getBoundariesSelection(); 
       bool*         getBoundariesSelection(); 

       std::vector< std::pair<int, std::string> >& getBoundaries(); 

   private:
       //std::vector<int>  initIntegerSelection(unsigned int n);
       //std::vector<bool> initBooleanSelection(unsigned int n);
       bool* initBooleanSelection(unsigned int n);

       std::vector< std::pair<int, std::string> > findBoundaries();
       void dumpBoundaries(const char* msg);

   public:  
       void dump(const char* msg);

  private:
        NPY*                       m_npy ; 
        std::map<int, std::string> m_names ; 
        std::vector< std::pair<int, std::string> > m_boundaries ; 
        //std::vector< int >                         m_boundaries_selection ; 
        //std::vector<bool>                          m_boundaries_selection ; 
        bool*                                        m_boundaries_selection ; 
 
};



inline PhotonsNPY::PhotonsNPY(NPY* npy) 
       :  
       m_npy(npy)
{
}

inline NPY* PhotonsNPY::getNPY()
{
    return m_npy ; 
}

inline void PhotonsNPY::setBoundaryNames(std::map<int, std::string> names)
{
    m_names = names ; 
}


//std::vector<bool>& PhotonsNPY::getBoundariesSelection()
bool* PhotonsNPY::getBoundariesSelection()
{
    return m_boundaries_selection ; 
}


std::vector< std::pair<int, std::string> >& PhotonsNPY::getBoundaries()
{
    return m_boundaries ; 
}



