
#include <sstream>
#include <iostream>
#include <iomanip>
#include <climits>


#include "NGLM.hpp"
#include "GLMFormat.hpp"
#include "GLMPrint.hpp"
#include "Types.hpp"

#include "BoundariesNPY.hpp"

#include "PLOG.hh"


BoundariesNPY::BoundariesNPY(NPY<float>* photons) 
       :  
       m_photons(photons),
       m_total(0)
{
}
void BoundariesNPY::setTypes(Types* types)
{  
    m_types = types ; 
}
void BoundariesNPY::setBoundaryNames(std::map<unsigned int, std::string> names)
{
    m_names = names ; 
}
std::vector< std::pair<int, std::string> >& BoundariesNPY::getBoundaries()
{
    return m_boundaries ; 
}






bool BoundariesNPY::second_value_order(const std::pair<int,int>&a, const std::pair<int,int>&b)
{
    return a.second > b.second ;
}

void BoundariesNPY::indexBoundaries()
{
    assert(m_photons);
    m_boundaries.clear();

    printf("BoundariesNPY::indexBoundaries \n");

    std::vector<std::pair<int,int> > pairs = m_photons->count_uniquei_descending(3,0) ;

    m_total = 0 ; 

    for(unsigned int i=0 ; i < pairs.size() ; i++)
    {
        std::pair<int,int> p = pairs[i]; 
        int code = p.first ;
        int count = p.second ;
        m_total += count ;   

        std::string name ;
        if(m_names.count(abs(code)) > 0) name = m_names[abs(code)] ; 

        std::stringstream ss ; 
        ss << std::setw(5)  << code 
           << std::setw(10) << count
           << std::setw(60) << name ;

        std::string line = ss.str();
        m_boundaries.push_back( std::pair<int, std::string>( code, line ));
    }   


    //if(m_types)
    //{
    //    delete m_boundaries_selection ; 
    //    m_boundaries_selection = m_types->initBooleanSelection(m_boundaries.size());
    //}
}


void BoundariesNPY::dump(const char* msg)
{
    LOG(info) << msg << " total : " << m_total  ; 

    for(unsigned int i=0 ; i < m_boundaries.size() ; i++)
    {
         std::pair<int, std::string> p = m_boundaries[i];
         std::cout << std::setw(5) << p.first 
                   << " : "
                   << p.second  
                   << std::endl ; 
    }
}


glm::ivec4 BoundariesNPY::getSelection()
{
    // ivec4 containing 1st four boundary codes provided by the selection

    /*
    int v[4] ;
    unsigned int count(0) ; 
    for(unsigned int i=0 ; i < m_boundaries.size() ; i++)
    {
        if(m_boundaries_selection[i])
        {
            std::pair<int, std::string> p = m_boundaries[i];
            if(count < 4)
            {
                v[count] = p.first ; 
                count++ ; 
            }
            else
            {
                 break ;
            }
        }
    } 
    */
 
    glm::ivec4 iv(-INT_MAX,-INT_MAX,-INT_MAX,-INT_MAX);   // zero tends to be meaningful, so bad default for "unset"
    //if(count > 0) iv.x = v[0] ;
    //if(count > 1) iv.y = v[1] ;
    //if(count > 2) iv.z = v[2] ;
    //if(count > 3) iv.w = v[3] ;
    return iv ;     
}

