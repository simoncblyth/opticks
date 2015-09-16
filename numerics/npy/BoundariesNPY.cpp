#include "BoundariesNPY.hpp"

#include <glm/glm.hpp>
#include "limits.h"
#include "GLMFormat.hpp"
#include "GLMPrint.hpp"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal



bool BoundariesNPY::second_value_order(const std::pair<int,int>&a, const std::pair<int,int>&b)
{
    return a.second > b.second ;
}


void BoundariesNPY::indexBoundaries()
{
    assert(m_photons);
    m_boundaries.clear();

    printf("BoundariesNPY::indexBoundaries \n");
    std::map<int,int> uniqn = m_photons->count_uniquei(3,0) ;  // boundary signing now done by OptiX

    // To allow sorting by count
    //      map<boundary_code, count> --> vector <pair<boundary_code,count>>

    std::vector<std::pair<int,int> > pairs ; 
    for(std::map<int,int>::iterator it=uniqn.begin() ; it != uniqn.end() ; it++) pairs.push_back(*it);
    std::sort(pairs.begin(), pairs.end(), second_value_order );

    for(unsigned int i=0 ; i < pairs.size() ; i++)
    {
        std::pair<int,int> p = pairs[i]; 
        int code = p.first ;
        std::string name ;
        if(m_names.count(abs(code)) > 0) name = m_names[abs(code)] ; 

        char line[128] ;
        snprintf(line, 128, " %3d : %7d %s ", p.first, p.second, name.c_str() );
        m_boundaries.push_back( std::pair<int, std::string>( code, line ));
    }   

    delete m_boundaries_selection ; 
    m_boundaries_selection = m_types->initBooleanSelection(m_boundaries.size());
}


void BoundariesNPY::dump(const char* msg)
{
    printf("%s\n", msg);
    for(unsigned int i=0 ; i < m_boundaries.size() ; i++)
    {
         std::pair<int, std::string> p = m_boundaries[i];
         printf(" %2d : %s \n", p.first, p.second.c_str() );
    }
}


glm::ivec4 BoundariesNPY::getSelection()
{
    // ivec4 containing 1st four boundary codes provided by the selection

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
    glm::ivec4 iv(-INT_MAX,-INT_MAX,-INT_MAX,-INT_MAX);   // zero tends to be meaningful, so bad default for "unset"
    if(count > 0) iv.x = v[0] ;
    if(count > 1) iv.y = v[1] ;
    if(count > 2) iv.z = v[2] ;
    if(count > 3) iv.w = v[3] ;
    return iv ;     
}


