#include "PhotonsNPY.hpp"
#include "uif.h"
#include "NPY.hpp"

#include <set>
#include <map>

#include <glm/glm.hpp>
#include "limits.h"

#include "regexsearch.hh"


bool value_order(const std::pair<int,int>&a, const std::pair<int,int>&b)
{
    return a.second > b.second ;
}


void PhotonsNPY::classify(bool sign)
{
    m_boundaries.clear();
    m_boundaries = findBoundaries(sign);
    delete m_boundaries_selection ; 
    m_boundaries_selection = initBooleanSelection(m_boundaries.size());
    dumpBoundaries("PhotonsNPY::classify");
}

bool* PhotonsNPY::initBooleanSelection(unsigned int n)
{
    bool* selection = new bool[n];
    while(n--) selection[n] = false ; 
    return selection ;
}

glm::ivec4 PhotonsNPY::getSelection()
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



void PhotonsNPY::dumpBoundaries(const char* msg)
{
    printf("%s\n", msg);
    for(unsigned int i=0 ; i < m_boundaries.size() ; i++)
    {
         std::pair<int, std::string> p = m_boundaries[i];
         printf(" %2d : %s \n", p.first, p.second.c_str() );
    }
}


std::vector<std::pair<int, std::string> > PhotonsNPY::findBoundaries(bool sign)
{
    assert(m_npy);

    std::vector<std::pair<int, std::string> > boundaries ;  

    printf("PhotonsNPY::classify \n");


    std::map<int,int> uniqn = sign ? m_npy->count_uniquei(3,0,2,0) : m_npy->count_uniquei(3,0) ;

    // To allow sorting by count
    //      map<boundary_code, count> --> vector <pair<boundary_code,count>>

    std::vector<std::pair<int,int> > pairs ; 
    for(std::map<int,int>::iterator it=uniqn.begin() ; it != uniqn.end() ; it++) pairs.push_back(*it);
    std::sort(pairs.begin(), pairs.end(), value_order );


    for(unsigned int i=0 ; i < pairs.size() ; i++)
    {
        std::pair<int,int> p = pairs[i]; 
        int code = p.first ;
        std::string name ;
        if(m_names.count(abs(code)) > 0) name = m_names[abs(code)] ; 

        char line[128] ;
        snprintf(line, 128, " %3d : %7d %s ", p.first, p.second, name.c_str() );
        boundaries.push_back( std::pair<int, std::string>( code, line ));
    }   

    return boundaries ;
}


void PhotonsNPY::dump(const char* msg)
{
    if(!m_npy) return ;
    printf("%s\n", msg);

    unsigned int ni = m_npy->m_len0 ;
    unsigned int nj = m_npy->m_len1 ;
    unsigned int nk = m_npy->m_len2 ;
    std::vector<float>& data = m_npy->m_data ; 

    printf(" ni %u nj %u nk %u nj*nk %u \n", ni, nj, nk, nj*nk ); 

    uif_t uif ; 

    unsigned int check = 0 ;
    for(unsigned int i=0 ; i<ni ; i++ ){
    for(unsigned int j=0 ; j<nj ; j++ )
    {
       bool out = i == 0 || i == ni-1 ; 
       if(out) printf(" (%7u,%1u) ", i,j );
       for(unsigned int k=0 ; k<nk ; k++ )
       {
           unsigned int index = i*nj*nk + j*nk + k ;
           if(out)
           {
               uif.f = data[index] ;
               if( j == 3 && k == 0 ) printf(" %15d ",   uif.i );
               else                   printf(" %15.3f ", uif.f );
           }
           assert(index == check);
           check += 1 ; 
       }
       if(out)
       {
           if( j == 0 ) printf(" position/time ");
           if( j == 1 ) printf(" direction/wavelength ");
           if( j == 2 ) printf(" polarization/weight ");
           if( j == 3 ) printf(" boundary/cos_theta/distance_to_boundary/- ");

           printf("\n");
       }
    }
    }
}

void PhotonsNPY::readFlags(const char* path)
{
    enum_regexsearch( m_flags, path);
}

void PhotonsNPY::dumpFlags(const char* msg)
{
    printf("%s\n", msg);
    for(unsigned int i=0 ; i < m_flags.size() ; i++)
    {
         std::pair<int, std::string> p = m_flags[i];
         printf(" %10d : %10x :  %s \n", p.first, p.first,  p.second.c_str() );
    }
}











